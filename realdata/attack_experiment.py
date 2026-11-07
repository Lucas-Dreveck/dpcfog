"""Layer 2: sweep of mu with published attacks on a real topology.

    python -m realdata.attack_experiment otc

It is the only place in the work with both a **realistic background** and a
**complete ground truth**.

| Layer | Background | Ground truth | Answers |
|---|---|---|---|
| 1, real | real | partial (23% labelled) | does it work against real fraud? |
| 2, real + injection | real | complete | does it work against a known attack, and how does it vary with mu? |
| 3, simulation | synthetic | complete | isolates the effect of each parameter |
"""

from __future__ import annotations

import csv
import itertools
import platform
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from dpcfog import __version__, metrics, trust
from dpcfog.experiment import COLUMNS, filter_catalog
from dpcfog.filters import DPCFilter

from .attacks import EQUIVALENCE, Attack, inject
from .batches import build_batches, split_by_time
from .fine_label import label_per_recommendation
from .networks import load

MACHINE = f"{platform.system()} {platform.machine()} py{platform.python_version()}"

MU_GRID = (0.10, 0.20, 0.30, 0.40)
SEED_GRID = (0, 1, 2)


def run(
    network: str,
    attack: Attack,
    filter_names: Iterable[str] | None = None,
    window: int = 5,
    n_fog: int = 1,
    measure_ram: bool = True,
) -> list[dict]:
    net = inject(load(network), attack)
    batches = build_batches(net, window=window, n_fog=n_fog)
    train, evaluation = split_by_time(batches)
    if not evaluation:
        return []

    truth = np.concatenate([b.label for b in evaluation])
    mask = np.concatenate([b.evaluable for b in evaluation])
    # Second ground truth, per recommendation: only the distorted rating is
    # positive. 88.4% of the injected ratings are filler drawn from the real
    # distribution, indistinguishable from honest ratings by construction.
    fine = label_per_recommendation(net)
    fine_key = set(zip(net.rater[fine].tolist(), net.target[fine].tolist()))
    truth_rec = np.concatenate([
        np.array([(r, a) in fine_key for r, a in zip(b.rater, b.target)],
                 dtype=bool)
        for b in evaluation
    ])
    target = np.concatenate([b.target for b in evaluation])
    value = np.concatenate([b.value for b in evaluation])

    # Trust Score reference in layer 2: the mean of the real ratings about each
    # target, before injection. It exists because we know exactly what was added.
    clean = load(network)
    total = np.zeros(clean.n_targets)
    count = np.zeros(clean.n_targets)
    np.add.at(total, clean.target, clean.value)
    np.add.at(count, clean.target, 1.0)
    has_ref = count > 0
    t_ref = np.where(has_ref, total / np.where(has_ref, count, 1.0), np.nan)

    t_unf, no_evid_base = trust.trust_score(
        target, value, np.ones(value.size, dtype=bool), clean.n_targets
    )
    comparable_base = has_ref & ~no_evid_base

    factories = filter_catalog(attack.seed)
    chosen = list(filter_names) if filter_names else list(factories)

    rows: list[dict] = []
    for name in chosen:
        factory = factories[name]
        flt = factory()

        t0 = time.perf_counter()
        if flt.supervised:
            flt.fit(train)
        train_time = time.perf_counter() - t0

        predictions, total_time = metrics.measure_time(
            lambda f=flt: [f.flag(b) for b in evaluation]
        )
        predicted = np.concatenate(predictions)

        peak = 0
        if measure_ram:
            other = factory()
            if other.supervised:
                other.fit(train)
            _, peak = metrics.measure_memory(
                lambda f=other: [f.flag(b) for b in evaluation[:5]]
            )

        d = metrics.evaluate(predicted[mask], truth[mask])

        try:
            scorer = factory()
            if scorer.supervised:
                scorer.fit(train)
            parts = [scorer.score(b) for b in evaluation]
            rank = (
                metrics.ranking(
                    np.concatenate(parts)[mask], truth[mask]
                )
                if all(x is not None for x in parts)
                else {"auc": float("nan"), "average_precision": float("nan")}
            )
            rank_rec = (
                metrics.ranking(
                    np.concatenate(parts)[mask], truth_rec[mask]
                )
                if all(x is not None for x in parts)
                else {"auc": float("nan"), "average_precision": float("nan")}
            )
        except Exception as error:
            print(f"    [warning] score unavailable: {error}", flush=True)
            rank = {"auc": float("nan"), "average_precision": float("nan")}
            rank_rec = {"auc": float("nan"), "average_precision": float("nan")}

        t_f, no_evid = trust.trust_score(
            target, value, ~predicted, clean.n_targets
        )
        comp = comparable_base & ~no_evid
        mae_f = trust.mae(t_f[comp], t_ref[comp])
        mae_unf = trust.mae(t_unf[comp], t_ref[comp])

        dcs, clusters = [], []
        if isinstance(flt, DPCFilter):
            diag = flt.diagnostics()
            if diag:
                dcs.append(diag["dc"])
                clusters.append(diag["n_clusters"])

        rows.append(
            {
                "scenario": f"attack_{network}",
                "attack": attack.name,
                "mu": attack.attack_size,
                "seed": attack.seed,
                "n_nodes": net.n_raters,
                "k_targets": round(len(net) / net.n_raters, 2),
                "n_rounds": int(net.round_idx.max()) + 1,
                "n_fog": n_fog,
                "consensus_window": window,
                "filter": name,
                "variant": (
                    "realistic" if attack.realistic_filler else "literature"
                ),
                "tp": d.true_positive, "fp": d.false_positive,
                "fn": d.false_negative, "tn": d.true_negative,
                "precision": round(d.precision, 6), "recall": round(d.recall, 6),
                "f1": round(d.f1, 6), "accuracy": round(d.accuracy, 6),
                "auc": round(rank["auc"], 6),
                "average_precision": round(rank["average_precision"], 6),
                "auc_recommendation": round(rank_rec["auc"], 6),
                "average_precision_recommendation": round(
                    rank_rec["average_precision"], 6
                ),
                "malicious_rate_recommendation": round(
                    float(truth_rec[mask].mean()), 6
                ),
                "mae_filtered": round(mae_f, 6),
                "mae_unfiltered": round(mae_unf, 6),
                "mae_reduction": round(mae_unf - mae_f, 6),
                "nodes_without_evidence": int(comparable_base.sum() - comp.sum()),
                "discard_rate": round(float(predicted.mean()), 6),
                "mean_batch_time_ms": round(
                    1000 * total_time / len(evaluation), 4
                ),
                "total_time_s": round(total_time, 5),
                "train_time_s": round(train_time, 5),
                "peak_memory_kb": round(peak / 1024.0, 1),
                "n_batches": len(evaluation),
                "n_evaluated": int(mask.sum()),
                "malicious_rate": round(
                    float(truth[mask].mean()), 6
                ),
                "median_dc": round(float(np.median(dcs)), 5) if dcs else "",
                "median_n_clusters": int(np.median(clusters)) if clusters else "",
                "version": __version__, "machine": MACHINE,
            }
        )
    return rows


def sweep(
    network: str = "otc",
    models: Sequence[tuple[str, str]] = tuple(EQUIVALENCE),
    mus: Sequence[float] = MU_GRID,
    seeds: Sequence[int] = SEED_GRID,
    realistic: bool = True,
    output: str | Path | None = None,
    verbose: bool = True,
) -> Path:
    path = Path(output or f"results/attack_{network}.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    combinations = list(itertools.product(models, mus, seeds))
    start = time.perf_counter()

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        for n, ((model, direction), mu, seed) in enumerate(combinations, 1):
            attack = Attack(
                model=model, direction=direction, attack_size=mu,
                realistic_filler=realistic, seed=seed,
            )
            for row in run(network, attack):
                writer.writerow(row)
            fh.flush()
            if verbose:
                elapsed = time.perf_counter() - start
                print(
                    f"[{n:3d}/{len(combinations)}] {attack.name:24s} "
                    f"mu={mu:.2f} seed={seed}  "
                    f"elapsed={elapsed:6.1f}s "
                    f"remaining~{elapsed / n * (len(combinations) - n):6.1f}s",
                    flush=True,
                )
    return path


if __name__ == "__main__":
    import sys

    print(sweep(sys.argv[1] if len(sys.argv) > 1 else "otc"))
