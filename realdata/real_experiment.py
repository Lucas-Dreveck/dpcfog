"""Layer 1: the filters on real trust networks.

    python -m realdata.real_experiment otc alpha epinions

Produces the same CSV schema as the synthetic experiment. Columns that only make
sense in simulation are filled with what the real network offers: `mu` is the
rate of labelled fraudsters, `attack` is "real", `seed` is 0 because nothing is
drawn.

The metric only looks at ratings by labelled users. The filter processes all of
them, and they affect consensus, median and reputation as they would in
operation; they are just not counted as hits or misses, because nobody knows
what they are.
"""

from __future__ import annotations

import csv
import platform
import time
from pathlib import Path
from typing import Iterable

import numpy as np

from dpcfog import __version__, metrics, trust
from dpcfog.experiment import COLUMNS, filter_catalog
from dpcfog.filters import DPCFilter

from .batches import build_batches, reference_trust_score, split_by_time
from .networks import NETWORKS, load

MACHINE = f"{platform.system()} {platform.machine()} py{platform.python_version()}"


def run_network(
    network: str = "otc",
    filter_names: Iterable[str] | None = None,
    window: int = 5,
    n_fog: int = 1,
    measure_ram: bool = True,
    rounds_per_year: int = 12,
) -> list[dict]:
    net = load(network, rounds_per_year=rounds_per_year)
    batches = build_batches(net, window=window, n_fog=n_fog)
    train, evaluation = split_by_time(batches)
    if not evaluation:
        return []

    truth = np.concatenate([b.label for b in evaluation])
    mask = np.concatenate([b.evaluable for b in evaluation])
    target = np.concatenate([b.target for b in evaluation])
    value = np.concatenate([b.value for b in evaluation])

    # H2 on real data. The reference only exists for targets some benign user
    # rated, and the evaluation period sees only part of the network: even
    # Without a filter, 2,633 of the 5,858 Bitcoin OTC targets have no evidence
    # and fall to the 0.5 prior. An MAE over all of them would measure the
    # coverage of the period, not the filtering. The comparison therefore runs
    # on the intersection: targets with a reference and at least one accepted
    # rating in both conditions. The lost coverage goes to
    # `nodes_without_evidence`.
    t_ref, has_ref = reference_trust_score(net)
    everything = np.ones(value.size, dtype=bool)
    t_unf, no_evid_base = trust.trust_score(target, value, everything, net.n_targets)
    comparable_base = has_ref & ~no_evid_base

    factories = filter_catalog(0)
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
            sample = evaluation[: min(5, len(evaluation))]
            other = factory()
            if other.supervised:
                other.fit(train)
            _, peak = metrics.measure_memory(
                lambda f=other: [f.flag(b) for b in sample]
            )

        d = metrics.evaluate(predicted[mask], truth[mask])

        # Scoring pass on a fresh instance: stateful filters (reputation) must
        # not see the same batches twice.
        try:
            scorer = factory()
            if scorer.supervised:
                scorer.fit(train)
            parts = [scorer.score(b) for b in evaluation]
            rank = (
                metrics.ranking(np.concatenate(parts)[mask], truth[mask])
                if all(x is not None for x in parts)
                else {"auc": float("nan"), "average_precision": float("nan")}
            )
        except Exception as error:
            print(f"    [warning] score unavailable: {error}", flush=True)
            rank = {"auc": float("nan"), "average_precision": float("nan")}

        t_f, no_evid = trust.trust_score(
            target, value, ~predicted, net.n_targets
        )
        comparable = comparable_base & ~no_evid
        mae_f = trust.mae(t_f[comparable], t_ref[comparable])
        mae_unf_pair = trust.mae(t_unf[comparable], t_ref[comparable])

        dcs, clusters = [], []
        if isinstance(flt, DPCFilter):
            diag = flt.diagnostics()
            if diag:
                dcs.append(diag["dc"])
                clusters.append(diag["n_clusters"])

        rows.append(
            {
                "scenario": f"real_{network}",
                "attack": "real",
                "mu": round(float(net.label[net.evaluable].mean()), 4),
                "seed": 0,
                "n_nodes": net.n_raters,
                "k_targets": round(len(net) / net.n_raters, 2),
                "n_rounds": int(net.round_idx.max()) + 1,
                "n_fog": n_fog,
                "consensus_window": window,
                "filter": name,
                "variant": "",
                "tp": d.true_positive, "fp": d.false_positive,
                "fn": d.false_negative, "tn": d.true_negative,
                "precision": round(d.precision, 6), "recall": round(d.recall, 6),
                "f1": round(d.f1, 6), "accuracy": round(d.accuracy, 6),
                "auc": round(rank["auc"], 6),
                "average_precision": round(rank["average_precision"], 6),
                # Layer 1: no injection, so no second label convention.
                "auc_recommendation": "",
                "average_precision_recommendation": "",
                "malicious_rate_recommendation": "",
                "mae_filtered": round(mae_f, 6),
                "mae_unfiltered": round(mae_unf_pair, 6),
                "mae_reduction": round(mae_unf_pair - mae_f, 6),
                # comparable targets lost because of the filter, not of the period
                "nodes_without_evidence": int(comparable_base.sum() - comparable.sum()),
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


def sweep_real(
    networks: Iterable[str] = ("otc", "alpha"),
    output: str | Path = "results/real.csv",
    verbose: bool = True,
) -> Path:
    """Run the filters on the requested networks and consolidate the CSV.

    Each network goes to its own file, and `real.csv` joins all that exist, so
    running a single network never erases the others.
    """
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    for network in networks:
        partial = path.parent / f"real_{network}.csv"
        with partial.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=COLUMNS)
            writer.writeheader()
            if verbose:
                print(f"--- {network}", flush=True)
            # Epinions needs daily rounds and a Fog partition: with daily rounds
            # and a single node, the mean batch has 9,575 points and the
            # measured peak is 700 MB, well above the 256 MB budget of a
            # Raspberry Pi 3.
            extra = (
                dict(rounds_per_year=365, n_fog=8)
                if network in ("epinions", "amazon")
                else {}
            )
            for row in run_network(network, **extra):
                writer.writerow(row)
                if verbose:
                    print(
                        f"    {row['filter']:18s} "
                        f"AP={row['average_precision']:.3f} "
                        f"AUC={row['auc']:.3f} F1={row['f1']:.3f} "
                        f"discard={row['discard_rate']:.3f} "
                        f"{row['mean_batch_time_ms']:.2f}ms",
                        flush=True,
                    )

    consolidate(path)
    return path


def consolidate(path: Path) -> Path:
    """Join the existing `real_*.csv` files into the consolidated one."""
    parts = sorted(path.parent.glob("real_*.csv"))
    rows: list[dict] = []
    for part in parts:
        with part.open(encoding="utf-8") as fh:
            rows.extend(csv.DictReader(fh))
    if not rows:
        return path
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"consolidated: {path} ({len(rows)} rows from {len(parts)} networks)")
    return path


if __name__ == "__main__":
    import sys

    requested = sys.argv[1:] or ["otc", "alpha"]
    print(sweep_real([n for n in requested if n in NETWORKS]))
