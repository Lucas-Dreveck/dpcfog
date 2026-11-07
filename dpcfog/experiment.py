"""Evaluation loop for the simulation (layer 3).

One run = one (attack, mu, seed) combination. Within it, each Fog node
processes its rounds in batches, and every filter sees exactly the same batches.

Temporal split of the rounds:

    [0, warmup)       discarded: too little history for temporal consistency
    [warmup, cut)     training, used only by the supervised filters
    [cut, R)          evaluation, where every filter is measured

Training precedes evaluation in time. Shuffling rows would give the supervised
model access to the future.

The output is a long-format CSV, one row per (run, filter). No number in the
paper is typed by hand: `paper_tables.py` and `report.py` read the CSVs.
"""

from __future__ import annotations

import csv
import itertools
import platform
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from . import __version__, metrics, trust
from .config import MU_GRID, SEED_GRID, Params
from .features import extract
from .filters import (
    Batch,
    ConsensusFilter,
    DeepTrustFilter,
    DPCFilter,
    Filter,
    HanDPCFilter,
    MLPFilter,
    REV2Filter,
    ReputationFilter,
    TreeFilter,
)
from .generator import Simulation, generate
from .scenarios import apply as apply_scenario

COLUMNS = [
    "scenario", "attack", "mu", "seed", "n_nodes", "k_targets", "n_rounds", "n_fog",
    "consensus_window", "filter", "variant",
    "tp", "fp", "fn", "tn", "precision", "recall", "f1", "accuracy",
    "mae_filtered", "mae_unfiltered", "mae_reduction", "nodes_without_evidence",
    "discard_rate", "mean_batch_time_ms", "total_time_s", "train_time_s",
    "peak_memory_kb", "n_batches", "n_evaluated",
    "auc", "average_precision",
    # Layer 2 only: the same metrics under the per-recommendation label, where
    # only the distorted rating is positive. The default label above is per
    # profile (every rating of a fabricated account). Empty in layers 1 and 3,
    # where the distinction does not exist.
    "auc_recommendation", "average_precision_recommendation",
    "malicious_rate_recommendation",
    "malicious_rate", "median_dc", "median_n_clusters",
    "version", "machine",
]

MACHINE = f"{platform.system()} {platform.machine()} py{platform.python_version()}"


# ------------------------------------------------------------------ batches


def build_batches(sim: Simulation) -> list[Batch]:
    """One batch per (Fog node, round), with features taken from the window."""
    p = sim.params
    batches: list[Batch] = []
    for t in range(p.n_rounds):
        for f in range(p.n_fog):
            batch_idx = sim.batch_indices(f, t)
            if batch_idx.size < 3:
                continue
            window_idx = sim.window_indices(f, t, p.consensus_window)
            x = extract(
                sim.rater, sim.target, sim.value, sim.round_idx, window_idx, batch_idx
            )
            batches.append(
                Batch(
                    x=x,
                    value=sim.value[batch_idx],
                    rater=sim.rater[batch_idx],
                    target=sim.target[batch_idx],
                    round_no=t,
                    fog=f,
                    label=sim.label[batch_idx],
                )
            )
    return batches


def split(batches: Sequence[Batch], p: Params, train_fraction: float = 0.4):
    usable = [b for b in batches if b.round_no >= p.warmup_rounds]
    if not usable:
        return [], []
    rounds = sorted({b.round_no for b in usable})
    cut = rounds[max(1, int(round(train_fraction * len(rounds))))]
    train = [b for b in usable if b.round_no < cut]
    evaluation = [b for b in usable if b.round_no >= cut]
    return train, evaluation


# ------------------------------------------------------------- catalogue


def filter_catalog(seed: int) -> dict[str, Callable[[], Filter]]:
    """Factories, not instances: the reputation filter is stateful and must not
    cross runs."""
    return {
        "dpc": lambda: DPCFilter(dc_criterion="heuristic", dc_fraction=0.02),
        "dpc_entropy": lambda: DPCFilter(dc_criterion="entropy"),
        "dpc_cardinality": lambda: DPCFilter(legitimate_criterion="cardinality"),
        "dpc_recommender": lambda: DPCFilter(granularity="recommender"),
        "dpc_outlier": lambda: DPCFilter(rule="outlier"),
        "dpc_bimodal": lambda: DPCFilter(weights="bimodal"),
        "dpc_han": lambda: HanDPCFilter(),
        "dpc_han_no_id": lambda: HanDPCFilter(use_id_as_x=False),
        "consensus": lambda: ConsensusFilter(threshold=0.25),
        "reputation": lambda: ReputationFilter(alpha=0.8, threshold=0.5),
        "rev2": lambda: REV2Filter(),
        "tree3": lambda: TreeFilter(),
        "deeptrust": lambda: DeepTrustFilter(seed=seed),
        "mlp": lambda: MLPFilter(seed=seed),
    }


# ------------------------------------------------------------- execution


def run_one(
    p: Params,
    filter_names: Iterable[str] | None = None,
    dc_variants: Sequence[float] = (),
    measure_ram: bool = True,
    scenario: str = "basic",
) -> list[dict]:
    sim = generate(p)
    batches = build_batches(sim)
    train, evaluation = split(batches, p)
    if not evaluation:
        return []

    truth = np.concatenate([b.label for b in evaluation])
    target = np.concatenate([b.target for b in evaluation])
    value = np.concatenate([b.value for b in evaluation])

    # H2 reference condition: no filtering at all
    t_unf, _ = trust.trust_score(
        target, value, np.ones(value.size, dtype=bool), p.n_nodes
    )
    mae_unf = trust.mae(t_unf, sim.t_ref)

    factories = filter_catalog(p.seed)
    chosen = list(filter_names) if filter_names else list(factories)
    work: list[tuple[str, str, Callable[[], Filter]]] = [
        (n, "", factories[n]) for n in chosen
    ]
    for fraction in dc_variants:
        work.append(
            (
                "dpc",
                f"dc={fraction:.3f}",
                lambda fr=fraction: DPCFilter(dc_criterion="heuristic", dc_fraction=fr),
            )
        )

    rows: list[dict] = []
    for name, variant, factory in work:
        flt = factory()

        t0 = time.perf_counter()
        if flt.supervised:
            flt.fit(train)
        train_time = time.perf_counter() - t0

        def single_pass(f=flt):
            return [f.flag(b) for b in evaluation]

        predictions, total_time = metrics.measure_time(single_pass)
        predicted = np.concatenate(predictions)

        peak = 0
        if measure_ram:
            sample = evaluation[: min(5, len(evaluation))]
            flt_ram = factory()
            if flt_ram.supervised:
                flt_ram.fit(train)
            _, peak = metrics.measure_memory(
                lambda f=flt_ram: [f.flag(b) for b in sample]
            )

        d = metrics.evaluate(predicted, truth)

        # Scoring pass on a fresh instance: stateful filters (reputation) must
        # not see the same batches twice.
        try:
            scorer = factory()
            if scorer.supervised:
                scorer.fit(train)
            parts = [scorer.score(b) for b in evaluation]
            rank = (
                metrics.ranking(np.concatenate(parts), truth)
                if all(x is not None for x in parts)
                else {"auc": float("nan"), "average_precision": float("nan")}
            )
        except Exception as error:
            print(f"    [warning] score unavailable: {error}", flush=True)
            rank = {"auc": float("nan"), "average_precision": float("nan")}

        t_f, no_evid = trust.trust_score(target, value, ~predicted, p.n_nodes)
        mae_f = trust.mae(t_f, sim.t_ref)

        dcs, clusters = [], []
        if isinstance(flt, DPCFilter):
            diag = flt.diagnostics()
            if diag:
                dcs.append(diag["dc"])
                clusters.append(diag["n_clusters"])

        rows.append(
            {
                "scenario": scenario,
                "attack": p.attack, "mu": p.mu, "seed": p.seed,
                "n_nodes": p.n_nodes, "k_targets": p.k_targets,
                "n_rounds": p.n_rounds, "n_fog": p.n_fog,
                "consensus_window": p.consensus_window,
                "filter": name, "variant": variant,
                "tp": d.true_positive, "fp": d.false_positive,
                "fn": d.false_negative, "tn": d.true_negative,
                "precision": round(d.precision, 6), "recall": round(d.recall, 6),
                "f1": round(d.f1, 6), "accuracy": round(d.accuracy, 6),
                "auc": round(rank["auc"], 6),
                "average_precision": round(rank["average_precision"], 6),
                # Layer 3: the ground truth is already per recommendation, so
                # there is no second convention to report. Left empty rather
                # than repeated, so that nobody reads two equal columns as a
                # confirmation.
                "auc_recommendation": "",
                "average_precision_recommendation": "",
                "malicious_rate_recommendation": "",
                "mae_filtered": round(mae_f, 6),
                "mae_unfiltered": round(mae_unf, 6),
                "mae_reduction": round(mae_unf - mae_f, 6),
                "nodes_without_evidence": int(no_evid.sum()),
                "discard_rate": round(float(predicted.mean()), 6),
                "mean_batch_time_ms": round(1000 * total_time / len(evaluation), 4),
                "total_time_s": round(total_time, 5),
                "train_time_s": round(train_time, 5),
                "peak_memory_kb": round(peak / 1024.0, 1),
                "n_batches": len(evaluation),
                "n_evaluated": int(truth.size),
                "malicious_rate": round(float(truth.mean()), 6),
                "median_dc": round(float(np.median(dcs)), 5) if dcs else "",
                "median_n_clusters": int(np.median(clusters)) if clusters else "",
                "version": __version__, "machine": MACHINE,
            }
        )
    return rows


def sweep(
    scenario_names: Sequence[str] = ("basic", "heterogeneous", "stealthy", "adverse"),
    attacks: Sequence[str] = ("bad_mouthing", "ballot_stuffing", "collusion", "mixed"),
    mus: Sequence[float] = MU_GRID,
    seeds: Sequence[int] = SEED_GRID,
    base: Params | None = None,
    output: str | Path = "results/results.csv",
    filter_names: Iterable[str] | None = None,
    dc_variants: Sequence[float] = (),
    verbose: bool = True,
) -> Path:
    base = base or Params()
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)

    combinations = list(itertools.product(scenario_names, attacks, mus, seeds))
    start = time.perf_counter()
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        for n, (scenario, attack, mu, seed) in enumerate(combinations, 1):
            p = apply_scenario(
                replace(base, attack=attack, mu=mu, seed=seed), scenario
            )
            for row in run_one(p, filter_names, dc_variants, scenario=scenario):
                writer.writerow(row)
            fh.flush()
            if verbose:
                elapsed = time.perf_counter() - start
                remaining = elapsed / n * (len(combinations) - n)
                print(
                    f"[{n:4d}/{len(combinations)}] {scenario:13s} {attack:16s} mu={mu:.2f} "
                    f"seed={seed:2d}  elapsed={elapsed:6.1f}s "
                    f"remaining~{remaining:6.1f}s",
                    flush=True,
                )
    return path
