"""Supervised filters measured outside the condition they were trained on.

    python out_of_distribution.py

Trained on bad-mouthing and tested on ballot-stuffing, the MLP gave F1 = 0.000
against 0.995 within its own distribution: the in-distribution number memorises
the generator's signature. In operation there is no labelled data for the attack
that has not happened yet, so the honest number is the out-of-distribution one,
and the in-distribution one is the optimistic ceiling.

DPC and the heuristics do not train, so their row has the same meaning in every
cell.

Output: `results/out_of_distribution.csv`, one row per
(filter, training attack, test attack, seed).
"""

from __future__ import annotations

import csv
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from dpcfog import __version__, metrics
from dpcfog.config import Params
from dpcfog.experiment import build_batches, filter_catalog, split
from dpcfog.generator import generate
from dpcfog.scenarios import apply

ATTACKS = ("bad_mouthing", "ballot_stuffing", "collusion", "on_off")
SEEDS = (0, 1, 2, 3, 4)
SCENARIOS = ("basic", "calibrated")

COLUMNS = [
    "scenario", "train_attack", "test_attack", "in_distribution", "seed", "mu",
    "filter", "supervised", "auc", "average_precision", "f1", "discard_rate",
    "malicious_rate", "n_evaluated", "version",
]


def _prepare(scenario: str, attack: str, seed: int, mu: float):
    p = apply(replace(Params(), attack=attack, mu=mu, seed=seed), scenario)
    sim = generate(p)
    return split(build_batches(sim), p)


def run(scenario: str, mu: float = 0.20) -> list[dict]:
    rows: list[dict] = []
    factories = filter_catalog(0)

    for seed in SEEDS:
        # one test set per attack, generated once and reused by every training
        # run, so that cells do not mix test variation with training variation
        tests = {
            a: _prepare(scenario, a, seed, mu)[1] for a in ATTACKS
        }
        for a_train in ATTACKS:
            train, _ = _prepare(scenario, a_train, seed, mu)
            for name, factory in factories.items():
                flt = factory()
                if flt.supervised:
                    flt.fit(train)
                elif a_train != ATTACKS[0]:
                    # A filter that does not train gives the same result for
                    # every training row; repeating it would fill the table
                    # with identical rows that look like independent evidence.
                    continue

                for a_test in ATTACKS:
                    evaluation = tests[a_test]
                    if not evaluation:
                        continue
                    y = np.concatenate([b.label for b in evaluation])
                    if y.all() or not y.any():
                        continue
                    fresh = factory()
                    if fresh.supervised:
                        fresh.fit(train)
                    predicted = np.concatenate([fresh.flag(b) for b in evaluation])
                    d = metrics.evaluate(predicted, y)
                    scorer = factory()
                    if scorer.supervised:
                        scorer.fit(train)
                    try:
                        sc = np.concatenate([scorer.score(b) for b in evaluation])
                        rank = metrics.ranking(sc, y)
                    except Exception:
                        rank = {"auc": float("nan"),
                                "average_precision": float("nan")}
                    rows.append({
                        "scenario": scenario,
                        "train_attack": (
                            a_train if flt.supervised else "n/a"
                        ),
                        "test_attack": a_test,
                        "in_distribution": int(
                            flt.supervised and a_train == a_test
                        ),
                        "seed": seed,
                        "mu": mu,
                        "filter": name,
                        "supervised": int(flt.supervised),
                        "auc": round(rank["auc"], 6),
                        "average_precision": round(rank["average_precision"], 6),
                        "f1": round(d.f1, 6),
                        "discard_rate": round(float(predicted.mean()), 6),
                        "malicious_rate": round(float(y.mean()), 6),
                        "n_evaluated": int(y.size),
                        "version": __version__,
                    })
    return rows


def main(output: str = "results/out_of_distribution.csv") -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        for n, scenario in enumerate(SCENARIOS, 1):
            for row in run(scenario):
                writer.writerow(row)
            fh.flush()
            print(f"[{n}/{len(SCENARIOS)}] {scenario} "
                  f"elapsed={time.perf_counter() - start:.0f}s", flush=True)
    return path


if __name__ == "__main__":
    print(main())
