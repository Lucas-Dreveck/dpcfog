"""Does the committed CSV still match what the code produces today?

    python -m validation.stale_csv

The code evolves and an artefact can stay behind, so that a paper publishes a
number the code no longer produces. No unit test or code review catches it.

For each unsupervised filter, this recomputes the layer-1 AUC from the raw
network and compares it with the CSV. Supervised filters are left out because
training has random initialisation.

Tolerance: 0.001. A larger gap means a stale CSV, and the answer is to re-run
the experiment, never to edit the CSV.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from dpcfog import metrics
from dpcfog.experiment import filter_catalog
from realdata.batches import build_batches, split_by_time
from realdata.networks import load

TOLERANCE = 0.001
CSV = "results/real.csv"
NETWORKS = {"real_otc": "otc", "real_alpha": "alpha"}
# Epinions takes about half an hour and is off by default. Run
#     python -m validation.stale_csv epinions
# after re-running its experiment.
SLOW_NETWORKS = {"real_epinions": ("epinions", dict(rounds_per_year=365))}


def auc_today(network: str, names: list[str], load_args: dict | None = None,
              n_fog: int = 1) -> dict[str, float]:
    net = load(network, **(load_args or {}))
    _, evaluation = split_by_time(build_batches(net, window=5, n_fog=n_fog))
    m = np.concatenate([b.evaluable for b in evaluation])
    y = np.concatenate([b.label for b in evaluation])
    fac = dict(filter_catalog(0))

    out = {}
    for name in names:
        f = fac[name]()
        if f.supervised:
            continue
        try:
            parts = [f.score(b) for b in evaluation]
        except Exception:
            continue
        if any(p is None for p in parts):
            continue
        out[name] = metrics.ranking(np.concatenate(parts)[m], y[m])["auc"]
    return out


def main(argv: list[str] | None = None) -> int:
    targets = dict(NETWORKS)
    if argv and "epinions" in argv:
        targets["real_epinions"] = SLOW_NETWORKS["real_epinions"][0]
    try:
        d = pd.read_csv(CSV)
    except FileNotFoundError:
        print(f"{CSV} does not exist. Run `python -m realdata.real_experiment`.")
        return 1

    problems = []
    for scenario, network in targets.items():
        sub = d[d.scenario == scenario]
        if sub.empty:
            print(f"{scenario}: no rows in the CSV, skipping")
            continue
        published = dict(zip(sub["filter"], sub.auc))
        load_args, n_fog = ({}, 1)
        if scenario == "real_epinions":
            load_args, n_fog = dict(rounds_per_year=365), 8
        now = auc_today(network, list(published), load_args, n_fog)

        print(f"\n{'=' * 66}\n{scenario}\n{'=' * 66}")
        print(f"{'filter':>20} {'in CSV':>9} {'today':>9} {'difference':>11}")
        for name in sorted(now):
            a, b = published[name], now[name]
            dif = b - a
            mark = "" if abs(dif) <= TOLERANCE else "   <== STALE CSV"
            if mark:
                problems.append(f"{scenario}/{name}: CSV {a:.3f}, today {b:.3f}")
            print(f"{name:>20} {a:>9.3f} {b:>9.3f} {dif:>+11.3f}{mark}")

    print("\n" + "-" * 66)
    if problems:
        for p in problems:
            print(f"  FAIL: {p}")
        print("\nThe CSV does not reflect the current code. Re-run the experiment:")
        print("    python -m realdata.real_experiment otc alpha")
        print("Never edit the CSV by hand to make this script pass.")
        return 1
    print("The committed CSV matches what the code produces today.")
    print("Supervised filters are left out: training has random initialisation.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
