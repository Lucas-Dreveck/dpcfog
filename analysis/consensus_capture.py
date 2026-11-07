"""Consensus capture: why the main feature inverts.

    python -m analysis.consensus_capture otc

The `deviation` feature measures v minus the median of the recommendations
about the target, assuming an honest majority about that target. When an attack
concentrates many profiles on one target, the attackers become the median: their
deviation goes to zero and the honest rater is the one with a large deviation.

Every filter judging a recommendation against the consensus inherits this
inversion (`consensus`, `dpc` and partly `rev2`). `dpc_han` does not, because
its anchors are fixed at 0 and 1. The counter-test varies the minimum degree of
the attacked target: if capture is the cause, a target with an audience must
give the separation back to the feature.
"""
from __future__ import annotations

import sys

import numpy as np

from dpcfog import metrics
from dpcfog.experiment import filter_catalog
from dpcfog.features import FEATURE_NAMES
from realdata.attacks import Attack, inject
from realdata.batches import build_batches, split_by_time
from realdata.fine_label import label_per_recommendation
from realdata.networks import load

FILTERS = ("dpc", "consensus", "rev2", "dpc_han", "reputation")


def measure(clean, mu: float, n_targets: int = 1, seed: int = 0,
            min_degree: int = 0) -> dict:
    attack = Attack(model="random", direction="nuke", attack_size=mu,
                    realistic_filler=True, seed=seed,
                    n_attack_targets=n_targets,
                    target_min_popularity=min_degree)
    net = inject(clean, attack)
    fine = label_per_recommendation(net)
    targets = np.unique(net.target[fine])
    injected = int(fine.sum()) / max(targets.size, 1)
    honest = float(np.mean([(clean.target == t).sum() for t in targets]))

    batches = build_batches(net, window=5, n_fog=1)
    _, evaluation = split_by_time(batches)
    m = np.concatenate([b.evaluable for b in evaluation])
    x = np.concatenate([b.x for b in evaluation])
    key = set(zip(net.rater[fine].tolist(), net.target[fine].tolist()))
    y = np.concatenate([
        np.array([(r, t) in key for r, t in zip(b.rater, b.target)])
        for b in evaluation
    ])
    if not (y & m).any():
        return {}

    fac = filter_catalog(seed)
    aucs = {}
    for name in FILTERS:
        sc = np.concatenate([fac[name]().score(b) for b in evaluation])
        aucs[name] = metrics.ranking(sc[m], y[m])["auc"]

    sep = {n: abs(x[y & m, k].mean() - x[~y & m, k].mean())
           / max(x[m, k].std(), 1e-9) for k, n in enumerate(FEATURE_NAMES)}
    return {"mu": mu, "n_targets": n_targets, "min_degree": min_degree,
            "inj": injected, "hon": honest,
            "ratio": injected / max(honest, 1e-9), "sep": sep, "auc": aucs}


def main(network: str = "otc") -> None:
    clean = load(network)
    print(f"{network}: random/nuke attack, PER-RECOMMENDATION label\n")
    print(f"{'mu':>5} {'targets':>8} {'inj/tgt':>8} {'hon/tgt':>8} {'ratio':>7} "
          f"{'sep(dev)':>9} " + " ".join(f"{f:>10}" for f in FILTERS))
    for mu, n_targets in ((0.05, 1), (0.10, 1), (0.20, 1), (0.40, 1),
                          (0.20, 10), (0.20, 50), (0.20, 200)):
        r = measure(clean, mu, n_targets)
        if not r:
            continue
        print(f"{r['mu']:>5.2f} {r['n_targets']:>8} {r['inj']:>8.0f} "
              f"{r['hon']:>8.1f} {r['ratio']:>7.0f} {r['sep']['deviation']:>9.2f} "
              + " ".join(f"{r['auc'][f]:>10.3f}" for f in FILTERS))
    print("\nseparation per feature (mu=0.20, 1 target), in standard deviations:")
    r = measure(clean, 0.20, 1)
    for n, v in r["sep"].items():
        print(f"  {n:>12}: {v:>5.2f}")

    print("\nsame mu (0.05), varying the minimum degree of the attacked target:")
    print(f"{'min deg':>8} {'real deg':>9} {'ratio':>7} {'sep(dev)':>9} "
          + " ".join(f"{f:>10}" for f in FILTERS))
    for min_degree in (0, 5, 20, 71, 200, 400):
        r = measure(clean, 0.05, 1, min_degree=min_degree)
        if not r:
            continue
        print(f"{min_degree:>8} {r['hon']:>9.0f} {r['ratio']:>7.1f} "
              f"{r['sep']['deviation']:>9.2f} "
              + " ".join(f"{r['auc'][f]:>10.3f}" for f in FILTERS))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "otc")
