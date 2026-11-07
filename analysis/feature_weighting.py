"""Does weighting the features recover the signal DPC does not use?

    python -m analysis.feature_weighting otc

With five features and an isotropic Euclidean distance, the discriminant
direction is diluted by the other four. Three conditions on the same batches:

    equal     the filter as it is
    bimodal   weights from Sarle's bimodality coefficient; no label, so usable
              in operation
    oracle    weights proportional to the separation measured with the label.
              Not a filter but a ceiling: how much weighting could gain at most.
              No number from this row is a performance result.
"""
from __future__ import annotations

import sys

import numpy as np

from dpcfog import metrics
from dpcfog.features import FEATURE_NAMES, standardize
from dpcfog.filters import DPCFilter
from realdata.attacks import Attack, inject
from realdata.batches import build_batches, split_by_time
from realdata.fine_label import label_per_recommendation
from realdata.networks import load


def oracle_weights(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Separation per feature, in standard deviations, normalised to sum to 1."""
    z = standardize(x)
    sep = np.array([abs(z[y, k].mean() - z[~y, k].mean()) for k in range(z.shape[1])])
    return sep / sep.sum() if sep.sum() > 0 else np.ones(z.shape[1]) / z.shape[1]


def evaluate(evaluation, y, m, factory) -> float:
    sc = np.concatenate([factory().score(b) for b in evaluation])
    return metrics.ranking(sc[m], y[m])["auc"]


def main(network: str = "otc", mu: float = 0.05, min_degree: int = 400) -> None:
    clean = load(network)
    attack = Attack(model="random", direction="nuke", attack_size=mu,
                    realistic_filler=True, seed=0,
                    target_min_popularity=min_degree)
    net = inject(clean, attack)
    fine = label_per_recommendation(net)
    _, evaluation = split_by_time(build_batches(net, window=5, n_fog=1))
    m = np.concatenate([b.evaluable for b in evaluation])
    x = np.concatenate([b.x for b in evaluation])
    key = set(zip(net.rater[fine].tolist(), net.target[fine].tolist()))
    y = np.concatenate([
        np.array([(r, t) in key for r, t in zip(b.rater, b.target)])
        for b in evaluation
    ])

    w_or = oracle_weights(x[m], y[m])
    print(f"{network}, mu={mu}, target degree >= {min_degree}, per-recommendation label")
    print("oracle weights (measured separation):")
    for n, w in zip(FEATURE_NAMES, w_or):
        print(f"  {n:>12}: {w:>5.3f}")
    print()
    print(f"{'condition':>10} {'AUC':>8}")
    print(f"{'equal':>10} {evaluate(evaluation, y, m, lambda: DPCFilter()):>8.3f}")
    print(f"{'bimodal':>10} "
          f"{evaluate(evaluation, y, m, lambda: DPCFilter(weights='bimodal')):>8.3f}")
    print(f"{'oracle':>10} "
          f"{evaluate(evaluation, y, m, lambda: DPCFilter(weights=w_or)):>8.3f}   <- ceiling, "
          f"uses the label, NOT a performance result")


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["otc"]))
