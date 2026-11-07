"""Estimate the simulation parameters from a real network.

    python -m realdata.calibration otc

The reference parameters fix benign T* at U(0.7, 1.0), malicious T* at
U(0.0, 0.3), sigma = 0.05 and K = 10, none of them measured. This module
estimates them from Bitcoin OTC with the labels of Kumar et al. (2018) and
returns a `Params` with per-field provenance. Layer 3 thus stops depending on
hand-picked numbers.

Pitfall: `rater` and `target` are independent index spaces in `RealNetwork`.
The REV2 label belongs to the rater, while T* is a statistic of the rated user.
Joining them requires the original identifier, via `meta["rater_code"]` and
`meta["target_code"]`. Joining index with index gives a plausible-looking and
meaningless estimate.
"""

from __future__ import annotations

import collections
import sys
from dataclasses import replace

import numpy as np

from dpcfog.config import Params

from .networks import RealNetwork, load

# Minimum degree to estimate within-target spread. Below it the sample
# deviation is noise.
MIN_RATINGS_PER_TARGET = 5


def _t_by_class(net: RealNetwork) -> tuple[np.ndarray, np.ndarray]:
    """Observed T* of labelled nodes, split by class.

    A node's T* is the mean of the ratings it received. The label belongs to the
    node as a rater. The join goes through the original id.
    """
    total = np.zeros(net.n_targets)
    count = np.zeros(net.n_targets)
    np.add.at(total, net.target, net.value)
    np.add.at(count, net.target, 1.0)
    has = count > 0
    t = np.where(has, total / np.where(has, count, 1.0), np.nan)

    code_r = net.meta["rater_code"]
    code_t = net.meta["target_code"]
    original = {v: k for k, v in code_r.items()}

    label_of_rater: dict[int, bool] = {}
    for r, lab, ev in zip(net.rater, net.label, net.evaluable):
        if ev:
            label_of_rater[int(r)] = bool(lab)

    benign, malicious = [], []
    for index_r, fraudulent in label_of_rater.items():
        original_id = original[index_r]
        if original_id not in code_t:
            continue          # the node rates but was never rated
        index_t = code_t[original_id]
        if not has[index_t]:
            continue
        (malicious if fraudulent else benign).append(t[index_t])
    return np.array(benign), np.array(malicious)


def _within_target_spread(net: RealNetwork) -> np.ndarray:
    """Spread of the honest ratings about the same target.

    An upper bound for sigma: the observed spread adds observation noise to real
    variation of opinion about the target, and separating them would take one
    more assumption.
    """
    by_target: dict[int, list[float]] = collections.defaultdict(list)
    for a, v, ev, lab in zip(net.target, net.value, net.evaluable, net.label):
        if ev and not lab:
            by_target[int(a)].append(float(v))
    return np.array([
        np.std(v) for v in by_target.values() if len(v) >= MIN_RATINGS_PER_TARGET
    ])


def _effective_k(net: RealNetwork) -> float:
    """Ratings issued per node, per round in which the node was active."""
    degree = np.bincount(net.rater)
    degree = degree[degree > 0]
    rounds = collections.defaultdict(set)
    for r, t in zip(net.rater, net.round_idx):
        rounds[int(r)].add(int(t))
    active = np.array([len(v) for v in rounds.values()])
    return float(degree.mean() / active.mean())


def estimate(network: str = "otc") -> tuple[Params, dict]:
    """Return the calibrated parameters and the provenance of each field."""
    net = load(network)
    ben, mal = _t_by_class(net)
    spread = _within_target_spread(net)
    k = _effective_k(net)

    t_ben = (float(np.percentile(ben, 10)), float(np.percentile(ben, 90)))
    t_mal = (float(np.percentile(mal, 10)), float(np.percentile(mal, 90)))
    sigma = float(np.median(spread))
    band = (float(np.percentile(spread, 10)), float(np.percentile(spread, 90)))

    p = replace(
        Params(),
        t_benign=t_ben,
        t_malicious=t_mal,
        noise_sigma=round(sigma, 4),
        sigma_range=(round(band[0], 4), round(band[1], 4)),
        k_targets=max(2, int(round(k))),
    )
    provenance = {
        "t_benign": f"measured on {network}: p10-p90 of {ben.size} benign nodes "
                    f"(reference: 0.70-1.00)",
        "t_malicious": f"measured on {network}: p10-p90 of {mal.size} fraudsters "
                       f"(reference: 0.00-0.30)",
        "noise_sigma": f"measured on {network}: median within-target spread, "
                       f"{spread.size} targets with >= {MIN_RATINGS_PER_TARGET} honest "
                       f"ratings; UPPER BOUND (reference: 0.05)",
        "sigma_range": f"measured on {network}: p10-p90 of the same spread "
                       f"(reference: 0.02-0.20)",
        "k_targets": f"measured on {network}: {k:.2f} ratings per node per active round "
                     f"(reference: 10)",
        "n_nodes": "reference value; not estimable without fixing the scale",
        "n_rounds": "reference value",
        "mu": "reference grid; it is the swept variable",
        "measured_separation": f"{abs(ben.mean() - mal.mean()) / np.sqrt((ben.var() + mal.var()) / 2):.2f} sigma "
                               f"(the reference implies ~4.3)",
    }
    return p, provenance


def main(network: str = "otc") -> None:
    p, prov = estimate(network)
    print(f"parameters calibrated on {network}\n")
    for name in ("t_benign", "t_malicious", "noise_sigma", "sigma_range",
                 "k_targets"):
        print(f"  {name:>12} = {getattr(p, name)}")
        print(f"                 {prov[name]}")
    print(f"\n  class separation: {prov['measured_separation']}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "otc")
