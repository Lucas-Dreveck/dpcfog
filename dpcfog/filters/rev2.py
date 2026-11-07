"""REV2 as a published baseline (Kumar et al., 2018, WSDM).

REV2 has public code, was evaluated on these same networks and is the algorithm
that defined the labels used here. The three quantities are mutually recursive,
and the implementation follows the `rev2code.py` distributed by the authors:

    goodness(p)          = mean of the ratings of p, weighted by each rating's
                           reliability
    reliability(u, p)    = mean of u's fairness and how close the rating is to
                           p's goodness
    fairness(u)          = mean reliability of the ratings issued by u

with the rating distance `1 - |rating - goodness| / 2` on the original [-1, 1]
scale. Our values live in [0, 1], so `w = 2v - 1` is applied on input.

Two differences from the original:

1. The cold-start terms (alpha, beta) and the behavioural ones (gamma_3, on
   inter-rating time and review text) are switched off. gamma_3 needs review
   text, which the Bitcoin networks lack; the others need global statistics a
   Fog node would not have. The configuration is gamma_1 = gamma_2 = 1,
   gamma_3 = 0, alpha = beta = 0, one of those the authors sweep.

2. REV2 was designed to run **once over the whole graph**, offline. Here it runs
   per batch, over the same window the other filters see. `score_global` keeps
   the original form as a ceiling.
"""

from __future__ import annotations

import numpy as np

from .base import Batch, Filter


def score(
    rater: np.ndarray,
    target: np.ndarray,
    value: np.ndarray,
    iterations: int = 15,
    tolerance: float = 1e-4,
    alpha1: float = 0.0,
    beta1: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (fairness per rater, goodness per target, reliability per rating).

    Rater and target are re-indexed internally, so the first two outputs follow
    the order of `np.unique`. The third has the size of the input.
    """
    if value.size == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)

    u, iu = np.unique(rater, return_inverse=True)
    p, ip = np.unique(target, return_inverse=True)
    w = 2.0 * value - 1.0  # [0,1] -> [-1,1], the paper's scale

    fairness = np.ones(u.size)
    reliability = np.ones(value.size)
    goodness = np.zeros(p.size)

    for _ in range(iterations):
        # target goodness: rating mean weighted by reliability
        weight = np.zeros(p.size)
        total = np.zeros(p.size)
        np.add.at(weight, ip, reliability)
        np.add.at(total, ip, reliability * w)
        if beta1 > 0:
            # Cold start: a target with few ratings is pulled towards the
            # network's median goodness instead of trusting its single rating.
            median_g = float(np.median(goodness)) if p.size else 0.0
            total = total + beta1 * median_g
            weight = weight + beta1
        new_goodness = np.where(weight > 0, total / np.where(weight > 0, weight, 1.0), 0.0)
        new_goodness = np.clip(new_goodness, -1.0, 1.0)

        # rating reliability: half the rater's fairness, half the closeness
        closeness = 1.0 - np.abs(w - new_goodness[ip]) / 2.0
        new_reliability = np.clip(
            (fairness[iu] + closeness) / 2.0, 0.0, 1.0
        )

        # rater fairness: mean reliability of the ratings issued
        count = np.zeros(u.size)
        acc = np.zeros(u.size)
        np.add.at(count, iu, 1.0)
        np.add.at(acc, iu, new_reliability)
        if alpha1 > 0:
            median_f = float(np.median(fairness)) if u.size else 1.0
            acc = acc + alpha1 * median_f
            count = count + alpha1
        new_fairness = np.where(count > 0, acc / np.where(count > 0, count, 1.0), 1.0)

        delta = max(
            np.abs(new_goodness - goodness).max(initial=0.0),
            np.abs(new_reliability - reliability).max(initial=0.0),
            np.abs(new_fairness - fairness).max(initial=0.0),
        )
        goodness, reliability, fairness = (
            new_goodness, new_reliability, new_fairness
        )
        if delta < tolerance:
            break

    return fairness, goodness, reliability


def score_global(network, iterations: int = 15, alpha1: float = 0.0, beta1: float = 0.0):
    """REV2 in its original form: one pass over the whole network.

    The ceiling for the per-batch variant; the gap between the two measures how
    much REV2 loses on a Fog node that does not see the whole graph.
    """
    return score(
        network.rater, network.target, network.value, iterations, alpha1=alpha1, beta1=beta1
    )


class REV2Filter(Filter):
    name = "rev2"

    def __init__(self, threshold: float = 0.5, iterations: int = 15):
        self.threshold = threshold
        self.iterations = iterations
        self._last: dict = {}

    def flag(self, batch: Batch) -> np.ndarray:
        fairness, goodness, reliability = score(
            batch.rater, batch.target, batch.value, self.iterations
        )
        self._last = {
            "fairness": fairness, "goodness": goodness,
            "reliability": reliability,
        }
        return reliability < self.threshold

    def score(self, batch: Batch) -> np.ndarray:
        _, _, reliability = score(
            batch.rater, batch.target, batch.value, self.iterations
        )
        return 1.0 - reliability

    def diagnostics(self) -> dict:
        return self._last
