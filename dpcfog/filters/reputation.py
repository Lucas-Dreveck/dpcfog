"""Baseline: filter by the rater's reputation.

Inspired by classic trust management models for wireless networks (Boukerch,
Xu and El-Khatib, 2007): instead of judging the recommendation, it judges who
issued it, and reputation accumulates over time with a forgetting factor.

    rep_i(t) = alpha * rep_i(t-1) + (1 - alpha) * agreement_i(t)

Every recommendation from a rater below the threshold is discarded. It is the
only filter with state across rounds, which makes it strong against persistent
attackers and blind to attackers who act only occasionally.
"""

from __future__ import annotations

import numpy as np

from .base import Batch, Filter


class ReputationFilter(Filter):
    name = "reputation"

    def __init__(
        self,
        alpha: float = 0.8,
        threshold: float = 0.5,
        tolerance: float = 0.15,
        optimistic_prior: bool = False,
    ):
        self.alpha = alpha
        self.threshold = threshold
        self.tolerance = tolerance
        self.optimistic_prior = optimistic_prior
        self.reputation: dict[int, float] = {}

    def reset(self) -> None:
        """Called between runs: reputation does not cross seeds."""
        self.reputation.clear()

    def flag(self, batch: Batch) -> np.ndarray:
        agrees = np.abs(batch.x[:, 0]) <= self.tolerance
        current = np.empty(batch.x.shape[0])

        for i in np.unique(batch.rater):
            m = batch.rater == i
            observed = float(agrees[m].mean())
            if int(i) in self.reputation:
                new = (
                    self.alpha * self.reputation[int(i)]
                    + (1.0 - self.alpha) * observed
                )
            elif self.optimistic_prior:
                new = self.alpha * 1.0 + (1.0 - self.alpha) * observed
            else:
                # A newcomer has no reputation, and assuming 1.0 is an
                # unjustified optimistic prior. On Bitcoin OTC, with prior 1.0
                # and alpha 0.8, reputation only falls below 0.5 after several
                # rounds of disagreement, while the median rater appears in a
                # single round: the filter never discarded anything (F1 = 0).
                # Without the prior, the first evidence counts.
                new = observed
            self.reputation[int(i)] = new
            current[m] = new

        self._last_reputation = current
        return current < self.threshold

    def score(self, batch: Batch) -> np.ndarray:
        # Always recompute. Reusing the state of the last `flag` call would
        # return a vector the size of the previous batch. Because this filter
        # is stateful, the caller must use a fresh instance for scoring.
        self.flag(batch)
        return 1.0 - self._last_reputation
