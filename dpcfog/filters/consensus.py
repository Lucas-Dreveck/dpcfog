"""Baseline: static consensus heuristic.

Discards a recommendation whose deviation from the median of the
recommendations about the same target exceeds a fixed threshold. It is the
cheapest possible filter and stands for the family of static heuristics.

Its expected weakness is structural: the threshold is absolute and the decision
looks at one recommendation at a time. Under collusion the attackers shift the
median itself and end up inside the threshold.
"""

from __future__ import annotations

import numpy as np

from .base import Batch, Filter


class ConsensusFilter(Filter):
    name = "consensus"

    def __init__(self, threshold: float = 0.25):
        self.threshold = threshold

    def flag(self, batch: Batch) -> np.ndarray:
        return np.abs(batch.x[:, 0]) > self.threshold

    def score(self, batch: Batch) -> np.ndarray:
        return np.abs(batch.x[:, 0])
