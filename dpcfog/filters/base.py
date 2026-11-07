"""Common filter interface.

Every filter receives the same batch and returns the same kind of output, so
that comparing DPC with the baselines needs no per-filter adaptation. Contract:
`flag` returns a boolean vector the size of the batch, True for a recommendation
judged malicious (that is, discarded).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Batch:
    """What a Fog node sees in one round."""

    x: np.ndarray            # raw features (n, 5)
    value: np.ndarray        # v of the recommendations in the batch
    rater: np.ndarray
    target: np.ndarray
    round_no: int
    fog: int
    label: np.ndarray        # ground truth; used by the evaluator, never by the filter
    evaluable: np.ndarray | None = None
    # `evaluable` marks the positions with a trustworthy label. In synthetic
    # data all of them are. In real data most users are unlabelled and the
    # metric must ignore them: counting unknown as benign would invent 76% of
    # the Bitcoin OTC ground truth. The filter does not receive this
    # information, as it would not in operation.

    def __post_init__(self):
        if self.evaluable is None:
            self.evaluable = np.ones(self.label.shape, dtype=bool)


class Filter:
    """Abstract base. Subclasses implement `flag`."""

    name = "base"
    supervised = False

    def fit(self, batches: list[Batch]) -> None:
        """Called only for supervised filters, with the training batches."""
        return None

    def flag(self, batch: Batch) -> np.ndarray:
        raise NotImplementedError

    def score(self, batch: Batch) -> np.ndarray | None:
        """Continuous suspicion per recommendation, higher meaning more suspicious.

        Comparing filters by F1 also compares each filter's choice of threshold,
        and measured discard rates range from 0.3% to 51%. A ranking metric
        (AUC, average precision) separates signal quality from cut calibration.

        Returns None for a filter that can only decide, not rank.
        """
        return None

    def diagnostics(self) -> dict[str, Any]:
        """Internal information about the last decision, for figures."""
        return {}
