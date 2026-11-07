"""Detection and cost metrics.

`tracemalloc` instruments every interpreter allocation and slows the run down,
so timing and memory are measured in two separate passes over the same input:

    measure_time    perf_counter, with no instrumentation active
    measure_memory  tracemalloc peak allocation; the time of this pass is discarded
"""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Detection:
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int

    @property
    def precision(self) -> float:
        d = self.true_positive + self.false_positive
        return self.true_positive / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.true_positive + self.false_negative
        return self.true_positive / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def accuracy(self) -> float:
        total = (
            self.true_positive
            + self.false_positive
            + self.false_negative
            + self.true_negative
        )
        return (
            (self.true_positive + self.true_negative) / total
            if total
            else 0.0
        )


def evaluate(predicted: np.ndarray, truth: np.ndarray) -> Detection:
    predicted = predicted.astype(bool)
    truth = truth.astype(bool)
    return Detection(
        true_positive=int((predicted & truth).sum()),
        false_positive=int((predicted & ~truth).sum()),
        false_negative=int((~predicted & truth).sum()),
        true_negative=int((~predicted & ~truth).sum()),
    )


def ranking(score: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """AUC and average precision, which measure the signal independently of the cut.

    F1 mixes how well a filter ranks recommendations by suspicion with where it
    decided to cut. Measured discard rates range from 0.3% to 51%, so comparing
    by F1 alone mostly compares the calibration of the cuts.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    truth = truth.astype(bool)
    if truth.all() or not truth.any() or score.size == 0:
        return {"auc": float("nan"), "average_precision": float("nan")}
    finite = np.isfinite(score)
    if not finite.all():
        score = np.where(finite, score, np.nanmin(score[finite]))
    return {
        "auc": float(roc_auc_score(truth, score)),
        "average_precision": float(average_precision_score(truth, score)),
    }


def measure_time(func: Callable[[], object], repetitions: int = 1) -> tuple[object, float]:
    """Return (result, mean time in seconds). No profiler active."""
    if tracemalloc.is_tracing():
        raise RuntimeError(
            "tracemalloc active while timing: the number would be invalid"
        )
    result = None
    start = time.perf_counter()
    for _ in range(repetitions):
        result = func()
    return result, (time.perf_counter() - start) / repetitions


def measure_memory(func: Callable[[], object]) -> tuple[object, int]:
    """Return (result, peak allocation in bytes). The time of this pass is
    meaningless and must not be reused."""
    tracemalloc.start()
    try:
        result = func()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, int(peak)
