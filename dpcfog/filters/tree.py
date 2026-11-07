"""Control baseline: a shallow decision tree.

It measures how much structure the task actually has. In the synthetic setting
a depth-3 tree reached F1 = 0.998, above the three-hidden-layer perceptron: if a
model with at most eight leaves ties with a neural network, the task did not
require depth, and a comparison against "Deep Learning" would be a comparison
against a label, not against a difficulty.

Inherits from MLPFilter to reuse the handling of trustworthy labels in training.
"""

from __future__ import annotations

import numpy as np

from .base import Batch
from .mlp import MLPFilter


class TreeFilter(MLPFilter):
    name = "tree3"
    supervised = True

    def __init__(self, depth: int = 3, seed: int = 0):
        super().__init__(seed=seed)
        self.depth = depth

    def fit(self, batches: list[Batch]) -> None:
        from sklearn.preprocessing import StandardScaler
        from sklearn.tree import DecisionTreeClassifier

        x = np.vstack([b.x[b.evaluable] for b in batches])
        y = np.concatenate([b.label[b.evaluable] for b in batches])
        if y.sum() == 0 or y.sum() == y.size:
            self.single_class = bool(y[0]) if y.size else False
            return
        self.single_class = None
        self.scaler = StandardScaler().fit(x)
        self.model = DecisionTreeClassifier(
            max_depth=self.depth, random_state=self.seed
        )
        self.model.fit(self.scaler.transform(x), y)
