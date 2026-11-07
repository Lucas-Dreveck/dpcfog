"""Baseline: supervised neural network (multilayer perceptron).

It is the accuracy ceiling against which hypothesis H1 measures its five
percentage points, and the cost floor against which it measures savings.

Three asymmetries favour this baseline:

1. It is supervised and sees the ground truth in training; DPC never sees labels.
2. It is trained on the whole history of the network, while DPC decides with
   what a single Fog node sees in the current window.
3. Its scaler is global; DPC's is local to the batch.

The number it produces is therefore an optimistic upper bound. The architecture
(three hidden layers of 64, 32 and 16 units) is not a deep model of the size
found in the literature, so its measured cost underestimates that of a real deep
model, which makes the cost comparison conservative against the hypothesis too.
"""

from __future__ import annotations

import numpy as np

from .base import Batch, Filter


class MLPFilter(Filter):
    name = "mlp"
    supervised = True

    def __init__(
        self,
        layers: tuple[int, ...] = (64, 32, 16),
        max_iter: int = 200,
        seed: int = 0,
    ):
        self.layers = layers
        self.max_iter = max_iter
        self.seed = seed
        self.model = None
        self.scaler = None
        self.single_class: bool | None = None

    def fit(self, batches: list[Batch]) -> None:
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler

        # Only trustworthy labels enter training. In real data most users are
        # unlabelled, and treating "unknown" as "benign" would teach the model
        # to be wrong: on Bitcoin OTC, 71.9% of the training set would fall in
        # the wrong class and the model learned never to flag anyone
        # (recall 0.075). That would handicap the baseline through our own
        # defect and bias the comparison in favour of DPC.
        x = np.vstack([b.x[b.evaluable] for b in batches])
        y = np.concatenate([b.label[b.evaluable] for b in batches])

        if y.sum() == 0 or y.sum() == y.size:
            # Without both classes there is nothing to train. It happens with
            # low mu and rare attacks; the run is marked, not masked.
            self.single_class = bool(y[0]) if y.size else False
            return

        self.single_class = None
        self.scaler = StandardScaler().fit(x)
        self.model = MLPClassifier(
            hidden_layer_sizes=self.layers,
            activation="relu",
            solver="adam",
            max_iter=self.max_iter,
            random_state=self.seed,
            early_stopping=False,
        )
        import warnings
        from sklearn.exceptions import ConvergenceWarning

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.model.fit(self.scaler.transform(x), y)

    def flag(self, batch: Batch) -> np.ndarray:
        if self.model is None:
            value = bool(self.single_class) if self.single_class is not None else False
            return np.full(batch.x.shape[0], value, dtype=bool)
        return self.model.predict(self.scaler.transform(batch.x)).astype(bool)

    def score(self, batch: Batch) -> np.ndarray:
        if self.model is None:
            return np.zeros(batch.x.shape[0])
        return self.model.predict_proba(self.scaler.transform(batch.x))[:, 1]
