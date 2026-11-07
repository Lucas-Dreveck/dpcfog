"""Reimplementation of Deep Trust (Ullah et al., 2024).

The original paper describes the framework, not the model: its pseudocode
leaves learning rate, epochs and batch size as free parameters, it cites RNN,
LSTM and CNN without choosing one, and it gives neither the number of layers
nor of units. This is therefore a **declared operationalisation**, not a
replica. What the paper specifies was followed; what it omits was chosen here
and is listed below.

What the paper specifies:

| Element | Where in the paper | Implementation |
|---|---|---|
| PCA over the features | eqs. (2)-(3) | scikit-learn `PCA`, retained variance below |
| Recurrent model over the temporal sequence | Sec. III-C-2, h_t = f(U x_t + W h_(t-1) + b) | one-layer GRU |
| Non-linear activation (tanh or ReLU) | Sec. III-C-2 | tanh in the GRU gates, ReLU in the output head |
| Adam optimiser | Table 4 | `torch.optim.Adam` |
| Per-device output over time | Algorithm 1, steps 12-14 | sequence of the rater's last `steps` ratings |

What the paper does not specify:

| Hyperparameter | Value | Rationale |
|---|---|---|
| Hidden layers | 1 GRU | the minimum that realises the cited recurrence |
| Hidden size | 64 | usual order of magnitude for a 5-feature input |
| Sequence steps | 8 | ratings per rater on Bitcoin OTC: median 2, 90th percentile 8 |
| PCA retained variance | 95% | conventional value |
| Learning rate | 1e-3 | Adam default |
| Epochs | 40 | fixed value; with 10, 20 and 80 epochs the layer-1 AUC changes by at most 0.043, within the confidence interval |
| Batch size | 256 | gradient noise versus CPU time |
| Loss | class-weighted binary cross-entropy | the malicious class is a minority (19% to 23%); without the weight the model never flags anyone |

There is no separate validation set: the model trains only on the training
partition.
"""

from __future__ import annotations

import numpy as np

from .base import Batch, Filter

STEPS = 8
HIDDEN_SIZE = 64
PCA_VARIANCE = 0.95
LEARNING_RATE = 1e-3
EPOCHS = 40
TRAIN_BATCH = 256


def _sequences(x: np.ndarray, rater: np.ndarray, steps: int) -> np.ndarray:
    """For each recommendation, the sequence of the last `steps` ones from the
    same rater within the batch, padding the start by repeating the first.

    It is the "temporal sequence of device interactions" of Algorithm 1. With a
    median of 2 ratings per user in the real networks, most sequences are
    mostly padding, a limitation of the data rather than of the implementation.
    """
    n, d = x.shape
    out = np.repeat(x[:, None, :], steps, axis=1)
    order = np.lexsort((np.arange(n), rater))
    positions_by_rater: dict[int, list[int]] = {}
    for pos in order:
        positions_by_rater.setdefault(int(rater[pos]), []).append(int(pos))
    for indices in positions_by_rater.values():
        for k, target in enumerate(indices):
            window = indices[max(0, k - steps + 1) : k + 1]
            block = x[window]
            if block.shape[0] < steps:
                block = np.vstack(
                    [np.repeat(block[:1], steps - block.shape[0], axis=0), block]
                )
            out[target] = block
    return out


class DeepTrustFilter(Filter):
    name = "deeptrust"
    supervised = True

    def __init__(
        self,
        seed: int = 0,
        steps: int = STEPS,
        hidden: int = HIDDEN_SIZE,
        epochs: int = EPOCHS,
    ):
        self.seed = seed
        self.steps = steps
        self.hidden = hidden
        self.epochs = epochs
        self.model = None
        self.pca = None
        self.scaler = None
        self.single_class: bool | None = None

    # -------------------------------------------------------------- training

    def fit(self, batches: list[Batch]) -> None:
        import torch
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        x = np.vstack([b.x[b.evaluable] for b in batches])
        y = np.concatenate([b.label[b.evaluable] for b in batches])
        rater = np.concatenate([b.rater[b.evaluable] for b in batches])
        if y.sum() == 0 or y.sum() == y.size:
            self.single_class = bool(y[0]) if y.size else False
            return
        self.single_class = None

        self.scaler = StandardScaler().fit(x)
        self.pca = PCA(n_components=PCA_VARIANCE, random_state=self.seed)
        z = self.pca.fit_transform(self.scaler.transform(x))

        seq = torch.tensor(
            _sequences(z, rater, self.steps), dtype=torch.float32
        )
        target = torch.tensor(y.astype(np.float32))

        self.model = _Network(z.shape[1], self.hidden)
        optimiser = torch.optim.Adam(
            self.model.parameters(), lr=LEARNING_RATE
        )
        # minority-class weight: without it the model never flags anyone
        positives = float(target.sum())
        weight = torch.tensor(
            (len(target) - positives) / max(positives, 1.0), dtype=torch.float32
        )
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=weight)

        self.model.train()
        n = len(target)
        for _ in range(self.epochs):
            shuffled = torch.randperm(n)
            for start in range(0, n, TRAIN_BATCH):
                idx = shuffled[start : start + TRAIN_BATCH]
                optimiser.zero_grad()
                loss = loss_fn(self.model(seq[idx]), target[idx])
                loss.backward()
                optimiser.step()
        self.model.eval()

    # ---------------------------------------------------------- inference

    def _logits(self, batch: Batch):
        import torch

        z = self.pca.transform(self.scaler.transform(batch.x))
        seq = torch.tensor(
            _sequences(z, batch.rater, self.steps), dtype=torch.float32
        )
        with torch.no_grad():
            return self.model(seq).numpy()

    def flag(self, batch: Batch) -> np.ndarray:
        if self.model is None:
            value = bool(self.single_class) if self.single_class is not None else False
            return np.full(batch.x.shape[0], value, dtype=bool)
        return self._logits(batch) > 0.0

    def score(self, batch: Batch) -> np.ndarray:
        if self.model is None:
            return np.zeros(batch.x.shape[0])
        return self._logits(batch)


def _Network(n_inputs: int, hidden: int):
    """One-layer GRU plus a linear head. Defined inside a function so that
    `import torch` stays lazy and the rest of the package does not depend on it."""
    import torch

    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = torch.nn.GRU(n_inputs, hidden, batch_first=True)
            self.head = torch.nn.Sequential(
                torch.nn.Linear(hidden, hidden // 2),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden // 2, 1),
            )

        def forward(self, seq):
            _, h = self.gru(seq)
            return self.head(h[-1]).squeeze(-1)

    return Network()
