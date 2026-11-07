"""Feature vector fed to the filters.

The five features, in this order:

    0  deviation     v minus the median of the recommendations about the same target
    1  rater_mean    mean of the recommendations issued by the rater
    2  rater_var     variance of those recommendations
    3  agreement     fraction of them close to the consensus
    4  consistency   variation of the rater's mean deviation across rounds

Two implementation choices affect how results read:

1. The deviation is signed. Bad-mouthing produces a negative deviation and
   ballot-stuffing a positive one; taking the absolute value would discard what
   separates the two attacks in the decision graph.

2. Everything is computed from what one Fog node sees in its history window,
   with no global view of the network. Otherwise the filter would use
   information that would not exist in operation.
"""

from __future__ import annotations

import numpy as np

FEATURE_NAMES = (
    "deviation",
    "rater_mean",
    "rater_var",
    "agreement",
    "consistency",
)


def _by_group(keys: np.ndarray, values: np.ndarray, func):
    """Apply `func` to the values of each distinct key. Returns a dict."""
    order = np.argsort(keys, kind="stable")
    k = keys[order]
    v = values[order]
    cuts = np.flatnonzero(np.diff(k)) + 1
    starts = np.concatenate(([0], cuts))
    ends = np.concatenate((cuts, [k.size]))
    return {int(k[a]): func(v[a:b]) for a, b in zip(starts, ends)}


def extract(
    rater: np.ndarray,
    target: np.ndarray,
    value: np.ndarray,
    round_idx: np.ndarray,
    window_idx: np.ndarray,
    batch_idx: np.ndarray,
    tau: float = 0.10,
) -> np.ndarray:
    """(len(batch_idx), 5) feature matrix, raw and unstandardised.

    `window_idx` is the history visible to the Fog node; `batch_idx` are the
    recommendations to classify now. The batch is always a subset of the window.
    """
    r_w, j_w, v_w, t_w = (
        rater[window_idx],
        target[window_idx],
        value[window_idx],
        round_idx[window_idx],
    )

    # consensus per target, within the window
    median = _by_group(j_w, v_w, np.median)
    deviation_w = v_w - np.array([median[int(j)] for j in j_w])

    # statistics per rater, within the window
    mean_r = _by_group(r_w, v_w, np.mean)
    var_r = _by_group(r_w, v_w, np.var)
    agree_r = _by_group(r_w, np.abs(deviation_w) <= tau, np.mean)

    # temporal consistency: spread of the rater's mean deviation across rounds.
    # A single round of history cannot show alternation, so the feature is
    # zero, which is why there is a warm-up.
    consist_r: dict[int, float] = {}
    order = np.lexsort((t_w, r_w))
    rc, tc, dc = r_w[order], t_w[order], deviation_w[order]
    cuts = np.flatnonzero(np.diff(rc)) + 1
    for a, b in zip(
        np.concatenate(([0], cuts)), np.concatenate((cuts, [rc.size]))
    ):
        tt, dd = tc[a:b], dc[a:b]
        rounds = np.unique(tt)
        if rounds.size < 2:
            consist_r[int(rc[a])] = 0.0
        else:
            means = np.array([dd[tt == r].mean() for r in rounds])
            consist_r[int(rc[a])] = float(np.std(means))

    # projection onto the batch
    r_b, j_b, v_b = rater[batch_idx], target[batch_idx], value[batch_idx]
    x = np.empty((batch_idx.size, 5))
    x[:, 0] = v_b - np.array([median[int(j)] for j in j_b])
    x[:, 1] = [mean_r[int(i)] for i in r_b]
    x[:, 2] = [var_r[int(i)] for i in r_b]
    x[:, 3] = [agree_r[int(i)] for i in r_b]
    x[:, 4] = [consist_r[int(i)] for i in r_b]
    return x


def standardize(x: np.ndarray) -> np.ndarray:
    """Column-wise z-score, guarded against constant columns.

    Needed because DPC uses Euclidean distance: unstandardised, the variance
    (order 1e-2) would be swamped by the mean (order 1e-1).
    """
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (x - mu) / sd


# ------------------------------------------------------- feature selection


def bimodality_coefficient(column: np.ndarray) -> float:
    """Sarle's bimodality coefficient: (skewness^2 + 1) / kurtosis.

    Computed without any label. It is 1/3 for a normal distribution and grows
    when the distribution has two modes; above 0.555 is the usual rule to
    suspect bimodality. A unimodal feature has no density valley for DPC to
    find and only adds distance.
    """
    x = column[np.isfinite(column)]
    n = x.size
    if n < 4:
        return 0.0
    sd = x.std()
    if sd < 1e-12:
        return 0.0
    z = (x - x.mean()) / sd
    skewness = float((z**3).mean())
    # Excess kurtosis, not raw: with the raw one the normal gives 0.165 instead
    # of 1/3 and a two-mode distribution falls below it.
    kurtosis = float((z**4).mean()) - 3.0
    # finite-sample correction, as in Sarle's formula
    denominator = kurtosis + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    if abs(denominator) < 1e-12:
        return 0.0
    return float((skewness**2 + 1.0) / denominator)


def bimodality_weights(x: np.ndarray, floor: float = 0.05) -> np.ndarray:
    """Weight per feature, proportional to the evidence of two modes.

    Normalised to sum to 1. The floor avoids zeroing a feature from a single
    batch of a few hundred points.
    """
    raw = np.array(
        [max(bimodality_coefficient(x[:, k]), 0.0) for k in range(x.shape[1])]
    )
    if raw.sum() <= 0:
        return np.ones(x.shape[1]) / x.shape[1]
    weights = raw / raw.sum()
    weights = np.maximum(weights, floor)
    return weights / weights.sum()
