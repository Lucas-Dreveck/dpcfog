"""Aggregated Trust Score and its error against the reference (hypothesis H2).

T_j is the mean of the recommendations about j that survived pre-filtering. H2
compares the MAE of T against T* with and without a filter.

A node can end up with no accepted recommendation at all. It then receives the
neutral prior 0.5, and the number of such nodes is reported next to the MAE: a
filter that lowers the error by discarding almost everything is refusing to
answer, and the `nodes_without_evidence` column exposes it.
"""

from __future__ import annotations

import numpy as np

NEUTRAL_PRIOR = 0.5


def trust_score(
    target: np.ndarray,
    value: np.ndarray,
    accepted: np.ndarray,
    n_nodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (estimated T per node, mask of nodes without accepted evidence)."""
    total = np.zeros(n_nodes)
    count = np.zeros(n_nodes)
    np.add.at(total, target[accepted], value[accepted])
    np.add.at(count, target[accepted], 1.0)
    no_evidence = count == 0
    t = np.where(no_evidence, NEUTRAL_PRIOR, total / np.where(count == 0, 1.0, count))
    return t, no_evidence


def mae(t_estimated: np.ndarray, t_ref: np.ndarray) -> float:
    return float(np.mean(np.abs(t_estimated - t_ref)))
