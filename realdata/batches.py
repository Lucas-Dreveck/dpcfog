"""From a real network to the batches the filters consume.

Same decomposition as the synthetic experiment, one batch per (Fog node, round)
with features taken from a history window, so that numbers measured on Bitcoin
OTC and in the simulation are comparable. The filter code does not change.

**Fog partition.** A real network carries no locality, and inventing one would
invent a scenario. By default a single Fog node sees the whole round. A
partition into F nodes exists as a parameter for the scalability analysis and
for Epinions, where it is declared arbitrary.

**Trust Score reference.** There is no real T*. As an approximation, the mean of
the ratings a target received **from users labelled benign** is used. It exists
only for targets that some benign user rated.
"""

from __future__ import annotations

import numpy as np

from dpcfog.features import extract
from dpcfog.filters import Batch

from .networks import RealNetwork


def reference_trust_score(net: RealNetwork) -> tuple[np.ndarray, np.ndarray]:
    """(approximate T* per target, mask of targets that have a reference).

    Only ratings from users labelled benign enter. A target no benign user
    rated has no reference and is left out of the MAE.
    """
    from_benign = net.evaluable & ~net.label
    total = np.zeros(net.n_targets)
    count = np.zeros(net.n_targets)
    np.add.at(total, net.target[from_benign], net.value[from_benign])
    np.add.at(count, net.target[from_benign], 1.0)
    has = count > 0
    t_ref = np.where(has, total / np.where(has, count, 1.0), np.nan)
    return t_ref, has


def build_batches(
    net: RealNetwork,
    window: int = 5,
    n_fog: int = 1,
    minimum: int = 10,
    fog_seed: int = 0,
) -> list[Batch]:
    """One batch per (Fog node, round). Rounds that are too short are dropped.

    `minimum` exists because DPC needs points for density to exist: a round
    with 8 ratings produces noise, not structure.
    """
    if n_fog == 1:
        fog_of_rater = np.zeros(net.n_raters, dtype=int)
    else:
        rng = np.random.default_rng(fog_seed)
        fog_of_rater = rng.integers(0, n_fog, size=net.n_raters)
    fog_of_edge = fog_of_rater[net.rater]

    batches: list[Batch] = []
    for t in range(int(net.round_idx.max()) + 1):
        in_round = net.round_idx == t
        if not in_round.any():
            continue
        start = max(0, t - window + 1)
        in_window = (net.round_idx >= start) & (net.round_idx <= t)
        for f in range(n_fog):
            batch_idx = np.flatnonzero(in_round & (fog_of_edge == f))
            if batch_idx.size < minimum:
                continue
            window_idx = np.flatnonzero(in_window & (fog_of_edge == f))
            x = extract(
                net.rater, net.target, net.value, net.round_idx,
                window_idx, batch_idx,
            )
            batches.append(
                Batch(
                    x=x,
                    value=net.value[batch_idx],
                    rater=net.rater[batch_idx],
                    target=net.target[batch_idx],
                    round_no=t,
                    fog=f,
                    label=net.label[batch_idx],
                    evaluable=net.evaluable[batch_idx],
                )
            )
    return batches


def split_by_time(batches, train_fraction: float = 0.4, warmup: int = 3):
    """Training strictly before evaluation, as in the synthetic experiment."""
    usable = [b for b in batches if b.round_no >= warmup]
    if not usable:
        return [], []
    rounds = sorted({b.round_no for b in usable})
    cut = rounds[max(1, int(round(train_fraction * len(rounds))))]
    return (
        [b for b in usable if b.round_no < cut],
        [b for b in usable if b.round_no >= cut],
    )
