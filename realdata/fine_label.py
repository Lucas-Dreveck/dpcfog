"""Per-recommendation label in layer 2, next to the per-profile label.

`inject` marks every rating of a fabricated profile as malicious, including the
filler, which is drawn from the real distribution precisely to be
indistinguishable from an honest rating. On Bitcoin OTC with mu = 0.20: 8,324
labelled, 963 with a distorted value.

They are two questions, and the ranking of the filters changes between them:

    per profile         "did this rating come from a fabricated account?"
    per recommendation  "was this value distorted?"
"""

from __future__ import annotations

import numpy as np


def label_per_recommendation(net) -> np.ndarray:
    """True only for injected ratings whose value was actually distorted.

    The attack picks `n_attack_targets` items and gives them the extreme value;
    the rest of the profile is filler. The distorted rating is recognised by its
    target: the one that receives injections from practically all fabricated
    profiles.
    """
    inj = net.label
    if not inj.any():
        return np.zeros(inj.size, dtype=bool)
    targets, counts = np.unique(net.target[inj], return_counts=True)
    attacked = set(targets[counts >= 0.9 * counts.max()].tolist())
    return inj & np.array([a in attacked for a in net.target], dtype=bool)
