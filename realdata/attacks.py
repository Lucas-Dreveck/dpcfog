"""Layer 2: profile injection attacks on real networks.

The attacks are the canonical models of **Mobasher, Burke, Bhaumik and Williams
(2007), ACM TOIT 7(4)**, with the two formal parameters of the area:

    attack size     fraction of injected profiles over the number of genuine
                    users; it is mu
    filler size     fraction of items each fake profile rates besides the target

Correspondence with the attacks studied here:

| Literature model | Attack |
|---|---|
| random / average, nuke mode | Bad-mouthing |
| random / average, push mode | Ballot-stuffing |
| bandwagon and segment       | Collusion |

The average attack rates filler items around each item's real mean, so the
profile looks normal: it is the stealthy variant.

## Why `realistic_filler` exists

Recommender-system benchmarks use movie datasets, where an ordinary user rates
tens or hundreds of items, and usual filler sizes (1% to 5% of the catalogue)
match that. In a trust network the **median is 2 ratings per user** on Bitcoin
OTC, and 3% of 5,858 targets would be 176: such a profile is detectable by its
degree alone.

    realistic_filler=False   follows the literature literally
    realistic_filler=True    draws the profile size from the network's own
                             degree distribution

## Labels in layer 2

The question here is "was this rating injected by us?", and the answer is known
for every rating: injected ones are positive and **every real rating is
negative by definition**. The only exception are ratings by users REV2 labels
fraudulent: malicious for another reason, they are left out of the metric so as
not to punish a filter for being right.

With ten attack items per profile, each fake profile issued 17 ratings against
a real median of 2, and injected ratings were 70% of the evaluated set. The
literature uses **one** attack item per attack, which is the default.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from .networks import RealNetwork

Model = Literal["random", "average", "bandwagon", "segment"]
Direction = Literal["push", "nuke"]

MODELS: tuple[Model, ...] = ("random", "average", "bandwagon", "segment")

# Declared correspondence, used in reports
EQUIVALENCE = {
    ("random", "nuke"): "bad_mouthing",
    ("average", "nuke"): "bad_mouthing_stealthy",
    ("random", "push"): "ballot_stuffing",
    ("average", "push"): "ballot_stuffing_stealthy",
    ("bandwagon", "push"): "collusion",
    ("segment", "push"): "collusion_segmented",
}


@dataclass(frozen=True)
class Attack:
    model: Model = "average"
    direction: Direction = "nuke"
    attack_size: float = 0.20          # fraction of injected profiles
    filler_size: float = 0.03          # fraction of items per profile
    realistic_filler: bool = True
    n_attack_targets: int = 1          # items the attack wants to push or nuke
    # Minimum degree of the attacked item. 0 draws uniformly among all items,
    # including never-rated ones. With a median degree of 2 on Bitcoin OTC this
    # puts hundreds of fabricated profiles on an item with 2 honest ratings,
    # and the attacker becomes the consensus. Values > 0 pick the target among
    # items with an audience, as a real attacker would. The default keeps the
    # frozen protocol.
    target_min_popularity: int = 0
    popular_fraction: float = 0.01     # bandwagon: most-rated items
    seed: int = 0

    @property
    def name(self) -> str:
        return EQUIVALENCE.get(
            (self.model, self.direction), f"{self.model}_{self.direction}"
        )


def _stats_per_target(net: RealNetwork) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(net.n_targets)
    total2 = np.zeros(net.n_targets)
    count = np.zeros(net.n_targets)
    np.add.at(total, net.target, net.value)
    np.add.at(total2, net.target, net.value**2)
    np.add.at(count, net.target, 1.0)
    mean = np.where(count > 0, total / np.where(count > 0, count, 1.0), net.value.mean())
    var = np.where(
        count > 1, total2 / np.where(count > 0, count, 1.0) - mean**2, net.value.var()
    )
    return mean, np.sqrt(np.clip(var, 1e-6, None))


def inject(net: RealNetwork, attack: Attack) -> RealNetwork:
    """Return a copy of the network with the attack profiles added.

    The original network is not modified. Fake profiles get new identifiers
    above the existing ones, and timestamps drawn within the network's period so
    that they fall in the same rounds as genuine traffic.
    """
    rng = np.random.default_rng(attack.seed)
    target_mean, target_sd = _stats_per_target(net)
    degrees = np.bincount(net.rater, minlength=net.n_raters)
    degrees = degrees[degrees > 0]

    n_fake = max(1, int(round(attack.attack_size * net.n_raters)))
    extreme_value = 1.0 if attack.direction == "push" else 0.0

    # items the attack wants to manipulate
    popularity = np.bincount(net.target, minlength=net.n_targets)
    if attack.target_min_popularity > 0:
        eligible = np.flatnonzero(popularity >= attack.target_min_popularity)
        if eligible.size == 0:
            raise ValueError(
                f"no item with degree >= {attack.target_min_popularity}; "
                f"maximum degree is {int(popularity.max())}"
            )
    else:
        eligible = np.arange(net.n_targets)
    attack_targets = rng.choice(
        eligible, size=min(attack.n_attack_targets, eligible.size), replace=False
    )

    # selected items I_S, which set bandwagon and segment apart
    if attack.model == "bandwagon":
        n_pop = max(1, int(round(attack.popular_fraction * net.n_targets)))
        selected = np.argsort(-popularity)[:n_pop]
    elif attack.model == "segment":
        # a cohesive segment: items sharing raters with the targets
        raters_of_target = np.isin(net.target, attack_targets)
        neighbours = np.unique(net.rater[raters_of_target])
        selected = np.unique(net.target[np.isin(net.rater, neighbours)])
    else:
        selected = np.array([], dtype=int)

    raters, targets, values, times = [], [], [], []
    t_min, t_max = net.time.min(), net.time.max()

    for k in range(n_fake):
        identifier = net.n_raters + k

        if attack.realistic_filler:
            n_filler = int(rng.choice(degrees))
        else:
            n_filler = max(
                1, int(round(attack.filler_size * net.n_targets))
            )
        n_filler = min(n_filler, net.n_targets - attack_targets.size)

        candidates = np.setdiff1d(np.arange(net.n_targets), attack_targets)
        filler = rng.choice(candidates, size=n_filler, replace=False)

        if attack.model == "average":
            # rating around each item's real mean: the profile looks normal
            v_filler = np.clip(
                rng.normal(target_mean[filler], target_sd[filler]),
                0.0, 1.0,
            )
        else:
            # rating around the global mean, without knowing the item
            v_filler = np.clip(
                rng.normal(net.value.mean(), net.value.std(), size=n_filler),
                0.0, 1.0,
            )

        # selected items get the extreme opposite to the target's
        if selected.size:
            n_sel = min(selected.size, max(1, n_filler // 4))
            sel = rng.choice(selected, size=n_sel, replace=False)
            v_sel = np.full(n_sel, 1.0 if attack.model == "bandwagon" else 0.0)
        else:
            sel = np.array([], dtype=int)
            v_sel = np.array([])

        target_k = np.concatenate([filler, sel, attack_targets])
        value_k = np.concatenate(
            [v_filler, v_sel, np.full(attack_targets.size, extreme_value)]
        )
        raters.append(np.full(target_k.size, identifier))
        targets.append(target_k)
        values.append(value_k)
        times.append(rng.uniform(t_min, t_max, size=target_k.size))

    new_rater = np.concatenate([net.rater] + raters).astype(np.int32)
    new_target = np.concatenate([net.target] + targets).astype(np.int32)
    new_value = np.concatenate([net.value] + values)
    new_time = np.concatenate([net.time] + times)
    n_injected = new_value.size - len(net)

    # Layer-2 label: positive = injected. Every real rating is negative by
    # definition; there is no unknown in this question.
    new_label = np.concatenate(
        [np.zeros(len(net), dtype=bool), np.ones(n_injected, dtype=bool)]
    )
    # Out of the metric: ratings by real fraudsters. Malicious for another
    # reason, counting them as negatives would punish the filter for being right.
    real_fraudster = net.evaluable & net.label
    new_evaluable = np.concatenate(
        [~real_fraudster, np.ones(n_injected, dtype=bool)]
    )

    order = np.argsort(new_time, kind="stable")
    from .networks import group_into_rounds

    return replace(
        net,
        name=f"{net.name}+{attack.name}",
        rater=new_rater[order],
        target=new_target[order],
        value=new_value[order],
        time=new_time[order],
        round_idx=group_into_rounds(
            new_time[order], net.meta.get("rounds_per_year", 12)
        ),
        label=new_label[order],
        evaluable=new_evaluable[order],
        n_raters=net.n_raters + n_fake,
        raw_rating=np.concatenate([net.raw_rating, np.full(n_injected, np.nan)])[
            order
        ],
        meta={
            **net.meta,
            "attack": attack.name,
            "model": attack.model,
            "direction": attack.direction,
            "attack_size": attack.attack_size,
            "realistic_filler": attack.realistic_filler,
            "injected_profiles": n_fake,
            "injected_ratings": int(n_injected),
            "models_source": "Mobasher et al. (2007), ACM TOIT 7(4)",
        },
    )
