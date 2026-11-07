"""Simulation parameters and the frozen experimental grid.

Every deviation from the reference values is documented next to the field.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal

Attack = Literal[
    "bad_mouthing", "ballot_stuffing", "collusion", "on_off", "mixed"
]

ATTACKS: tuple[Attack, ...] = (
    "bad_mouthing", "ballot_stuffing", "collusion", "on_off", "mixed"
)


@dataclass(frozen=True)
class Params:
    # --- network
    n_nodes: int = 100               # N
    mu: float = 0.20                 # fraction of malicious nodes, 0.10 to 0.40
    k_targets: int = 10              # K -- nodes recommended by each node per round
    n_rounds: int = 50               # R
    noise_sigma: float = 0.05        # std of benign observation noise

    # --- reference Trust Score T*
    t_benign: tuple[float, float] = (0.7, 1.0)
    t_malicious: tuple[float, float] = (0.0, 0.3)

    # --- Fog-IoT heterogeneity (see scenarios.py)
    heterogeneous_sigma: bool = False  # each node observes with its own sigma
    sigma_range: tuple[float, float] = (0.02, 0.20)

    # --- attack
    attack: Attack = "mixed"
    stealthy: bool = False           # deviates by a bounded margin, not a fixed value
    stealthy_deviation: tuple[float, float] = (0.15, 0.30)
    collusion_focus: float = 0.5     # share of the K slots aimed at the common victims
    victim_fraction: float = 0.15    # size of the collusion victim set
    attack_prob: float = 1.0         # 1.0 = no camouflage; < 1 = intermittent attack
    # On-Off: the node alternates phases. In each cycle of `on_off_period`
    # rounds it attacks during `active_fraction` of the cycle and behaves
    # honestly for the rest, rebuilding reputation between attacks. A per
    # recommendation `attack_prob` < 1 would add noise without the temporal
    # pattern that the variance and consistency features measure.
    on_off_period: int = 10          # rounds in one full cycle
    active_fraction: float = 0.4     # fraction of the cycle spent attacking
    collusion_size: float = 0.5      # fraction of malicious nodes acting in coordination
    bad_mouthing_value: float = 0.3  # value reported about an attacked benign node
    ballot_stuffing_value: float = 0.9  # value reported about a malicious ally

    # --- Fog architecture
    n_fog: int = 2                   # Fog nodes; each one filters its own neighbourhood
    consensus_window: int = 5        # rounds of history used by the features
    warmup_rounds: int = 5           # minimum history before evaluation

    # --- reproducibility
    seed: int = 0

    def as_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        if not 0.0 <= self.mu <= 1.0:
            raise ValueError(f"mu outside [0,1]: {self.mu}")
        if self.k_targets >= self.n_nodes:
            raise ValueError("K must be smaller than N")
        if self.n_fog < 1 or self.n_fog > self.n_nodes:
            raise ValueError("invalid n_fog")
        if self.n_rounds <= self.warmup_rounds:
            raise ValueError("R must exceed the warm-up rounds")
        if self.attack not in ATTACKS:
            raise ValueError(f"unknown attack: {self.attack}")


# Frozen experimental grid.
MU_GRID: tuple[float, ...] = (0.10, 0.20, 0.30, 0.40)
SEED_GRID: tuple[int, ...] = tuple(range(30))
DC_SENSITIVITY_GRID: tuple[float, ...] = tuple(
    round(0.005 * i, 3) for i in range(1, 11)
)  # 0.5% to 5.0% in steps of 0.5%
