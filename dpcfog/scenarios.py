"""Difficulty scenarios.

With the reference parameters, the deviation from the median separates the two
classes almost perfectly (|deviation| p99 = 0.139 for benign recommendations,
p01 = 0.113 for malicious ones), so a single fixed threshold reaches F1 = 0.99.
The other scenarios make the problem hard through mechanisms that exist in
Fog-IoT, and none of them was chosen after seeing which filter won:

    basic          reference parameters; a sanity floor.
    heterogeneous  each node observes with its own sigma, so a global threshold
                   is wrong by construction.
    stealthy       the attacker deviates by a bounded margin instead of
                   reporting an absolute value; the class tails overlap.
    adverse        heterogeneity, stealth and concentrated collusion together.
    calibrated     the only measured scenario: the five values are estimated on
                   Bitcoin OTC by `realdata.calibration`, with the labels of
                   Kumar et al. (2018). It is filled on demand so that importing
                   this module does not require the real data.
"""

from __future__ import annotations

from dataclasses import replace

from .config import Params

SCENARIOS = {
    "basic": {},
    "heterogeneous": {
        "heterogeneous_sigma": True,
    },
    "stealthy": {
        "stealthy": True,
    },
    "adverse": {
        "heterogeneous_sigma": True,
        "stealthy": True,
        "collusion_focus": 1.0,
        "victim_fraction": 0.05,
    },
    "calibrated": {},
}


def _calibrated(source: str = "otc") -> dict:
    """Measured parameters, cached because estimating reads the whole network."""
    if not SCENARIOS["calibrated"]:
        from realdata.calibration import estimate

        p, _ = estimate(source)
        SCENARIOS["calibrated"] = {
            "t_benign": p.t_benign,
            "t_malicious": p.t_malicious,
            "noise_sigma": p.noise_sigma,
            "sigma_range": p.sigma_range,
            "k_targets": p.k_targets,
        }
    return SCENARIOS["calibrated"]


def apply(base: Params, scenario: str) -> Params:
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario: {scenario}")
    fields = _calibrated() if scenario == "calibrated" else SCENARIOS[scenario]
    return replace(base, **fields)
