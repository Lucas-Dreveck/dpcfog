"""Pre-filtering of malicious recommendations in Fog-IoT with Density Peaks Clustering.

Module map:

    config      simulation parameters and the frozen experimental grid
    scenarios   difficulty scenarios, including the one calibrated on Bitcoin OTC
    generator   synthetic Fog-IoT network, attacks and per-recommendation ground truth
    features    the five per-recommendation features fed to the filters
    filters     DPC (the proposed filter) and the baselines
    trust       aggregated Trust Score and its error against the reference
    metrics     detection, ranking, time and memory measurements
    experiment  evaluation loop over the malicious fraction mu and seeds
"""

__version__ = "0.1.0"
