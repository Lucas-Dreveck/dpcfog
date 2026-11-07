"""Real trust networks with published fraud labels (evaluation layers 1 and 2).

No parameter chosen by us enters the labels: they come from the protocol and
the files published by Kumar et al. (2018).
"""

from .networks import NETWORKS, RealNetwork, load
from .paths import data_root

__all__ = ["data_root", "NETWORKS", "RealNetwork", "load"]
