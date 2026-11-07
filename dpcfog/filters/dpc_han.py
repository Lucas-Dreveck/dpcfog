"""DPC as applied by Han et al. (2024), the closest prior work, adapted.

What Han et al. do (Section 4.2, steps 1 to 9):

1. Points are **two-dimensional**: (x, y), with x = neighbour identifier, 1 to q,
   and y = the trust value it recommended. There is no feature vector.

2. **K = 3, fixed.** Two **artificial anchors** enter the input together with
   the data: an upper bound standing for collusion and ballot-stuffing, and a
   lower bound standing for bad-mouthing.

3. Standard DPC: Gaussian rho, delta, d_c at the 1% to 2% percentile, decision
   value phi = rho * delta, and the three largest phi become centres.

4. **Decision rule:** recommendations in the same cluster as an anchor are
   false; those in the remaining cluster are accepted.

The authors explain why: a traditional DPC needs K in advance, but whether the
network is under attack is unknown, so K would vary between 1 and 3. The
anchors define a priori where the malicious regions are, and clustering only
decides who falls near them. With anchors the "normal" cluster exists by
construction, which avoids the 86.7% discard rate that plain partitioning shows
at mu = 0.

Two necessary adaptations, both declared:

**Where the anchors are.** The paper says "artificially fixed" and shows them
in a figure without numeric values. We use the ends of the normalised scale,
lower_bound = 0.0 and upper_bound = 1.0. This is our choice, and conclusions
drawn from this filter hold for the adaptation, not for the original method.

**The x coordinate.** In the paper it is the neighbour identifier, which makes
distances depend on an arbitrary numbering. `use_id_as_x=True` keeps the
original formulation (default); `False` uses only the value.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.distance import pdist, squareform

from .base import Batch, Filter
from .dpc import assign, delta_and_parent, density

N_HAN_CLUSTERS = 3


class HanDPCFilter(Filter):
    """Adaptation of the filtering algorithm of Han et al. (2024), Section 4.2."""

    name = "dpc_han"

    def __init__(
        self,
        lower_bound: float = 0.0,
        upper_bound: float = 1.0,
        dc_fraction: float = 0.02,
        use_id_as_x: bool = True,
        per_target: bool = True,
    ):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.dc_fraction = dc_fraction
        self.use_id_as_x = use_id_as_x
        # In the paper, node i filters the recommendations it received about
        # one neighbour j. Clustering recommendations about different targets
        # would mix values that are not comparable.
        self.per_target = per_target
        self._last: dict[str, Any] = {}

    # ------------------------------------------------------------ core

    def _decide_one_target(self, values: np.ndarray) -> np.ndarray:
        """Takes the recommendations about a single target; returns which are false."""
        n = values.size
        if n < 2:
            return np.zeros(n, dtype=bool)

        # artificial anchors first, data afterwards
        y = np.concatenate(
            ([self.lower_bound, self.upper_bound], values)
        )
        if self.use_id_as_x:
            x = np.arange(y.size, dtype=float)
            # the x axis must be on a scale comparable to the value, or the
            # distance becomes a function of the identifier only
            x = x / max(x.max(), 1.0)
            points = np.column_stack([x, y])
        else:
            points = y[:, None]

        d = squareform(pdist(points).astype(np.float32, copy=False))
        triangle = d[np.triu_indices(d.shape[0], k=1)]
        dc = float(np.percentile(triangle, 100.0 * self.dc_fraction))
        if dc <= 0:
            dc = float(triangle[triangle > 0].min()) if (triangle > 0).any() else 1.0

        rho = density(d, dc)
        delta, parent = delta_and_parent(d, rho)
        phi = rho * delta  # decision value, Eq. (14)

        k = min(N_HAN_CLUSTERS, d.shape[0])
        centers = np.argsort(-phi)[:k]
        cluster = assign(rho, parent, centers)

        # step 8: whatever falls in an anchor's cluster is a false recommendation
        anchor_clusters = {int(cluster[0]), int(cluster[1])}
        false = np.array([int(g) in anchor_clusters for g in cluster[2:]], dtype=bool)

        self._last = {
            "dc": dc, "rho": rho, "delta": delta, "phi": phi,
            "cluster": cluster, "centers": centers,
            "anchor_clusters": sorted(anchor_clusters),
            "n_clusters": int(np.unique(cluster).size),
        }
        return false

    # ------------------------------------------------------------ filter

    def flag(self, batch: Batch) -> np.ndarray:
        out = np.zeros(batch.x.shape[0], dtype=bool)
        if not self.per_target:
            return self._decide_one_target(batch.value)
        for target in np.unique(batch.target):
            m = batch.target == target
            out[m] = self._decide_one_target(batch.value[m])
        return out

    def score(self, batch: Batch) -> np.ndarray:
        """Continuous suspicion: distance from the value to the nearest anchor,
        negated. Being close to an anchor is suspicious."""
        v = batch.value
        d_low = np.abs(v - self.lower_bound)
        d_up = np.abs(v - self.upper_bound)
        return -np.minimum(d_low, d_up)

    def diagnostics(self) -> dict[str, Any]:
        return self._last
