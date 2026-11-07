"""Density Peaks Clustering as a pre-filter.

The algorithm is that of Rodriguez and Laio (2014): local density rho, distance
to the nearest denser neighbour delta, centres read from the rho x delta
decision graph, and a halo separating core from border.

Three criteria for choosing d_c are available:

    heuristic  percentile of the distances, 1% to 2% of the points as neighbours
               (Rodriguez and Laio, 2014)
    entropy    minimum potential entropy of the data field (Wang et al., 2016),
               with no free parameter
    fixed      an imposed value, used in the 0.5% to 5% sensitivity analysis

Two decision rules are available:

    partition  clusters and discards everything outside the legitimate cluster.
               The legitimate cluster is the one holding the density peak
               (default) or the largest one. This assumes an honest majority:
               if coordinated attackers form the largest group, the filter
               inverts.
    outlier    does not partition. For each point it asks whether the nearest
               denser neighbour lies beyond d_c ("high delta and low rho").

A clustering algorithm always partitions, even a homogeneous cloud: with mu = 0,
no attacker at all, the partition rule discards 86.7% of the recommendations
and the outlier rule 8.0%.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.distance import pdist, squareform

from ..features import bimodality_weights, standardize
from .base import Batch, Filter


# ---------------------------------------------------------------- DPC core


def distance_matrix(x: np.ndarray, dtype=np.float32) -> np.ndarray:
    """Pairwise distance matrix.

    float32 because the O(n^2) matrix dominates the filter's memory: in float64
    a batch of 1,701 recommendations takes 23.1 MB for it alone, in float32
    11.6 MB. Distances between standardised points are of the order of units,
    and float32's seven decimal digits are ample for a neighbourhood comparison.
    """
    return squareform(pdist(x, metric="euclidean").astype(dtype, copy=False))


def condensed_distances(x: np.ndarray, dtype=np.float32) -> np.ndarray:
    """Upper triangle only, without the square matrix. Half the memory."""
    return pdist(x, metric="euclidean").astype(dtype, copy=False)


def dc_from_condensed(condensed: np.ndarray, fraction: float = 0.02) -> float:
    """d_c from the condensed vector, without materialising indices.

    Two corrections to the naive form:

    1. It does not materialise `np.triu_indices_from`, whose two int64 index
       vectors cost 23 MB in a batch of 1,701, more than the distance matrix.

    2. **Zero distances are ignored.** In real networks ratings are discrete and
       most users have one or two ratings: a batch of 700 points has 51 distinct
       deviation values, 93% ties. The 2% percentile then falls exactly on zero,
       d_c = 0, and density becomes a division by zero. The percentile over the
       positive distances measures what the Rodriguez and Laio (2014) heuristic
       intends, the radius holding 1% to 2% of the neighbours, without being
       dominated by coincident points.
    """
    if condensed.size == 0:
        return 1.0
    positive = condensed[condensed > 0]
    if positive.size == 0:
        return 1.0
    return float(np.percentile(positive, 100.0 * fraction))


def dc_heuristic(d: np.ndarray, fraction: float = 0.02) -> float:
    """d_c such that each point has, on average, `fraction` of the points as neighbours.

    Kept for use with a ready square matrix (tests, diagnostics). On the hot
    path the filter uses `dc_from_condensed`, which allocates no indices.
    """
    n = d.shape[0]
    if n < 2:
        return 1.0
    return dc_from_condensed(d[np.triu_indices(n, k=1)], fraction)


def dc_entropy(d: np.ndarray, n_steps: int = 40) -> float:
    """d_c from the minimum potential entropy of the data field (Wang et al., 2016).

    The data field assigns each point a potential phi_i summing Gaussian kernels
    of width sigma. The Shannon entropy of the normalised potential is minimal
    when sigma best describes the structure. The classic relation
    d_c = 3 sigma / sqrt(2) links the impact factor to the influence radius.
    """
    triangle = d[np.triu_indices_from(d, k=1)]
    if triangle.size == 0:
        return 1.0
    lo = max(np.percentile(triangle, 0.5), 1e-9)
    hi = np.percentile(triangle, 50.0)
    if hi <= lo:
        return float(hi if hi > 0 else 1.0)

    sigmas = np.linspace(lo, hi, n_steps)
    off_diagonal = ~np.eye(d.shape[0], dtype=bool)
    entropies = np.empty(n_steps)
    for k, s in enumerate(sigmas):
        phi = np.where(off_diagonal, np.exp(-((d / s) ** 2)), 0.0).sum(axis=1)
        z = phi.sum()
        if z <= 0:
            entropies[k] = np.inf
            continue
        p = phi / z
        p = np.where(p > 0, p, 1.0)
        entropies[k] = float(-(p * np.log(p)).sum())
    sigma = float(sigmas[int(np.argmin(entropies))])
    return sigma * 3.0 / np.sqrt(2.0)


def density(d: np.ndarray, dc: float, kernel: str = "gaussian") -> np.ndarray:
    """rho_i. Gaussian kernel by default: in a small batch the cut-off kernel
    produces many ties in rho, and ties break the construction of delta.

    Written with in-place operations. The direct form chains four temporary
    n x n matrices plus a boolean mask (49.2 MB measured for a batch of 1,701,
    against 23.1 MB for the distance matrix). Here there is a single temporary,
    and the diagonal is zeroed instead of masked.
    """
    if kernel == "cutoff":
        tmp = (d < dc).astype(np.float32)
    else:
        tmp = d.astype(np.float32, copy=True)
        tmp /= dc
        tmp *= tmp
        np.negative(tmp, out=tmp)
        np.exp(tmp, out=tmp)
    np.fill_diagonal(tmp, 0.0)
    return tmp.sum(axis=1, dtype=np.float64)


def delta_and_parent(d: np.ndarray, rho: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """delta_i and the index of the nearest neighbour with higher density.

    The highest-rho point has no parent; it gets delta equal to the largest
    distance in the set, as in Rodriguez and Laio (2014), and itself as parent.
    """
    n = d.shape[0]
    order = np.argsort(-rho, kind="stable")
    delta = np.empty(n)
    parent = np.arange(n)
    delta[order[0]] = d[order[0]].max()
    for position in range(1, n):
        i = order[position]
        denser = order[:position]
        distances = d[i, denser]
        winner = int(np.argmin(distances))
        delta[i] = distances[winner]
        parent[i] = denser[winner]
    return delta, parent


def choose_centers(
    rho: np.ndarray, delta: np.ndarray, maximum: int = 8
) -> np.ndarray:
    """Centres from the largest jump in gamma = rho x delta, both normalised.

    No number of clusters is given: that is the DPC property the work relies on,
    and reintroducing k as a parameter would be inconsistent.
    """
    n = rho.size
    if n < 3:
        return np.array([int(np.argmax(rho))])
    rn = (rho - rho.min()) / (np.ptp(rho) + 1e-12)
    dn = (delta - delta.min()) / (np.ptp(delta) + 1e-12)
    gamma = rn * dn
    # The highest-rho point gets delta equal to the largest distance by
    # construction, and therefore always the largest gamma. Leaving it in the
    # race makes the largest jump fall between it and the runner-up, and the
    # search ends at k = 1 in every batch. It is a centre by definition: it
    # leaves the ranking and the jump is searched among the others.
    peak = int(np.argmax(rho))
    rest = np.array([i for i in np.argsort(-gamma, kind="stable") if i != peak])
    top = min(maximum, rest.size)
    if top < 2:
        return np.array([peak] + rest[:top].tolist())
    # The jump is measured on a log scale. Gamma is a product and spans orders
    # of magnitude, so the absolute difference is dominated by the top of the
    # list. With three separated blobs the candidates' gammas are
    # 0.522 / 0.157 / 0.004: the largest absolute jump falls between the first
    # and second (0.366), while the real cliff is between the second and third
    # (ratio 36 against 3).
    log_gamma = np.log(gamma[rest[:top]] + 1e-12)
    jumps = log_gamma[:-1] - log_gamma[1:]
    k = int(np.argmax(jumps)) + 1
    return np.array([peak] + rest[:k].tolist())


def assign(rho: np.ndarray, parent: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Each point inherits the cluster of its nearest denser neighbour, densest first."""
    label = np.full(rho.size, -1, dtype=int)
    for g, c in enumerate(centers):
        label[c] = g
    for i in np.argsort(-rho, kind="stable"):
        if label[i] == -1:
            label[i] = label[parent[i]]
    return label


def halo(d: np.ndarray, rho: np.ndarray, label: np.ndarray, dc: float) -> np.ndarray:
    """True for a border point, as defined by Rodriguez and Laio (2014).

    For each cluster, the border density rho_b is the highest density among the
    points that have, within d_c, a neighbour from another cluster. Points below
    it are border, not core.
    """
    border = np.zeros(rho.size, dtype=bool)
    for g in np.unique(label):
        in_cluster = label == g
        foreign_neighbour = ((d < dc) & (~in_cluster)[None, :]).any(axis=1)
        frontier = in_cluster & foreign_neighbour
        if not frontier.any():
            continue
        rho_b = rho[frontier].max()
        border |= in_cluster & (rho < rho_b)
    return border


# ---------------------------------------------------------------- the filter


class DPCFilter(Filter):
    name = "dpc"

    def __init__(
        self,
        dc_criterion: str = "heuristic",
        dc_fraction: float = 0.02,
        kernel: str = "gaussian",
        use_halo: bool = True,
        fixed_dc: float | None = None,
        granularity: str = "recommendation",
        legitimate_criterion: str = "peak",
        rule: str = "partition",
        delta_factor: float = 1.0,
        weights: str | np.ndarray = "equal",
    ):
        self.dc_criterion = dc_criterion
        self.dc_fraction = dc_fraction
        self.kernel = kernel
        self.use_halo = use_halo
        self.fixed_dc = fixed_dc
        self.granularity = granularity
        self.legitimate_criterion = legitimate_criterion
        self.rule = rule
        self.delta_factor = delta_factor
        self.weights = weights
        self._last: dict[str, Any] = {}

    def _dc(self, d: np.ndarray) -> float:
        if self.dc_criterion == "fixed":
            if self.fixed_dc is None:
                raise ValueError("dc_criterion='fixed' requires fixed_dc")
            return self.fixed_dc
        if self.dc_criterion == "entropy":
            return dc_entropy(d)
        return dc_heuristic(d, self.dc_fraction)

    def _dc_condensed(self, condensed: np.ndarray) -> float:
        if self.dc_criterion == "fixed":
            if self.fixed_dc is None:
                raise ValueError("dc_criterion='fixed' requires fixed_dc")
            return self.fixed_dc
        if self.dc_criterion == "entropy":
            # The potential entropy needs the square matrix; it is the only
            # criterion that cannot avoid the early expansion.
            return dc_entropy(squareform(condensed))
        return dc_from_condensed(condensed, self.dc_fraction)

    def flag(self, batch: Batch) -> np.ndarray:
        n = batch.x.shape[0]
        if n < 3:
            return np.zeros(n, dtype=bool)

        if self.granularity == "recommender":
            # Four of the five features are properties of the rater, not of the
            # recommendation: in feature space the K recommendations of one
            # rater collapse into a tight cloud. Aggregating per rater first
            # makes that structure explicit and cuts the cost from O(500^2) to
            # O(50^2).
            raters, inverse = np.unique(batch.rater, return_inverse=True)
            if raters.size < 3:
                return np.zeros(n, dtype=bool)
            xa = np.vstack(
                [batch.x[inverse == k].mean(axis=0) for k in range(raters.size)]
            )
            suspicious = self._decide(xa)
            return suspicious[inverse]

        return self._decide(batch.x)

    def _decide(self, x: np.ndarray) -> np.ndarray:
        z = standardize(x)
        if isinstance(self.weights, str):
            if self.weights == "bimodal":
                # Rescale each axis by the evidence that it has two modes. No
                # label is involved: it is a property of the batch distribution.
                z = z * np.sqrt(bimodality_weights(z) * z.shape[1])
        else:
            # Explicit weight vector, one per feature (used by the oracle
            # ceiling in `analysis/feature_weighting.py`). The rescaling must
            # come after `standardize`, or the column-wise z-score undoes it.
            w = np.asarray(self.weights, dtype=float)
            if w.size != z.shape[1]:
                raise ValueError(
                    f"weights has {w.size} entries for {z.shape[1]} features"
                )
            z = z * np.sqrt(w / w.sum() * z.shape[1])
        # Minimal-memory path: the condensed vector serves for d_c and only then
        # becomes a square matrix, released as soon as possible.
        condensed = condensed_distances(z)
        dc = self._dc_condensed(condensed)
        d = squareform(condensed)
        del condensed
        if dc <= 0:
            dc = float(d[d > 0].min()) if (d > 0).any() else 1.0

        rho = density(d, dc, self.kernel)
        delta, parent = delta_and_parent(d, rho)

        if self.rule == "outlier":
            # "Recommendations in low-density regions or far from the legitimate
            # centres (high delta and low rho)". Does not partition: asks whether
            # the nearest denser neighbour lies beyond d_c, that is, whether the
            # point is an island. Without any attack (mu = 0) the partition rule
            # discards 87% of the recommendations; this rule 0.5% to 9%.
            self._last = {
                "n_points": int(z.shape[0]), "dc": dc, "rho": rho, "delta": delta,
                "cluster": np.zeros(rho.size, dtype=int), "centers": np.array([0]),
                "legitimate_cluster": 0, "n_clusters": 1,
            }
            return delta > self.delta_factor * dc

        centers = choose_centers(rho, delta)
        cluster = assign(rho, parent, centers)

        # Which cluster is the honest consensus? Two readings:
        #   cardinality  the largest cluster (honest-majority assumption).
        #   peak         the cluster holding the highest-rho point, DPC's own
        #                reading: consensus is the density peak, not the count.
        # In the adverse scenario, cardinality picks the wrong cluster: the
        # diffuse cluster of 332 points with 17% malicious beats the tight core
        # of 168 points with 1%. F1 goes from 0.246 to 0.332 when switching,
        # hence "peak" as the default.
        if self.legitimate_criterion == "cardinality":
            legitimate = int(np.argmax(np.bincount(cluster, minlength=centers.size)))
        else:
            legitimate = int(cluster[int(np.argmax(rho))])

        malicious = cluster != legitimate
        if self.use_halo:
            malicious |= halo(d, rho, cluster, dc) & (cluster == legitimate)

        self._last = {
            "n_points": int(z.shape[0]),
            "dc": dc,
            "rho": rho,
            "delta": delta,
            "cluster": cluster,
            "centers": centers,
            "legitimate_cluster": legitimate,
            "n_clusters": int(centers.size),
        }
        return malicious

    def score(self, batch: Batch) -> np.ndarray:
        """Continuous suspicion from DPC's own quantities.

        "High delta and low rho": standardising both and taking the difference
        gives a single rankable score, without the binary cut, which lets DPC's
        signal be compared with the other filters' independently of thresholds.
        """
        self.flag(batch)
        d = self._last
        if not d:
            return np.zeros(batch.x.shape[0])
        rho, delta = d["rho"], d["delta"]
        z = lambda v: (v - v.mean()) / (v.std() + 1e-12)
        raw = z(delta) - z(rho)
        if self.granularity == "recommender":
            _, inverse = np.unique(batch.rater, return_inverse=True)
            if raw.size == inverse.max() + 1:
                return raw[inverse]
        return raw

    def diagnostics(self) -> dict[str, Any]:
        return self._last
