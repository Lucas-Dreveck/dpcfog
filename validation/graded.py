"""Is the DPC implementation right? A graded test, from trivial to hard.

    python -m validation.graded

The reproduction of Figure 2 of Rodriguez & Laio (`rodriguez2014.py`) gives an
ambiguous signal, because the paper publishes the distribution without its
parameters. This test separates a defect in our DPC from a defect in our
reconstruction. Every case has a ground truth by construction. If DPC gets the
easy cases right and fails the hard ones in the expected order, the
implementation is right.

Metric: adjusted Rand index (ARI), 1 for a partition identical to the ground
truth and 0 for a random one. The comparison is against k-means with the
correct K, which is the comparison the paper itself makes in Figs. S3 and S4:

> "Even if the K-means optimization is performed with use of the correct value
> of K, the assignations are, in most of the cases, not compliant with visual
> intuition."
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.metrics import adjusted_rand_score

from dpcfog.filters.dpc import assign, dc_from_condensed, delta_and_parent, density

N = 1000
SEEDS = (0, 1, 2, 3, 4)


def dpc(points: np.ndarray, k: int, dc_fraction: float = 0.02):
    cond = pdist(points).astype(np.float32)
    d = squareform(cond)
    dc = dc_from_condensed(cond, fraction=dc_fraction)
    rho = density(d, dc)
    delta, parent = delta_and_parent(d, rho)
    gamma = rho * delta
    centres = np.argsort(-gamma)[:k]
    ordered = np.sort(gamma)[::-1]
    jump = ordered[k - 1] / ordered[k] if ordered.size > k else float("nan")
    return assign(rho, parent, centres), jump


# ------------------------------------------------------------------- cases


def separated_blobs(seed):
    x, y = make_blobs(n_samples=N, centers=3, cluster_std=0.6,
                      random_state=seed)
    return x, y, 3


def overlapping_blobs(seed):
    x, y = make_blobs(n_samples=N, centers=3, cluster_std=1.8,
                      random_state=seed)
    return x, y, 3


def unequal_densities(seed):
    """Peaks with very different densities, the property of Fig. 2."""
    rng = np.random.default_rng(seed)
    sizes = [int(N * f) for f in (0.60, 0.25, 0.15)]
    centres = [(0, 0), (5, 4), (-5, 4)]
    parts, lab = [], []
    for k, (n, c) in enumerate(zip(sizes, centres)):
        parts.append(rng.normal(c, 0.9, size=(n, 2)))
        lab.append(np.full(n, k))
    return np.vstack(parts), np.concatenate(lab), 3


def anisotropic(seed):
    """Non-spherical: stretched and rotated blobs."""
    x, y = make_blobs(n_samples=N, centers=3, cluster_std=0.7,
                      random_state=seed)
    t = np.array([[0.6, -0.63], [-0.4, 0.85]])
    return x @ t, y, 3


def moons(seed):
    x, y = make_moons(n_samples=N, noise=0.06, random_state=seed)
    return x, y, 2


def circles(seed):
    x, y = make_circles(n_samples=N, noise=0.05, factor=0.45,
                        random_state=seed)
    return x, y, 2


CASES = [
    ("separated blobs", separated_blobs, "trivial: any method gets it"),
    ("overlapping blobs", overlapping_blobs, "strong overlap"),
    ("unequal densities", unequal_densities, "property of Fig. 2"),
    ("anisotropic", anisotropic, "non-spherical; k-means usually fails"),
    ("two moons", moons, "arbitrary shape; the paper claims to handle it"),
    ("concentric circles", circles, "arbitrary shape, extreme case"),
]


def main() -> int:
    print(f"{'case':>20} {'DPC ARI':>18} {'k-means ARI':>12} "
          f"{'gamma jump':>11}")
    results = {}
    for name, generate, _ in CASES:
        aris, kms, jumps = [], [], []
        for s in SEEDS:
            x, y, k = generate(s)
            lab, jump = dpc(x, k)
            aris.append(adjusted_rand_score(y, lab))
            kms.append(adjusted_rand_score(
                y, KMeans(n_clusters=k, n_init=10, random_state=s).fit_predict(x)
            ))
            jumps.append(jump)
        a, km = np.array(aris), np.array(kms)
        results[name] = (a.mean(), km.mean())
        print(f"{name:>20} {a.mean():>10.3f} +-{a.std():>5.3f} "
              f"{km.mean():>12.3f} {np.mean(jumps):>10.2f}x")

    print("\nReading:")
    # An absolute threshold does not work: overlapping blobs with std 1.8 are
    # genuinely ambiguous, and k-means also stays at 0.69 there. What separates
    # an implementation defect from data difficulty is the comparison with
    # k-means run with the correct K.
    trivial = results["separated blobs"][0]
    density_ok = results["unequal densities"][0]
    if trivial > 0.95 and density_ok > 0.95:
        print(f"  Cases with clear density structure: ARI {trivial:.3f} and "
              f"{density_ok:.3f}.")
        print("  The implementation finds the peaks when they exist, including")
        print("  with very unequal densities, the property of Fig. 2.")
    else:
        print("  WARNING: a case with clear structure failed. The defect is ours.")

    wins = [c for c in results if results[c][0] > results[c][1] + 0.05]
    losses = [c for c in results if results[c][1] > results[c][0] + 0.05]
    if wins:
        print(f"  Beats k-means with the correct K on: {', '.join(wins)},")
        print("  the behaviour the paper reports in Figs. S3 and S4.")
    if losses:
        print(f"  Loses to k-means on: {', '.join(losses)}.")
    print("  Concentric circles fail through a known limitation of the method:")
    print("  density is uniform along each ring, so there is no gradient for the")
    print("  nearest denser neighbour to follow.")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
