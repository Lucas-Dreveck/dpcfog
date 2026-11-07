"""Reproduce the quantitative claim of Figure 2 of Rodriguez & Laio (2014).

    python -m validation.rodriguez2014

This test separates "our DPC does badly on trust data" from "our DPC is
wrong". If the implementation reproduces the published result of the paper
that defines it, the failure on real data belongs to the data.

## The claim

From the paper (Science 344:1492-1496, benchmark of Fig. 2):

> "we performed the analysis by drawing 10,000 points from the distribution in
> Fig. 2A, considering as a reference the cluster assignment obtained on that
> sample. We then obtained reduced samples by retaining only a fraction of
> points and performed cluster assignment for each reduced sample
> independently. [...] The fraction of misclassified points remains well below
> 1% even for small samples containing 1000 points."

The distribution has five non-spherical, strongly overlapping peaks whose
maxima differ by almost an order of magnitude, over a 20% uniform background.

## What is ours

The paper shows the distribution in a figure without its parameters. The
values in `PEAKS` are our reconstruction satisfying the four declared
properties, so this reproduces the reported property, not the exact numbers.
The weights are equal and the WIDTH varies: the maximum of a 2-D Gaussian is
w / (2 pi sqrt(det Sigma)), so a 114x range in det(Sigma) gives about 10.7x in
the maxima. d_c follows the paper's rule of thumb (1% to 2% of the points as
neighbours), the 2% default of `DPCFilter`.

This test does not decide on its own; `validation.graded` does.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist, squareform

from dpcfog.filters.dpc import assign, dc_from_condensed, delta_and_parent, density

# Five peaks: (weight, centre, covariance).
_SCALES = (0.30, 0.55, 1.00, 1.80, 3.20)   # det varies ~114x -> maxima ~10.7x
_SHAPES = (
    ((1.00, 0.60), (0.60, 0.75)),
    ((0.90, -0.55), (-0.55, 0.85)),
    ((1.10, 0.20), (0.20, 0.70)),
    ((0.70, 0.35), (0.35, 1.15)),
    ((1.05, -0.40), (-0.40, 0.80)),
)
_CENTRES = ((0.0, 0.0), (5.4, 2.4), (2.4, 5.8), (-4.6, 3.9), (1.5, -5.2))
PEAKS = [
    (0.20, c, tuple(tuple(e * s for e in row) for row in f))
    for c, f, s in zip(_CENTRES, _SHAPES, _SCALES)
]
BACKGROUND_FRACTION = 0.20
LIMITS = (-7.0, 8.0)
SIZES = (10000, 8000, 6000, 4000, 2000, 1000)


def sample(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """n points and the peak that generated each; -1 for the uniform background.

    The generative ground truth exists because the paper's metric, disagreement
    between two runs, also punishes runs that both find the structure and only
    disagree at the border between overlapping peaks.
    """
    rng = np.random.default_rng(seed)
    n_bg = int(round(BACKGROUND_FRACTION * n))
    n_peak = n - n_bg
    weights = np.array([p[0] for p in PEAKS])
    weights = weights / weights.sum()
    which = rng.choice(len(PEAKS), size=n_peak, p=weights)
    points = np.empty((n, 2))
    for k, (_, centre, cov) in enumerate(PEAKS):
        m = which == k
        if m.any():
            points[: n_peak][m] = rng.multivariate_normal(
                np.asarray(centre), np.asarray(cov), size=int(m.sum())
            )
    points[n_peak:] = rng.uniform(LIMITS[0], LIMITS[1], size=(n_bg, 2))
    origin = np.concatenate([which, np.full(n_bg, -1)])
    return points, origin


def dpc(points: np.ndarray, n_centres: int | None = None) -> tuple:
    """DPC as the paper defines it. Returns (label, centres, rho, delta)."""
    cond = pdist(points).astype(np.float32)
    d = squareform(cond)
    dc = dc_from_condensed(cond, fraction=0.02)   # the paper's rule of thumb
    rho = density(d, dc)
    delta, parent = delta_and_parent(d, rho)
    gamma = rho * delta
    k = n_centres if n_centres else 5
    centres = np.argsort(-gamma)[:k]
    return assign(rho, parent, centres), centres, rho, delta


def disagreement(ref_points, ref_label, idx, label) -> float:
    """Fraction of the subsample's points in a cluster other than the reference.

    Labels from two runs are not directly comparable, so clusters are matched
    by maximum overlap with the Hungarian algorithm, the most favourable
    reading: any disagreement left is real.
    """
    target = ref_label[idx]
    k = max(target.max(), label.max()) + 1
    table = np.zeros((k, k), dtype=int)
    for a, b in zip(target, label):
        table[a, b] += 1
    row, col = linear_sum_assignment(-table)
    hits = table[row, col].sum()
    return 1.0 - hits / target.size


def run(seed: int = 0) -> list[dict]:
    full, origin = sample(SIZES[0], seed)
    ref, centres_ref, rho_ref, delta_ref = dpc(full)
    gamma = rho_ref * delta_ref
    ordered = np.sort(gamma)[::-1]
    # The decision graph must show five outstanding points. The jump between the
    # fifth and sixth largest gamma measures that without visual inspection.
    jump = ordered[4] / ordered[5] if ordered.size > 5 else float("nan")

    from sklearn.metrics import adjusted_rand_score

    from_peak = origin >= 0   # the uniform background belongs to no peak
    rng = np.random.default_rng(seed + 1000)
    rows = [{
        "n": SIZES[0], "disagreement": 0.0,
        "ari": adjusted_rand_score(origin[from_peak], ref[from_peak]),
        "n_clusters": int(np.unique(ref).size), "jump_5_6": jump,
    }]
    for n in SIZES[1:]:
        idx = rng.choice(SIZES[0], size=n, replace=False)
        lab, _, _, _ = dpc(full[idx])
        m = origin[idx] >= 0
        rows.append({
            "n": n,
            "disagreement": disagreement(full, ref, idx, lab),
            "ari": adjusted_rand_score(origin[idx][m], lab[m]),
            "n_clusters": int(np.unique(lab).size),
            "jump_5_6": float("nan"),
        })
    return rows


def main(seeds=(0, 1, 2, 3, 4)) -> int:
    print("Claim of the paper: the fraction of misclassified points stays WELL")
    print("BELOW 1% even for samples of 1000 points.\n")

    all_d: dict[int, list[float]] = {n: [] for n in SIZES}
    aris: dict[int, list[float]] = {n: [] for n in SIZES}
    jumps, clusters = [], []
    for s in seeds:
        for row in run(s):
            all_d[row["n"]].append(row["disagreement"])
            aris[row["n"]].append(row["ari"])
            if row["n"] == SIZES[0]:
                jumps.append(row["jump_5_6"])
                clusters.append(row["n_clusters"])

    print(f"{'n points':>9} {'disagreement':>13} {'worst':>8} {'passes':>7}   "
          f"{'ARI vs generating peak':>23}")
    all_ok = True
    for n in SIZES:
        v = np.array(all_d[n])
        a = np.array(aris[n])
        passes = v.max() < 0.01
        if n != SIZES[0]:
            all_ok &= bool(passes)
        mark = "--" if n == SIZES[0] else ("yes" if passes else "NO")
        print(f"{n:>9} {100 * v.mean():>12.2f}% {100 * v.max():>7.2f}% "
              f"{mark:>7}   {a.mean():>16.3f} +-{a.std():>5.3f}")

    print(f"\ndecision graph on the {SIZES[0]} sample: "
          f"{int(np.median(clusters))} clusters, "
          f"jump between 5th and 6th largest gamma = {np.mean(jumps):.2f}x")
    print(f"\nClaim of the paper (disagreement < 1%): "
          f"{'REPRODUCED' if all_ok else 'NOT REPRODUCED'}.")
    print("Recovered structure (ARI against the generating peak): "
          f"{np.mean([np.mean(aris[n]) for n in SIZES]):.3f} on average.")
    print()
    print("If ARI is high and disagreement does not reach 1%, DPC FINDS the")
    print("structure at every scale, and what varies between runs is the")
    print("assignment of border points where peaks overlap. The exact")
    print("reproduction depends on the paper's distribution, published without")
    print("parameters, so this test does NOT decide alone: see validation.graded.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
