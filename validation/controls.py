"""Controls and confidence intervals for layer 1.

    python -m validation.controls

**Negative control.** The same filters, on the same batches, with shuffled
labels. Every AUC must fall to 0.5. A filter that stays above it has a leak:
label information reaching it through a path that should not exist.

**Positive control.** A filter that receives the label itself must reach
AUC 1.0. If it does not, the defect is in the measurement harness.

**Confidence interval.** Bootstrap over the evaluated recommendations, 1,000
resamples, 2.5 and 97.5 percentiles. The paper's tables use the per-rater
cluster bootstrap of `paper_tables.py`, which is wider; this one is a quick
check.
"""

from __future__ import annotations

import sys

import numpy as np

from dpcfog import metrics
from dpcfog.experiment import filter_catalog
from dpcfog.filters.base import Batch, Filter
from realdata.batches import build_batches, split_by_time
from realdata.networks import load

N_BOOTSTRAP = 1000
SEED = 20260827


class OracleFilter(Filter):
    """Sees the label. Not a filter: the positive control of the harness."""

    name = "oracle"

    def flag(self, batch: Batch) -> np.ndarray:
        return batch.label.copy()

    def score(self, batch: Batch) -> np.ndarray:
        return batch.label.astype(float)


def bootstrap(scores: np.ndarray, y: np.ndarray, n: int = N_BOOTSTRAP):
    """2.5 and 97.5 percentiles of the AUC, resampling with replacement."""
    rng = np.random.default_rng(SEED)
    size = y.size
    out = np.empty(n)
    valid = 0
    for i in range(n):
        idx = rng.integers(0, size, size=size)
        yy = y[idx]
        if yy.all() or not yy.any():
            continue
        out[valid] = metrics.ranking(scores[idx], yy)["auc"]
        valid += 1
    if valid < n // 2:
        return float("nan"), float("nan")
    v = out[:valid]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def prepare(network: str):
    net = load(network)
    train, evaluation = split_by_time(build_batches(net, window=5, n_fog=1))
    m = np.concatenate([b.evaluable for b in evaluation])
    y = np.concatenate([b.label for b in evaluation])
    return train, evaluation, m, y


def scores_of(name: str, factory, train, evaluation) -> np.ndarray | None:
    f = factory()
    if f.supervised:
        f.fit(train)
    try:
        parts = [f.score(b) for b in evaluation]
    except Exception:
        return None
    if any(p is None for p in parts):
        return None
    return np.concatenate(parts)


def shuffle(evaluation, seed: int, scope: str = "batch"):
    """Same batches, permuted labels. Two scopes, and the difference matters.

    `batch`   permutes within each batch, keeping its number of positives. It
              destroys the link between a recommendation and its label but not
              the variation of the fraud rate across batches.

    `global`  permutes over all evaluable recommendations at once, destroying
              the per-batch structure as well.

    With `batch`, rev2 stayed at 0.622 on Bitcoin OTC without using any label:
    the fraud rate varies across batches and rev2's score varies with it. The
    difference between the two scopes measures how much of each filter's
    performance comes from telling recommendations apart and how much from
    telling batches apart.
    """
    rng = np.random.default_rng(seed)
    if scope == "global":
        everything = np.concatenate([b.label[b.evaluable] for b in evaluation])
        everything = rng.permutation(everything)
        new, i = [], 0
        for b in evaluation:
            r = b.label.copy()
            where = np.flatnonzero(b.evaluable)
            r[where] = everything[i:i + where.size]
            i += where.size
            new.append(
                Batch(x=b.x, value=b.value, rater=b.rater, target=b.target,
                      round_no=b.round_no, fog=b.fog, label=r, evaluable=b.evaluable)
            )
        return new
    new = []
    for b in evaluation:
        r = b.label.copy()
        where = np.flatnonzero(b.evaluable)
        if where.size > 1:
            r[where] = rng.permutation(r[where])
        new.append(
            Batch(x=b.x, value=b.value, rater=b.rater, target=b.target,
                  round_no=b.round_no, fog=b.fog, label=r, evaluable=b.evaluable)
        )
    return new


def main(networks=("otc", "alpha")) -> int:
    problems = 0
    for network in networks:
        train, evaluation, m, y = prepare(network)
        factories = dict(filter_catalog(0))
        factories["oracle"] = OracleFilter

        print(f"\n{'=' * 78}\n{network}: {int(m.sum())} evaluable recommendations, "
              f"{100 * y[m].mean():.1f}% malicious\n{'=' * 78}")
        print(f"{'filter':>18} {'AUC':>7} {'95% CI':>17} "
              f"{'shuf. batch':>12} {'shuf. global':>13} {'batch only':>11}")

        shuf_batch = shuffle(evaluation, SEED, "batch")
        y_batch = np.concatenate([b.label for b in shuf_batch])
        shuf_glob = shuffle(evaluation, SEED, "global")
        y_glob = np.concatenate([b.label for b in shuf_glob])

        for name in sorted(factories):
            sc = scores_of(name, factories[name], train, evaluation)
            if sc is None:
                print(f"{name:>18}   score unavailable")
                continue
            auc = metrics.ranking(sc[m], y[m])["auc"]
            lo, hi = bootstrap(sc[m], y[m])

            s1 = scores_of(name, factories[name], train, shuf_batch)
            a1 = (metrics.ranking(s1[m], y_batch[m])["auc"]
                  if s1 is not None else float("nan"))
            s2 = scores_of(name, factories[name], train, shuf_glob)
            a2 = (metrics.ranking(s2[m], y_glob[m])["auc"]
                  if s2 is not None else float("nan"))

            alert = ""
            if name == "oracle":
                if auc < 0.999:
                    alert = "  <== HARNESS BROKEN"
                    problems += 1
            elif abs(a2 - 0.5) > 0.05:
                alert = "  <== LEAK"
                problems += 1
            # how much of the gain over chance comes from telling batches apart
            from_batch = max(a1 - 0.5, 0.0) / max(auc - 0.5, 1e-9) if auc > 0.5 else float("nan")
            print(f"{name:>18} {auc:>7.3f} [{lo:>6.3f}, {hi:>6.3f}] "
                  f"{a1:>12.3f} {a2:>13.3f} {100 * from_batch:>10.0f}%{alert}")

    print(f"\n{'-' * 78}")
    if problems == 0:
        print("Controls OK: the oracle reaches 1.000, and no filter survives the")
        print("GLOBAL label shuffle. There is no leak.")
    else:
        print(f"{problems} control(s) FAILED; see the alerts above.")
    print()
    print("'batch only' is the fraction of the gain over chance that survives")
    print("shuffling labels WITHIN each batch: a filter reacting to batches that")
    print("happen to hold more fraud, not to the recommendation itself.")
    print("Filters whose intervals overlap are NOT separated by the evidence.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(tuple(sys.argv[1:]) or ("otc", "alpha")))
