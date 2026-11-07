"""Real trust networks with published fraud labels.

Data: SNAP (Kumar et al., 2016). Labels: the protocol of Kumar et al. (2018),
REV2, WSDM, distributed with the paper.

    Bitcoin OTC / Alpha  benign users are the platform founder and those the
                         founder rated >= +0.5; fraudulent users are those
                         these trusted raters rate uniformly <= -0.5.
    Epinions             benign if the total trust in the separate trust
                         network is >= +10; fraudulent if <= -10.

The protocol is not reimplemented: the label files published by the authors are
used as they are.

Two properties of these data change how any result reads, so they are exposed
in the structure:

1. **Most users have no label.** On Bitcoin OTC, 316 of the 4,814 raters are
   labelled, and their ratings are 23.5% of the total. The others are not
   benign, they are unknown. Hence the `evaluable` vector, which is all the
   metric looks at. The filter sees everything, as it would in operation.

2. **The median is 2 ratings per user**, against a fixed K = 10 in the
   simulation. Four of the five features are rater statistics, and with two
   observations they barely exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from .paths import require, rev2_path

Network = Literal["otc", "alpha", "epinions", "amazon"]

# Rating scale of each network, to map v onto [0,1].
SCALES: dict[str, tuple[float, float]] = {
    "otc": (-10.0, 10.0),
    "alpha": (-10.0, 10.0),
    # Epinions is nominally 1 to 5, but the file holds 96,287 ratings of 6
    # (0.7% of the total). The clip treats them as the maximum rating.
    "epinions": (1.0, 5.0),
    "amazon": (1.0, 5.0),
}

# Expected values, from Table 2 of Kumar et al. (2018). Ingestion fails if they
# do not match.
EXPECTED: dict[str, dict[str, int]] = {
    "otc": {"edges": 35592, "users": 4814, "targets": 5858, "fraud": 180, "benign": 136},
    "alpha": {"edges": 24186, "users": 3286, "targets": 3754, "fraud": 102, "benign": 138},
    "epinions": {"edges": 4835208, "users": 120486, "targets": 132585},
}

NETWORKS: tuple[str, ...] = ("otc", "alpha", "epinions", "amazon")


@dataclass
class RealNetwork:
    """A real rating network, in the same format the simulation produces."""

    name: str
    rater: np.ndarray        # i, re-indexed from 0
    target: np.ndarray       # j, re-indexed from 0
    value: np.ndarray        # v in [0,1]
    time: np.ndarray         # seconds since the epoch
    round_idx: np.ndarray    # time window, see `group_into_rounds`
    label: np.ndarray        # True = malicious (rater labelled fraudulent)
    evaluable: np.ndarray    # True = the rater has a label; only these count
    n_raters: int
    n_targets: int
    raw_rating: np.ndarray   # the original rating, before normalisation
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.value.size)

    @property
    def malicious_rate(self) -> float:
        """Among labelled ratings. A rate over the total would be meaningless."""
        if not self.evaluable.any():
            return 0.0
        return float(self.label[self.evaluable].mean())

    def summary(self) -> str:
        return (
            f"{self.name}: {len(self):,} ratings, {self.n_raters:,} raters, "
            f"{self.n_targets:,} targets, {int(self.evaluable.sum()):,} evaluable "
            f"({100 * self.evaluable.mean():.1f}%), "
            f"{100 * self.malicious_rate:.1f}% malicious among evaluable, "
            f"{self.round_idx.max() + 1} rounds"
        )


def _time_to_seconds(column: pd.Series) -> np.ndarray:
    """OTC and Alpha carry epoch seconds; Epinions carries YYYY/MM/DD.

    The conversion does not go through `.astype("int64") / 1e9`: the internal
    resolution of `datetime64` changed across pandas versions, and the result
    came out in microseconds, putting all of Epinions in a single round without
    any error. `total_seconds` does not depend on the internal resolution.
    """
    if pd.api.types.is_numeric_dtype(column):
        return column.to_numpy(dtype=float)
    dates = pd.to_datetime(column, format="%Y/%m/%d", errors="coerce")
    epoch = pd.Timestamp("1970-01-01")
    return (dates - epoch).dt.total_seconds().to_numpy(dtype=float)


def group_into_rounds(time: np.ndarray, rounds_per_year: int = 12) -> np.ndarray:
    """Convert continuous time into the discrete rounds the simulation assumes.

    A calendar window, not an equal-count one: the Fog node processes what
    arrives in a time interval and does not wait to fill a batch. Monthly by
    default, which turns the five years of Bitcoin OTC into about 62 rounds of a
    few hundred ratings each.
    """
    if time.size == 0:
        return np.zeros(0, dtype=np.int32)
    window = 365.25 * 24 * 3600 / rounds_per_year
    return ((time - time.min()) // window).astype(np.int32)


def load(
    network: Network = "otc",
    rounds_per_year: int = 12,
    validate: bool = True,
) -> RealNetwork:
    folder = require(rev2_path(network))
    edges = pd.read_csv(
        require(folder / f"{network}_network.csv"),
        header=None,
        names=["source", "target", "rating", "time"],
    )
    # The Epinions file distributed with REV2 repeats edges: 13,668,104 rows for
    # 4,835,208 distinct pairs, 6,060,876 of them identical in all four
    # columns. Without deduplication each rating would weigh about 2.8 in the
    # consensus. The number of distinct pairs matches Table 2 exactly.
    before = len(edges)
    edges = edges.sort_values("time").drop_duplicates(
        subset=["source", "target"], keep="last"
    )
    duplicates = before - len(edges)
    gt = pd.read_csv(
        require(folder / f"{network}_gt.csv"), header=None, names=["node", "label"]
    )

    if validate and network in EXPECTED:
        e = EXPECTED[network]
        measured = {
            "edges": len(edges),
            "users": edges.source.nunique(),
            "targets": edges.target.nunique(),
            "fraud": int((gt.label == -1).sum()),
            "benign": int((gt.label == 1).sum()),
        }
        mismatched = {
            k: (measured[k], v) for k, v in e.items() if measured.get(k) != v
        }
        if mismatched:
            raise ValueError(
                f"{network} does not match Table 2 of Kumar et al. (2018): "
                f"{mismatched} (measured, expected)"
            )

    # re-indexing: original ids are sparse and the filters use dense indices
    rater_code = {v: k for k, v in enumerate(sorted(edges.source.unique()))}
    target_code = {v: k for k, v in enumerate(sorted(edges.target.unique()))}
    rater = edges.source.map(rater_code).to_numpy(dtype=np.int32)
    target = edges.target.map(target_code).to_numpy(dtype=np.int32)

    lo, hi = SCALES[network]
    rating = edges.rating.to_numpy(dtype=float)
    value = np.clip((rating - lo) / (hi - lo), 0.0, 1.0)

    time = _time_to_seconds(edges.time)
    order = np.argsort(time, kind="stable")

    mapping = dict(zip(gt.node, gt.label))
    raw_label = edges.source.map(mapping)
    evaluable = raw_label.notna().to_numpy()
    label = (raw_label == -1).to_numpy()

    rater, target, value, time, rating = (
        rater[order], target[order], value[order], time[order], rating[order]
    )
    evaluable, label = evaluable[order], label[order]

    return RealNetwork(
        name=network,
        rater=rater,
        target=target,
        value=value,
        time=time,
        round_idx=group_into_rounds(time, rounds_per_year),
        label=label,
        evaluable=evaluable,
        n_raters=len(rater_code),
        n_targets=len(target_code),
        raw_rating=rating,
        meta={
            "scale": (lo, hi),
            "rounds_per_year": rounds_per_year,
            # The two index spaces are independent: rater 5 and target 5 are not
            # the same user. Keeping the maps avoids crossing a rater label with
            # a target statistic (see realdata/calibration.py).
            "rater_code": rater_code,
            "target_code": target_code,
            "n_labelled": int(len(gt)),
            "duplicate_rows_dropped": int(duplicates),
            "data_source": "SNAP / Kumar et al. (2016)",
            "label_source": "REV2, Kumar et al. (2018), WSDM",
        },
    )
