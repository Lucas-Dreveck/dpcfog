"""AUC interval resampling raters, not recommendations.

    python -m analysis.ci_by_rater
    python -m analysis.ci_by_rater --networks otc alpha

In layer 1 the label belongs to the user: every recommendation from a fraudulent
rater is positive. A bootstrap that draws recommendations as if independent
leaves the between-rater variation out and gives intervals that are too narrow.
Here both are computed side by side, same seed and same 1,000 resamples: the
per-recommendation bootstrap of `validation.controls` and the per-rater cluster
bootstrap of `paper_tables`. The ratio of the widths measures the problem; in
the paper's data it was 4 to 10, and DPC's interval contains 0.5 on the three
networks once raters are resampled.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

import paper_tables as pt
from dpcfog.experiment import filter_catalog
from realdata.batches import build_batches, split_by_time
from realdata.networks import load
from validation.controls import bootstrap as bootstrap_per_recommendation

OUTPUT = Path(__file__).resolve().parent.parent / "outputs" / "ci_by_rater.json"
FILTERS = ("dpc", "mlp", "reputation", "deeptrust")


def measure(network: str) -> dict:
    load_args = dict(rounds_per_year=365) if network == "epinions" else {}
    n_fog = 8 if network == "epinions" else 1
    net = load(network, **load_args)
    train, evaluation = split_by_time(build_batches(net, window=5, n_fog=n_fog))
    m = np.concatenate([b.evaluable for b in evaluation])
    y = np.concatenate([b.label for b in evaluation])[m]
    rater = np.concatenate([b.rater for b in evaluation])[m]
    fac = filter_catalog(0)
    out = {"n_recommendations": int(y.size),
           "n_raters": int(np.unique(rater).size),
           "n_fraudulent_raters": int(np.unique(rater[y]).size)}
    for name in FILTERS:
        f = fac[name]()
        if f.supervised:
            f.fit(train)
        sc = np.concatenate([f.score(b) for b in evaluation])[m]
        auc = float(roc_auc_score(y, sc))
        lo_r, hi_r = bootstrap_per_recommendation(sc, y)
        lo, hi = pt.bootstrap(sc, y, rater)
        out[name] = {"auc": auc, "ci_recommendation": [lo_r, hi_r],
                     "ci_rater": [lo, hi]}
        print(f"{network:9s} {name:10s} AUC {auc:.3f}  "
              f"per recommendation [{lo_r:.3f}; {hi_r:.3f}]  "
              f"per rater [{lo:.3f}; {hi:.3f}]", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--networks", nargs="+", default=["otc", "alpha", "epinions"])
    a = ap.parse_args()
    OUTPUT.parent.mkdir(exist_ok=True)
    everything = json.loads(OUTPUT.read_text(encoding="utf-8")) if OUTPUT.exists() else {}
    for network in a.networks:
        everything[network] = measure(network)
        OUTPUT.write_text(json.dumps(everything, indent=1), encoding="utf-8")
        print(network, {k: v for k, v in everything[network].items() if k.startswith("n_")})


if __name__ == "__main__":
    main()
