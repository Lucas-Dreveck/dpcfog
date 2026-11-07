"""Reproduce the unsupervised REV2 experiment (Kumar et al., 2018).

    python -m validation.rev2_kumar2018

The paper has two experiments:

  Experiment 1, Section 5.3, Table 3: unsupervised. Metric: Average Precision,
  "calculated for the top and bottom 100 users". This is what our
  implementation corresponds to.

  Experiment 2, Section 5.4, Table 4: supervised, a 2187-dimensional vector of
  fairness scores fed to a random forest. The AUC >= 0.85 comes from there and
  is not the target here.

Published targets (Table 3, REV2 column, Average Precision x 100):

    fraudulent   OTC 96.30   Alpha 75.29
    benign       OTC 92.85   Alpha 84.85

Protocol. The paper does not detail how the top and bottom 100 are chosen,
so this is the reading used:

  1. REV2 runs once over the whole graph (`score_global`), as in the paper.
  2. The user score is fairness; lower = more fraudulent.
  3. Only users with a published label are considered.
  4. "top and bottom 100": the 100 labelled users with the lowest fairness and
     the 100 with the highest, joined. With fewer than 200, all of them.
  5. Fraud average precision uses -fairness as the score; benign uses +fairness.

Tolerance: 10 points of Average Precision, wide on purpose because item 4
reads an ambiguous sentence and because the authors use cold-start and
behavioural components our configuration switches off.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from dpcfog.filters.rev2 import score_global
from realdata.networks import load
from realdata.paths import require, rev2_path

TOLERANCE = 10.0

PUBLISHED = {
    "otc":   {"fraudulent": 96.30, "benign": 92.85},
    "alpha": {"fraudulent": 75.29, "benign": 84.85},
}


def labels_per_user(network: str, net) -> pd.Series:
    """Published label of each rater, on the network's internal index.

    The ground truth is in the original identifier space, and `rater_code`
    keeps the re-indexing order.
    """
    gt = pd.read_csv(require(rev2_path(network) / f"{network}_gt.csv"), header=None,
                     names=["node", "label"])
    mapping = dict(zip(gt.node, gt.label))
    code = net.meta["rater_code"]
    return pd.Series([mapping.get(u, 0) for u in code], index=range(len(code)))


def one_network(network: str) -> dict:
    net = load(network)
    fairness, _, _ = score_global(net)

    # `score_global` re-indexes with np.unique over the raters present.
    present = np.unique(net.rater)
    score = pd.Series(fairness, index=present)

    lab = labels_per_user(network, net)
    labelled = lab[lab != 0].index.intersection(score.index)
    if len(labelled) == 0:
        raise RuntimeError(f"{network}: no labelled user with a score")

    f = score.loc[labelled]
    y_fraud = (lab.loc[labelled] == -1).to_numpy()

    order = f.sort_values().index
    sel = order[:100].union(order[-100:]) if len(order) > 200 else order
    ff = f.loc[sel].to_numpy()
    yy = (lab.loc[sel] == -1).to_numpy()

    return {
        "network": network,
        "labelled": int(len(labelled)),
        "fraudulent": int(y_fraud.sum()),
        "benign": int((~y_fraud).sum()),
        "evaluated": int(len(sel)),
        "ap_fraudulent": 100 * average_precision_score(yy, -ff),
        "ap_benign": 100 * average_precision_score(~yy, ff),
        # for the record only: AP over all labelled users; not the paper's metric
        "ap_all": 100 * average_precision_score(y_fraud, -f.to_numpy()),
    }


# --------------------------------------------------------------------------
# Test of the explanation.
# --------------------------------------------------------------------------
# Table 3 reports the full REV2: network (N) + cold start (C) + behaviour (B).
# Our implementation has only the network part. Figure 6(a) of the paper
# measures the ablation on Bitcoin Alpha, in unsupervised average precision:
#
#     N = 64.0    N+B = 71.9    N+C = 74.9    N+C+B = 76.5
#
# Prediction: if the gap comes from the missing
# cold-start term, switching it on must increase Alpha's average precision by
# about the N -> N+C jump of the paper, around 11 points. Our parametrisation
# has a single prior term (alpha1/beta1) where the paper separates
# alpha1/beta1 from alpha2/beta2, so only the direction is comparable.

ALPHA_ABLATION = {"N": 64.0, "N+B": 71.9, "N+C": 74.9, "N+C+B": 76.5}
PREDICTED_JUMP = 11.0


def test_explanation() -> bool:
    print("\n" + "=" * 78)
    print("Test of the explanation: does the cold-start term explain the gap?")
    print(f"prediction: switching the term on raises "
          f"Alpha's AP\nby about {PREDICTED_JUMP:.0f} points (N -> N+C jump of "
          f"Figure 6a)\n")

    net = load("alpha")
    lab = labels_per_user("alpha", net)
    present = np.unique(net.rater)

    print(f"{'configuration':24} {'fraud AP':>10} {'published':>10}")
    results = {}
    for name, (a, b), target in (("network only (N)", (0.0, 0.0), ALPHA_ABLATION["N"]),
                                 ("network + prior (N+C)", (1.0, 1.0),
                                  ALPHA_ABLATION["N+C"])):
        fairness, _, _ = score_global(net, alpha1=a, beta1=b)
        score = pd.Series(fairness, index=present)
        labelled = lab[lab != 0].index.intersection(score.index)
        f = score.loc[labelled]
        order = f.sort_values().index
        sel = order[:100].union(order[-100:]) if len(order) > 200 else order
        ap = 100 * average_precision_score((lab.loc[sel] == -1).to_numpy(),
                                           -f.loc[sel].to_numpy())
        results[name] = ap
        print(f"{name:24} {ap:>10.2f} {target:>10.1f}")

    jump = results["network + prior (N+C)"] - results["network only (N)"]
    print(f"\nmeasured jump: {jump:+.2f} points   (predicted: "
          f"+{PREDICTED_JUMP:.0f})")
    if jump > PREDICTED_JUMP / 2:
        print("The explanation SURVIVES: the missing term moves the result in the")
        print("predicted direction and order of magnitude.")
        return True
    print("The explanation does NOT survive. The gap to the published value does")
    print("not come from the cold-start term and remains unexplained.")
    return False


def main(networks=("otc", "alpha")) -> int:
    print("Unsupervised REV2 against Table 3 of Kumar et al. (2018)")
    print(f"tolerance: {TOLERANCE:.0f} AP points\n")
    print(f"{'network':8} {'labelled':>9} {'eval.':>6} "
          f"{'fraud AP':>10} {'published':>10} {'error':>7}   "
          f"{'benign AP':>10} {'published':>10} {'error':>7}")

    failures = []
    for network in networks:
        r = one_network(network)
        p = PUBLISHED[network]
        e_f = r["ap_fraudulent"] - p["fraudulent"]
        e_b = r["ap_benign"] - p["benign"]
        mark = ""
        if abs(e_f) > TOLERANCE or abs(e_b) > TOLERANCE:
            mark = "  <== OUTSIDE"
            failures.append(network)
        print(f"{network:8} {r['labelled']:>9} {r['evaluated']:>6} "
              f"{r['ap_fraudulent']:>10.2f} {p['fraudulent']:>10.2f} "
              f"{e_f:>+7.2f}   {r['ap_benign']:>10.2f} "
              f"{p['benign']:>10.2f} {e_b:>+7.2f}{mark}")

    print("\n" + "-" * 78)
    if failures:
        print(f"NOT reproduced on: {', '.join(failures)}.")
        print("Every `rev2` number stays marked as not validated, and no")
        print("conclusion of the paper rests on it.")
        test_explanation()
        return 1
    test_explanation()
    print("Reproduces Table 3 within the declared tolerance. This validates the")
    print("UNSUPERVISED whole-graph implementation, not the per-batch Fog")
    print("variant nor the supervised experiment.")
    return 0


if __name__ == "__main__":
    sys.exit(main(tuple(sys.argv[1:]) or ("otc", "alpha")))
