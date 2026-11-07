"""Scores and decisions of every filter on the layer-1 evaluation, saved to disk.

    python -m analysis.layer1_predictions            # the three networks
    python -m analysis.layer1_predictions otc alpha

`analysis.paired_comparisons` needs the per-recommendation predictions of every
filter. Running the filters once and saving them avoids re-training in each
analysis and guarantees every analysis reads the same result.

The arrangement copies `realdata.real_experiment.run_network`: the decision
(`flag`) comes from the trained instance, and the score (`score`) from a fresh
one, because stateful filters must not see the same batches twice. `check`
authorises the cache: the F1 recomputed from it must match `results/real.csv`,
and the AUC must match `outputs/layer1_measurements.json` (from paper_tables.py).

Deep Trust is also run with 10, 20 and 80 epochs, to measure whether the 40
epochs of the experiment matter.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from dpcfog import metrics
from dpcfog.experiment import filter_catalog
from dpcfog.filters.deeptrust import DeepTrustFilter
from realdata.batches import build_batches, split_by_time
from realdata.networks import load

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs"
FILTERS = ["rev2", "reputation", "tree3", "mlp", "deeptrust", "consensus",
           "dpc_han", "dpc"]
EXTRA_EPOCHS = (10, 20, 80)
LOAD = {"otc": ({}, 1), "alpha": ({}, 1),
        "epinions": (dict(rounds_per_year=365), 8)}


def path_for(network: str) -> Path:
    return OUTPUT / f"layer1_predictions_{network}.npz"


def _run(factory, train, evaluation):
    f = factory()
    if f.supervised:
        f.fit(train)
    decision = np.concatenate([f.flag(b) for b in evaluation]).astype(bool)
    g = factory()
    if g.supervised:
        g.fit(train)
    parts = [g.score(b) for b in evaluation]
    score = (np.concatenate(parts).astype(float)
             if all(p is not None for p in parts) else None)
    return decision, score


def generate(network: str) -> Path:
    load_args, n_fog = LOAD[network]
    net = load(network, **load_args)
    train, evaluation = split_by_time(build_batches(net, window=5, n_fog=n_fog))
    data = {
        "rater": np.concatenate([b.rater for b in evaluation]),
        "target": np.concatenate([b.target for b in evaluation]),
        "value": np.concatenate([b.value for b in evaluation]),
        "label": np.concatenate([b.label for b in evaluation]),
        "evaluable": np.concatenate([b.evaluable for b in evaluation]),
        "train_raters": np.unique(np.concatenate(
            [b.rater[b.evaluable] for b in train])),
    }
    fac = dict(filter_catalog(0))
    for name in FILTERS:
        decision, score = _run(fac[name], train, evaluation)
        data[f"dec_{name}"] = decision
        if score is not None:
            data[f"score_{name}"] = score
        print(f"  {network} {name}", flush=True)
    for ep in EXTRA_EPOCHS:
        decision, score = _run(
            lambda e=ep: DeepTrustFilter(seed=0, epochs=e), train, evaluation)
        data[f"dec_deeptrust_ep{ep}"] = decision
        data[f"score_deeptrust_ep{ep}"] = score
        print(f"  {network} deeptrust {ep} epochs", flush=True)
    OUTPUT.mkdir(exist_ok=True)
    destination = path_for(network)
    np.savez_compressed(destination, **data)
    return destination


def load_predictions(network: str) -> dict[str, np.ndarray]:
    with np.load(path_for(network)) as z:
        return {k: z[k] for k in z.files}


def check(network: str) -> dict[str, float]:
    """Largest deviation between the cache and the published numbers; must be ~0."""
    import pandas as pd

    p = load_predictions(network)
    m = p["evaluable"]
    y = p["label"][m]
    real = pd.read_csv(ROOT / "results" / "real.csv")
    real = real[real.scenario == f"real_{network}"].set_index("filter")
    ci = json.loads((OUTPUT / "layer1_measurements.json").read_text(
        encoding="utf-8"))["data"][network]
    df1 = dauc = 0.0
    for name in FILTERS:
        f1 = metrics.evaluate(p[f"dec_{name}"][m], y).f1
        df1 = max(df1, abs(f1 - real.loc[name, "f1"]))
        if f"score_{name}" in p and name in ci:
            auc = metrics.ranking(p[f"score_{name}"][m], y)["auc"]
            dauc = max(dauc, abs(auc - ci[name]["auc"]))
    return {"largest_f1_deviation": df1, "largest_auc_deviation": dauc}


if __name__ == "__main__":
    for n in sys.argv[1:] or list(LOAD):
        print(generate(n), flush=True)
        print(n, check(n), flush=True)
