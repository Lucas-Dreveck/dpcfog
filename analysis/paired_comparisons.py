"""Paired comparisons with intervals on layer 1, from the prediction cache.

    python -m analysis.layer1_predictions      # once, beforehand
    python -m analysis.paired_comparisons      # the three networks
    python -m analysis.paired_comparisons otc

Five measurements, each stated as a hypothesis with the test that refutes it:

  new_vs_seen      AUC on recommendations from raters absent from training
                   versus raters seen in training. Hypothesis: trained models
                   win by recognising the rater. Refuted if the neural network
                   minus DPC gap does not shrink among new raters.
  f1_difference    DPC F1 minus neural-network F1, resampling raters. The lower
                   bound of the 90% interval of the difference is the one-sided
                   5% non-inferiority test against the -0.05 margin.
  mae_difference   Trust Score error with minus without the filter, resampling
                   targets.
  independent_ref  H2 reference independent of the filter: a random half of the
                   benign users defines the reference, and the Trust Score uses
                   only the other users' ratings, 20 draws. Hypothesis: part of
                   DPC's error increase comes from the original reference reusing
                   ratings the filter removes. Refuted if DPC keeps increasing
                   the error in most draws.
  deeptrust_epochs Deep Trust AUC with 10, 20, 40 and 80 epochs.

Resampling unit: the rater for everything measured on labelled
recommendations, because the label belongs to the user; the target for the
Trust Score error, because the error is a mean over targets.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from dpcfog import metrics, trust
from realdata.batches import reference_trust_score
from realdata.networks import load

from .layer1_predictions import EXTRA_EPOCHS, LOAD, load_predictions

ROOT = Path(__file__).resolve().parent.parent
N_BOOTSTRAP = 1000
SEED = 20260827
REFERENCE_DRAWS = 20
TRAINED = ("mlp", "deeptrust", "tree3")


def _groups(key: np.ndarray) -> list[np.ndarray]:
    _, inverse = np.unique(key, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    bounds = np.searchsorted(inverse[order], np.arange(inverse.max() + 2))
    return [order[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]


def _resamples(key: np.ndarray, n: int = N_BOOTSTRAP):
    groups = _groups(key)
    rng = np.random.default_rng(SEED)
    for _ in range(n):
        yield np.concatenate([groups[g] for g in
                              rng.integers(0, len(groups), size=len(groups))])


def _ci(values, level=95):
    v = np.asarray([x for x in values if np.isfinite(x)])
    a = (100 - level) / 2
    return float(np.percentile(v, a)), float(np.percentile(v, 100 - a))


def _auc(sc, y):
    if y.all() or not y.any():
        return float("nan")
    return float(roc_auc_score(y, sc))


def _f1(dec, y):
    return metrics.evaluate(dec, y).f1


def _clean(sc):
    finite = np.isfinite(sc)
    return sc if finite.all() else np.where(finite, sc, np.nanmin(sc[finite]))


def f1_difference(p) -> dict:
    m = p["evaluable"]
    y, r = p["label"][m], p["rater"][m]
    a, b = p["dec_dpc"][m], p["dec_mlp"][m]
    point = _f1(a, y) - _f1(b, y)
    difs = [(_f1(a[i], y[i]) - _f1(b[i], y[i])) for i in _resamples(r)]
    lo95, hi95 = _ci(difs, 95)
    lo90, hi90 = _ci(difs, 90)
    return {"f1_dpc": _f1(a, y), "f1_mlp": _f1(b, y), "dif": point,
            "ci95": [lo95, hi95], "ci90": [lo90, hi90],
            "non_inferior_at_5pp": bool(lo90 > -0.05)}


def mae_difference(p, net, flt="dpc") -> dict:
    """Same computation as `realdata.real_experiment`, with a per-target interval."""
    t_ref, has_ref = reference_trust_score(net)
    target, value, n = p["target"], p["value"], net.n_targets
    t_unf, no0 = trust.trust_score(target, value, np.ones(target.size, bool), n)
    t_f, no1 = trust.trust_score(target, value, ~p[f"dec_{flt}"], n)
    comp = np.flatnonzero(has_ref & ~no0 & ~no1)
    e_unf = np.abs(t_unf[comp] - t_ref[comp])
    e_f = np.abs(t_f[comp] - t_ref[comp])
    d = e_f - e_unf
    rng = np.random.default_rng(SEED)
    difs = [d[rng.integers(0, d.size, d.size)].mean() for _ in range(N_BOOTSTRAP)]
    return {"mae_unfiltered": float(e_unf.mean()), "mae_filtered": float(e_f.mean()),
            "dif": float(d.mean()), "ci95": list(_ci(difs, 95)),
            "targets": int(comp.size)}


def independent_reference(p, net, flt="dpc") -> dict:
    """Reference from half of the benign users; Trust Score from the others only."""
    benign = np.unique(net.rater[net.evaluable & ~net.label])
    target, value, rater, n = p["target"], p["value"], p["rater"], net.n_targets
    dec = p[f"dec_{flt}"]
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(REFERENCE_DRAWS):
        a = rng.choice(benign, size=benign.size // 2, replace=False)
        from_a = np.isin(net.rater, a) & net.evaluable & ~net.label
        total = np.bincount(net.target[from_a], weights=net.value[from_a], minlength=n)
        count = np.bincount(net.target[from_a], minlength=n)
        has = count > 0
        ref = np.where(has, total / np.maximum(count, 1), np.nan)
        outside = ~np.isin(rater, a)
        t_unf, s0 = trust.trust_score(target, value, outside, n)
        t_f, s1 = trust.trust_score(target, value, outside & ~dec, n)
        comp = has & ~s0 & ~s1
        e_unf = np.abs(t_unf[comp] - ref[comp]).mean()
        e_f = np.abs(t_f[comp] - ref[comp]).mean()
        draws.append({"mae_unfiltered": float(e_unf), "mae_filtered": float(e_f),
                      "dif": float(e_f - e_unf), "targets": int(comp.sum())})
    difs = np.array([s["dif"] for s in draws])
    return {"mean_dif": float(difs.mean()), "min_dif": float(difs.min()),
            "max_dif": float(difs.max()),
            "draws_with_increase": int((difs > 0).sum()),
            "draws": REFERENCE_DRAWS,
            "mean_mae_unfiltered": float(np.mean([s["mae_unfiltered"] for s in draws])),
            "mean_mae_filtered": float(np.mean([s["mae_filtered"] for s in draws])),
            "mean_targets": float(np.mean([s["targets"] for s in draws]))}


def new_vs_seen(p) -> dict:
    m = p["evaluable"]
    y, r = p["label"][m], p["rater"][m]
    seen = np.isin(r, p["train_raters"])
    out = {}
    for sub_name, sub in (("new", ~seen), ("seen", seen)):
        ys, rs = y[sub], r[sub]
        row = {"recommendations": int(sub.sum()),
               "fraudulent": int(ys.sum()),
               "raters": int(np.unique(rs).size),
               "fraudulent_raters": int(np.unique(rs[ys]).size)}
        for name in ("dpc", "dpc_han", "reputation", "consensus") + TRAINED:
            sc = _clean(p[f"score_{name}"][m][sub])
            row[f"auc_{name}"] = _auc(sc, ys)
        a = _clean(p["score_mlp"][m][sub])
        b = _clean(p["score_dpc"][m][sub])
        difs = [_auc(a[i], ys[i]) - _auc(b[i], ys[i]) for i in _resamples(rs)]
        row["mlp_minus_dpc"] = row["auc_mlp"] - row["auc_dpc"]
        row["mlp_minus_dpc_ci95"] = list(_ci(difs, 95))
        out[sub_name] = row
    return out


def deeptrust_epochs(p) -> dict:
    m = p["evaluable"]
    y, r = p["label"][m], p["rater"][m]
    out = {}
    for ep in sorted((40,) + EXTRA_EPOCHS):
        key = "score_deeptrust" if ep == 40 else f"score_deeptrust_ep{ep}"
        sc = _clean(p[key][m])
        vals = [_auc(sc[i], y[i]) for i in _resamples(r)]
        out[str(ep)] = {"auc": _auc(sc, y), "ci95": list(_ci(vals, 95))}
    return out


def measure(network: str) -> dict:
    load_args, _ = LOAD[network]
    net = load(network, **load_args)
    p = load_predictions(network)
    return {"f1_difference": f1_difference(p),
            "mae_difference_dpc": mae_difference(p, net),
            "independent_reference_dpc": independent_reference(p, net),
            "new_vs_seen": new_vs_seen(p),
            "deeptrust_epochs": deeptrust_epochs(p)}


def path_for(network: str) -> Path:
    return ROOT / "outputs" / f"paired_comparisons_{network}.json"


def read(network: str) -> dict:
    return json.loads(path_for(network).read_text(encoding="utf-8"))


if __name__ == "__main__":
    for n in sys.argv[1:] or list(LOAD):
        r = measure(n)
        path_for(n).parent.mkdir(exist_ok=True)
        path_for(n).write_text(json.dumps(r, indent=1), encoding="utf-8")
        print(n, json.dumps(r, indent=1), flush=True)
