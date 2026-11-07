"""Generate the paper's result tables from the CSVs and the raw networks.

    python paper_tables.py            # uses outputs/layer1_measurements.json if present
    python paper_tables.py --redo     # re-measures layer 1 (Epinions takes a while)

No number in the paper is typed by hand. Three tables, written to `outputs/`
as LaTeX (English labels; the paper itself is in Portuguese):

  table_layer1  detection on the three real networks, with confidence intervals.
                Real trust networks, not Fog-IoT.
  table_mu      AUC against the fraction of malicious nodes, in the simulation.
                It supports claims about mechanism only.
  table_cost    time per batch and memory against batch size.

The intervals are a 1,000-resample bootstrap by rater: a user is drawn with all
of their recommendations. The layer-1 label belongs to the user, and the
recommendations of one rater are not independent; resampling recommendations
made the intervals 4 to 10 times narrower than the sample of users allows.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from dpcfog import metrics
from dpcfog.experiment import filter_catalog
from realdata.batches import build_batches, split_by_time
from realdata.networks import load

OUTPUT = Path("outputs")
N_BOOTSTRAP = 1000
SEED = 20260827

# Filters in the paper, in the order they are discussed. DPC variants that give
# the same number as `dpc` become a sentence in the text, not a table row.
ORDER = ["rev2", "reputation", "tree3", "mlp", "deeptrust", "consensus",
         "dpc_han", "dpc"]

LABELS = {
    "rev2": r"REV2\textsuperscript{*}",
    "reputation": "Rater history",
    "tree3": "Depth-3 tree",
    "mlp": "Neural network",
    "deeptrust": "Deep Trust",
    "consensus": "Consensus",
    "dpc_han": "Adaptation of Han et al.",
    "dpc": "DPC (this work)",
}


def num(v, places=3):
    """Negative numbers in math mode, so the sign is a minus and not a hyphen."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "---"
    if v < 0:
        return f"${v:.{places}f}$"
    return f"{v:.{places}f}"


def bootstrap(scores, y, rater, n=N_BOOTSTRAP):
    """95% CI of the AUC by percentiles, drawing raters with all their
    recommendations."""
    from sklearn.metrics import roc_auc_score

    finite = np.isfinite(scores)
    if not finite.all():
        scores = np.where(finite, scores, np.nanmin(scores[finite]))
    _, inverse = np.unique(rater, return_inverse=True)
    n_groups = int(inverse.max()) + 1
    order = np.argsort(inverse, kind="stable")
    bounds = np.searchsorted(inverse[order], np.arange(n_groups + 1))
    groups = [order[bounds[i]:bounds[i + 1]] for i in range(n_groups)]
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(n):
        idx = np.concatenate([groups[g] for g in rng.integers(0, n_groups, size=n_groups)])
        yy = y[idx]
        if yy.all() or not yy.any():
            continue
        vals.append(roc_auc_score(yy, scores[idx]))
    if len(vals) < n // 2:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _layer1_batches(network: str):
    load_args = dict(rounds_per_year=365) if network == "epinions" else {}
    n_fog = 8 if network == "epinions" else 1
    net = load(network, **load_args)
    return split_by_time(build_batches(net, window=5, n_fog=n_fog))


def measure_network(network: str, with_interval: bool) -> dict[str, dict]:
    train, evaluation = _layer1_batches(network)
    m = np.concatenate([b.evaluable for b in evaluation])
    y = np.concatenate([b.label for b in evaluation])
    rater = np.concatenate([b.rater for b in evaluation])[m]
    fac = dict(filter_catalog(0))

    out = {}
    for name in ORDER:
        f = fac[name]()
        if f.supervised:
            f.fit(train)
        try:
            parts = [f.score(b) for b in evaluation]
        except Exception:
            continue
        if any(p is None for p in parts):
            continue
        sc = np.concatenate(parts)[m]
        yy = y[m]
        d = metrics.ranking(sc, yy)
        lo, hi = (bootstrap(sc, yy, rater) if with_interval
                  else (float("nan"),) * 2)
        out[name] = {"auc": d["auc"], "ap": d["average_precision"],
                     "lo": lo, "hi": hi}
    return out


def table_layer1(data: dict[str, dict], rates: dict[str, float]) -> str:
    # Sorted by Bitcoin OTC AUC, descending.
    order = sorted(ORDER,
                   key=lambda n: -data["otc"].get(n, {}).get("auc", -1))
    rows = []
    for name in order:
        c = [LABELS[name]]
        for network in ("otc", "alpha", "epinions"):
            d = data[network].get(name)
            if d is None:
                c += ["---", "---"]
                continue
            c.append(num(d["auc"]))
            c.append("---" if np.isnan(d["lo"])
                     else f"[{num(d['lo'])}; {num(d['hi'])}]")
        rows.append(" & ".join(c) + r" \\")

    head = (r"Filter & \multicolumn{2}{c}{Bitcoin OTC} & "
            r"\multicolumn{2}{c}{Bitcoin Alpha} & \multicolumn{2}{c}{Epinions} \\"
            "\n" + r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}" + "\n"
            + r" & AUC & 95\% CI & AUC & 95\% CI & AUC & 95\% CI \\")

    return (
        "\\begin{table}[!ht]\n\\centering\n"
        "\\caption{Detection, on the three real networks, of the recommendations "
        "issued by users labelled fraudulent, who are "
        f"{num(100*rates['otc'], 1)}\\% of the evaluated ones on Bitcoin OTC, "
        f"{num(100*rates['alpha'], 1)}\\% on Bitcoin Alpha and "
        f"{num(100*rates['epinions'], 1)}\\% on Epinions. The asterisk marks a "
        "reimplementation that was not validated.}\n"
        "\\label{tab:layer1}\n\\small\n\\setlength{\\tabcolsep}{3pt}\n"
        "\\begin{tabular}{lcccccc}\n\\toprule\n"
        + head + "\n\\midrule\n" + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
        "\\end{table}\n"
    )


def table_mu(csv: Path) -> str:
    d = pd.read_csv(csv)
    d = d[d.scenario == "calibrated"]
    t = d.pivot_table(index="filter", columns="mu", values="auc")
    # sorted by the change, from the largest loss to the smallest
    present = [n for n in ORDER if n in t.index]
    present.sort(key=lambda n: t.loc[n][0.4] - t.loc[n][0.1])
    rows = []
    for name in present:
        r = t.loc[name]
        change = r[0.4] - r[0.1]
        rows.append(" & ".join(
            [LABELS[name]] + [num(r[m]) for m in (0.1, 0.2, 0.3, 0.4)]
            + [num(change)]) + r" \\")
    n_runs = int(d.groupby("filter").size().iloc[0])
    return (
        "\\begin{table}[!ht]\n\\centering\n"
        "\\caption{Area under the curve against the fraction of malicious nodes, "
        f"in the calibrated simulation, mean of {n_runs // 4} runs per cell. "
        "It supports claims about mechanism, not about operation.}\n"
        "\\label{tab:mu}\n\\small\n"
        "\\begin{tabular}{lccccc}\n\\toprule\n"
        r"Filter & $\mu=0.10$ & $0.20$ & $0.30$ & $0.40$ & change \\"
        "\n\\midrule\n" + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def table_cost() -> str:
    import time
    import tracemalloc

    from dpcfog.filters.dpc import DPCFilter
    from scalability import BUDGET_MB, fake_batch

    rows = []
    for n in (1000, 2000, 4000, 8000):
        batch = fake_batch(n)
        DPCFilter().flag(batch)
        tracemalloc.start()
        DPCFilter().flag(batch)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        mb = peak / 1024 ** 2
        t0 = time.perf_counter()
        DPCFilter().flag(batch)
        ms = 1000 * (time.perf_counter() - t0)
        fits = "yes" if mb <= BUDGET_MB else r"\textbf{no}"
        rows.append(f"{n:,} & {num(mb, 1)} & {num(ms, 0)} & {fits}" + r" \\")
    return (
        "\\begin{table}[!ht]\n\\centering\n"
        "\\caption{Filter cost against batch size. The "
        f"{BUDGET_MB}~MB budget is a quarter of the memory of the Raspberry Pi~3 "
        "Ullah et al. use as an aggregation node.}\n"
        "\\label{tab:cost}\n\\small\n"
        "\\begin{tabular}{rrrc}\n\\toprule\n"
        "Recommendations per batch & Memory (MB) & Time (ms) & Within budget"
        r" \\" "\n\\midrule\n" + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def main(argv: list[str]) -> int:
    OUTPUT.mkdir(exist_ok=True)
    grid = Path("results/full_grid.csv")
    if not grid.exists():
        print(f"{grid} does not exist. Run the simulation grid first.")
        return 1

    cache = OUTPUT / "layer1_measurements.json"
    use_cache = "--redo" not in argv and cache.exists()
    if use_cache:
        stored = json.loads(cache.read_text(encoding="utf-8"))
        print(f"using {cache} (run with --redo to re-measure)")
    else:
        stored = None

    data, rates = {}, {}
    for network in ("otc", "alpha", "epinions"):
        if stored:
            data[network] = stored["data"][network]
            continue
        print(f"measuring {network}...", flush=True)
        data[network] = measure_network(network, with_interval=True)
    for network in ("otc", "alpha", "epinions"):
        if stored:
            rates[network] = stored["rates"][network]
            continue
        _, evaluation = _layer1_batches(network)
        m = np.concatenate([b.evaluable for b in evaluation])
        y = np.concatenate([b.label for b in evaluation])
        rates[network] = float(y[m].mean())

    if not stored:
        cache.write_text(json.dumps({"data": data, "rates": rates}, indent=1),
                         encoding="utf-8")
        print(f"measurements written to {cache}")

    (OUTPUT / "table_layer1.tex").write_text(
        table_layer1(data, rates), encoding="utf-8")
    (OUTPUT / "table_mu.tex").write_text(table_mu(grid), encoding="utf-8")
    (OUTPUT / "table_cost.tex").write_text(table_cost(), encoding="utf-8")

    for f in ("table_layer1.tex", "table_mu.tex", "table_cost.tex"):
        print("   ", OUTPUT / f)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
