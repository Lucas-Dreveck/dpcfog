"""Inference memory measured by two instruments under the same conditions.

    python -m analysis.memory_instruments                  # Bitcoin OTC
    python -m analysis.memory_instruments --network alpha

tracemalloc sees what goes through Python's allocator, which includes NumPy
arrays, and therefore DPC and the neural network (scikit-learn's MLPClassifier,
all NumPy). It does not see PyTorch nor the internal buffers of native
libraries, so Deep Trust shows an almost null peak under it.

Here each filter runs in its own clean process, which imports the same libraries
for every filter, loads the already-trained model and the same first five
evaluation batches the experiment uses, and measures the first inference pass
with:

    tracemalloc   the experiment's instrument, in a separate process
    process       private memory of the process, read from the operating
                  system, which sees any allocation: the increase during the
                  pass over the value measured just before it

Variants of `process`: `process_blas` warms up the linear-algebra library with
a small product first, `process_blas_large` with a large one, and
`process_warm` runs the pass once before measuring. Each measurement is repeated
in fresh processes and the median is kept. Training stays in the parent process.

The paper's numbers were measured on Windows, where psutil exposes the private
bytes and a peak counter; elsewhere the script falls back to USS without the
peak counter, which relies on the sampling thread alone.
"""

from __future__ import annotations

import argparse
import gc
import json
import pickle
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import tracemalloc
from pathlib import Path

FILTERS = ("dpc", "mlp", "deeptrust")
REPETITIONS = 5
INSTRUMENTS = ("tracemalloc", "process", "process_blas", "process_warm")
ROOT = Path(__file__).resolve().parent.parent


def private_bytes(proc) -> tuple[int, int]:
    """(private bytes, peak counter); the peak is 0 where it does not exist."""
    info = proc.memory_info()
    if hasattr(info, "private"):
        return int(info.private), int(getattr(info, "peak_pagefile", 0))
    return int(proc.memory_full_info().uss), 0


def _prepare(network: str, folder: Path) -> dict:
    """Train in the parent process and save what each child needs."""
    from dpcfog.experiment import filter_catalog
    from realdata.batches import build_batches, split_by_time
    from realdata.networks import load

    net = load(network)
    train, evaluation = split_by_time(build_batches(net, window=5, n_fog=1))
    sample = evaluation[:5]
    factories = filter_catalog(0)
    summary = {"n_recommendations_sample": int(sum(b.x.shape[0] for b in sample)),
               "largest_batch": int(max(b.x.shape[0] for b in sample))}
    for name in FILTERS:
        f = factories[name]()
        if f.supervised:
            f.fit(train)
        state = None
        params = 0
        if name == "deeptrust":
            state = {k: v.clone() for k, v in f.model.state_dict().items()}
            params = sum(int(v.numel()) * v.element_size() for v in state.values())
            f.model = None
        elif name == "mlp":
            params = sum(a.nbytes for a in f.model.coefs_ + f.model.intercepts_)
        with (folder / f"{name}.pkl").open("wb") as fh:
            pickle.dump({"filter": f, "state": state, "sample": sample}, fh)
        summary[f"model_bytes_{name}"] = params
    return summary


def _load_child(path: str):
    with open(path, "rb") as fh:
        payload = pickle.load(fh)
    f, sample = payload["filter"], payload["sample"]
    if payload["state"] is not None:
        from dpcfog.filters.deeptrust import _Network

        f.model = _Network(int(f.pca.n_components_), f.hidden)
        f.model.load_state_dict(payload["state"])
        f.model.eval()
    del payload
    gc.collect()
    return f, sample


def peak_increase(proc, run) -> tuple[int, bool]:
    """Increase of private memory while `run` executes, and whether the OS peak
    counter was exceeded."""
    base, peak_before = private_bytes(proc)
    maximum = [base]
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            v, _ = private_bytes(proc)
            if v > maximum[0]:
                maximum[0] = v

    t = threading.Thread(target=sample, daemon=True)
    t.start()
    time.sleep(0.02)
    run()
    stop.set()
    t.join()
    _, peak_after = private_bytes(proc)
    exceeded = peak_after > peak_before
    peak = max(maximum[0], peak_after if exceeded else 0)
    return int(peak - base), bool(exceeded)


def _child(path: str, instrument: str) -> None:
    import numpy
    import psutil
    import sklearn.neural_network  # noqa: F401
    import torch  # noqa: F401

    f, sample = _load_child(path)

    if instrument == "tracemalloc":
        tracemalloc.start()
        [f.flag(b) for b in sample]
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(json.dumps({"bytes": int(peak)}))
        return

    if instrument == "process_blas":
        numpy.ones((64, 64)) @ numpy.ones((64, 64))
        gc.collect()
    elif instrument == "process_blas_large":
        numpy.ones((2000, 64)) @ numpy.ones((64, 64))
        gc.collect()
    elif instrument == "process_warm":
        [f.flag(b) for b in sample]
        gc.collect()

    increase, exceeded = peak_increase(psutil.Process(), lambda: [f.flag(b) for b in sample])
    print(json.dumps({"bytes": increase, "counter_exceeded": exceeded}))


def measure(network: str, filters=FILTERS, instruments=INSTRUMENTS) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        summary = _prepare(network, folder)
        out = {"network": network, **summary}
        for name in filters:
            for instrument in instruments:
                values, exceeded = [], []
                for _ in range(REPETITIONS):
                    r = subprocess.run(
                        [sys.executable, "-m", "analysis.memory_instruments",
                         "--child", str(folder / f"{name}.pkl"), instrument],
                        cwd=ROOT, capture_output=True, text=True, check=True,
                    )
                    d = json.loads(r.stdout.strip().splitlines()[-1])
                    values.append(d["bytes"])
                    exceeded.append(d.get("counter_exceeded"))
                out[f"{name}_{instrument}_mb"] = round(
                    statistics.median(values) / 2**20, 2)
                out[f"{name}_{instrument}_all_mb"] = [
                    round(v / 2**20, 2) for v in values]
                if instrument != "tracemalloc":
                    out[f"{name}_{instrument}_counter_exceeded"] = exceeded
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", default="otc")
    ap.add_argument("--child", nargs=2)
    ap.add_argument("--filters", nargs="+", default=list(FILTERS))
    ap.add_argument("--instruments", nargs="+", default=list(INSTRUMENTS))
    a = ap.parse_args()
    if a.child:
        _child(*a.child)
        return
    result = measure(a.network, a.filters, a.instruments)
    for k, v in result.items():
        print(f"{k:36s} {v}")
    complete = (a.filters == list(FILTERS) and a.instruments == list(INSTRUMENTS))
    suffix = "" if complete else "_partial"
    destination = ROOT / "outputs" / f"memory_instruments_{a.network}{suffix}.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"written to {destination}")


if __name__ == "__main__":
    main()
