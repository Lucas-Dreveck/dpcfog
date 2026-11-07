"""Inference memory and time with 1, 4 and 16 threads.

    python -m analysis.cost_by_threads            # Bitcoin OTC

The paper's cost was measured with the 16 threads of the bench machine, while
an edge node such as the Raspberry Pi 3 has four cores. Part of the measured
memory is a reservation by the linear-algebra library, which could depend on the
number of threads.

Each measurement runs in a fresh process, with OPENBLAS_NUM_THREADS,
OMP_NUM_THREADS and MKL_NUM_THREADS set before importing anything, and
torch.set_num_threads in the child. In the same process it measures:

  memory   increase of private memory during the first pass over the first
           five evaluation batches (the `process` instrument of
           analysis.memory_instruments)
  time     median time per batch over three further passes

The model is trained once in the parent process; only inference changes
configuration. This does not emulate an edge processor, which is also slower
per core: it measures only the effect of the number of threads.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

from .memory_instruments import FILTERS, ROOT, _load_child, _prepare, peak_increase

THREADS = (1, 4, 16)
REPETITIONS = 5


def _child(path: str, n: int) -> None:
    import time

    import numpy  # noqa: F401
    import psutil
    import sklearn.neural_network  # noqa: F401
    import torch

    torch.set_num_threads(n)
    f, sample = _load_child(path)

    increase, _ = peak_increase(psutil.Process(), lambda: [f.flag(b) for b in sample])

    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        [f.flag(b) for b in sample]
        times.append((time.perf_counter() - t0) / len(sample))
    print(json.dumps({"bytes": increase,
                      "ms_per_batch": 1000 * statistics.median(times)}))


def measure(network: str) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        out = {"network": network, **_prepare(network, folder)}
        for n in THREADS:
            env = {**os.environ, "OPENBLAS_NUM_THREADS": str(n),
                   "OMP_NUM_THREADS": str(n), "MKL_NUM_THREADS": str(n)}
            for name in FILTERS:
                mem, elapsed = [], []
                for _ in range(REPETITIONS):
                    r = subprocess.run(
                        [sys.executable, "-m", "analysis.cost_by_threads",
                         "--child", str(folder / f"{name}.pkl"), str(n)],
                        cwd=ROOT, env=env, capture_output=True, text=True,
                        check=True)
                    d = json.loads(r.stdout.strip().splitlines()[-1])
                    mem.append(d["bytes"] / 2**20)
                    elapsed.append(d["ms_per_batch"])
                out[f"{name}_t{n}_mb"] = round(statistics.median(mem), 2)
                out[f"{name}_t{n}_mb_all"] = [round(v, 2) for v in mem]
                out[f"{name}_t{n}_ms"] = round(statistics.median(elapsed), 3)
                print(f"  {n:2d} threads {name:10s} "
                      f"{out[f'{name}_t{n}_mb']:7.2f} MB "
                      f"{out[f'{name}_t{n}_ms']:8.3f} ms/batch", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", default="otc")
    ap.add_argument("--child", nargs=2)
    a = ap.parse_args()
    if a.child:
        _child(a.child[0], int(a.child[1]))
        return
    r = measure(a.network)
    destination = ROOT / "outputs" / f"cost_by_threads_{a.network}.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(r, indent=1), encoding="utf-8")
    print(f"written to {destination}")


if __name__ == "__main__":
    main()
