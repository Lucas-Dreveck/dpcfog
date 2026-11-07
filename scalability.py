"""Does it fit on a Fog node? Filter cost as a function of batch size.

    python scalability.py

DPC computes the distance between every pair of recommendations in the batch,
so memory and time grow with the square of the batch size. That is the nature of
the algorithm, and it determines the largest batch a Fog node can process.

The target hardware is the aggregation node Ullah et al. (2024) used for Deep
Trust, the baseline of this work:

    Raspberry Pi 3 Model B
    ARM Cortex-A53 quad-core 1.2 GHz, 1 GB of RAM
    (Ullah et al., 2024, Section V-A)

The operating system takes part of the 1 GB. The budget adopted for the
filtering process is 256 MB, a quarter of the total.
"""

from __future__ import annotations

import time
import tracemalloc

import numpy as np

from dpcfog.features import standardize
from dpcfog.filters import Batch
from dpcfog.filters.dpc import DPCFilter

# Raspberry Pi 3 Model B, the gateway of Ullah et al. (2024)
TARGET_RAM_MB = 1024
BUDGET_MB = 256
# The Cortex-A53 at 1.2 GHz is much slower than the measurement machine. This
# factor is an estimate from public floating-point comparisons between ARM
# Cortex-A53 and desktop x86, not a measurement.
ARM_FACTOR = 8.0

SIZES = (100, 250, 500, 1000, 2000, 4000, 8000)


def fake_batch(n: int, seed: int = 0) -> Batch:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 5))
    x[: n // 5] += 2.5
    return Batch(
        x=x,
        value=rng.uniform(0, 1, n),
        rater=rng.integers(0, max(2, n // 10), n),
        target=rng.integers(0, max(2, n // 5), n),
        round_no=1,
        fog=0,
        label=np.arange(n) < n // 5,
    )


def measure(n: int, factory) -> tuple[float, float]:
    """(peak memory in MB, time in ms). Separate passes, as always."""
    batch = fake_batch(n)
    f = factory()
    f.flag(batch)  # warm-up, so as not to measure lazy imports

    start = time.perf_counter()
    repetitions = 3 if n <= 2000 else 1
    for _ in range(repetitions):
        factory().flag(batch)
    ms = 1000 * (time.perf_counter() - start) / repetitions

    tracemalloc.start()
    factory().flag(batch)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1e6, ms


def theoretical_mb(n: int, bytes_per_value: int = 4) -> float:
    """Square matrix plus condensed vector, the algorithm's minimum."""
    return (n * n * bytes_per_value + n * (n - 1) / 2 * bytes_per_value) / 1e6


def main() -> None:
    print(__doc__.split("\n\n", 1)[1].rstrip())
    print()
    print("=" * 78)
    print(f"{'batch':>6s} {'minimum':>9s} {'measured':>9s} {'excess':>8s} "
          f"{'time':>9s} {'on Pi 3':>10s} {'fits 256MB':>11s}")
    print(f"{'(n)':>6s} {'(MB)':>9s} {'(MB)':>9s} {'':>8s} "
          f"{'(ms)':>9s} {'(ms est.)':>10s}")
    print("-" * 78)

    limit = None
    for n in SIZES:
        mb, ms = measure(n, lambda: DPCFilter())
        t = theoretical_mb(n)
        fits = mb <= BUDGET_MB
        if fits:
            limit = n
        print(
            f"{n:6d} {t:9.1f} {mb:9.1f} {mb / t:7.2f}x {ms:9.1f} "
            f"{ms * ARM_FACTOR:10.0f} {'yes' if fits else 'NO':>11s}"
        )

    print("-" * 78)
    print(f"Largest measured batch within the {BUDGET_MB} MB budget: n = {limit}")
    n_max = int(np.sqrt(BUDGET_MB * 1e6 / (4 * 1.5)))
    print(f"Projection from the theoretical minimum: n ~ {n_max:,}")
    print()

    print("=" * 78)
    print("Variants compared, batch of 1,000")
    print("=" * 78)
    variants = {
        "dpc (default)": lambda: DPCFilter(),
        "dpc per recommender": lambda: DPCFilter(granularity="recommender"),
        "dpc outlier rule": lambda: DPCFilter(rule="outlier"),
        "dpc entropy d_c": lambda: DPCFilter(dc_criterion="entropy"),
    }
    for name, fac in variants.items():
        mb, ms = measure(1000, fac)
        print(f"  {name:24s} {mb:7.1f} MB {ms:8.1f} ms "
              f"({ms * ARM_FACTOR:6.0f} ms on Pi 3)")

    print()
    print("=" * 78)
    print("Numerical precision: does float32 change the result?")
    print("=" * 78)
    from dpcfog.filters import dpc as mod
    from dpcfog.metrics import evaluate

    for n in (500, 2000):
        batch = fake_batch(n, seed=3)
        z = standardize(batch.x)
        outputs = {}
        for name, dtype in (("float64", np.float64), ("float32", np.float32)):
            d = mod.squareform(mod.pdist(z).astype(dtype, copy=False))
            dc = mod.dc_heuristic(d, 0.02)
            rho = mod.density(d, dc)
            delta, parent = mod.delta_and_parent(d, rho)
            c = mod.choose_centers(rho, delta)
            g = mod.assign(rho, parent, c)
            outputs[name] = g != int(g[int(np.argmax(rho))])
        same = (outputs["float64"] == outputs["float32"]).mean()
        f64 = evaluate(outputs["float64"], batch.label).f1
        f32 = evaluate(outputs["float32"], batch.label).f1
        print(f"  n={n:5d}  identical decisions: {100 * same:5.1f}%  "
              f"F1 {f64:.4f} -> {f32:.4f}")


if __name__ == "__main__":
    main()
