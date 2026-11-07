"""Run the layer-3 simulation grid used in the paper.

    python run_grid.py            # 3 scenarios x 5 attacks x 4 mu x 10 seeds = 600 runs
    python run_grid.py --quick    # 1 seed, a smoke test of a few minutes

The full grid writes `results/full_grid.csv`, the source of every layer-3
number in the paper. The tests run first: a number produced with a broken test
is worth nothing.
"""

from __future__ import annotations

import subprocess
import sys

from dpcfog.config import MU_GRID
from dpcfog.experiment import sweep

SCENARIOS = ("basic", "calibrated", "adverse")
ATTACKS = ("bad_mouthing", "ballot_stuffing", "collusion", "on_off", "mixed")


def main(argv: list[str]) -> int:
    if subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"]).returncode:
        print("tests failed; the experiment does not run")
        return 1
    quick = "--quick" in argv
    output = "results/quick_grid.csv" if quick else "results/full_grid.csv"
    sweep(
        scenario_names=SCENARIOS,
        attacks=ATTACKS,
        mus=MU_GRID,
        seeds=range(1) if quick else range(10),
        output=output,
    )
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
