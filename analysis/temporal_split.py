"""Exact split between training and evaluation, per layer.

    python -m analysis.temporal_split

The rule lives in `realdata.batches.split_by_time` (real networks) and
`dpcfog.experiment.split` (simulation); here it is applied and printed in
rounds, dates and evaluable recommendations. There is no validation set: the
hyperparameters of the trained models are fixed values, with no tuning on
separate data, and training runs a fixed number of iterations.
"""

from __future__ import annotations

from datetime import datetime, timezone

from dpcfog.config import Params
from dpcfog.scenarios import apply
from realdata.batches import build_batches, split_by_time
from realdata.networks import load

LOAD = {
    "otc": (dict(), 1),
    "alpha": (dict(), 1),
    "epinions": (dict(rounds_per_year=365), 8),
}


def _date(net, round_no: int, rounds_per_year: int) -> str:
    """Start of the round, by the same arithmetic as networks.group_into_rounds."""
    sec = float(net.time.min()) + round_no * 365.25 * 24 * 3600 / rounds_per_year
    return datetime.fromtimestamp(sec, tz=timezone.utc).strftime("%m/%Y")


def real_split(network: str) -> dict:
    load_args, n_fog = LOAD[network]
    net = load(network, **load_args)
    batches = build_batches(net, window=5, n_fog=n_fog)
    train, evaluation = split_by_time(batches)
    rpy = load_args.get("rounds_per_year", 12)
    r_tr = sorted({b.round_no for b in train})
    r_ev = sorted({b.round_no for b in evaluation})
    return {
        "rounds_per_year": rpy,
        "n_fog": n_fog,
        "rounds_with_batch": len({b.round_no for b in batches}),
        "train_rounds": len(r_tr),
        "evaluation_rounds": len(r_ev),
        "train_start": _date(net, r_tr[0], rpy),
        "evaluation_start": _date(net, r_ev[0], rpy),
        "evaluation_end": _date(net, r_ev[-1], rpy),
        "train_recommendations": int(sum(b.evaluable.sum() for b in train)),
        "evaluation_recommendations": int(sum(b.evaluable.sum() for b in evaluation)),
        "train_batches": len(train),
        "evaluation_batches": len(evaluation),
    }


def simulated_split() -> dict:
    """The rule of dpcfog.experiment.split applied to a generated simulation."""
    from dpcfog.experiment import build_batches as sim_batches, split
    from dpcfog.generator import generate

    p = apply(Params(attack="mixed", mu=0.2, seed=0), "calibrated")
    train, evaluation = split(sim_batches(generate(p)), p)
    r_tr = sorted({b.round_no for b in train})
    r_ev = sorted({b.round_no for b in evaluation})
    return {
        "rounds": p.n_rounds,
        "warmup": p.warmup_rounds,
        "train": (r_tr[0], r_tr[-1], len(r_tr)),
        "evaluation": (r_ev[0], r_ev[-1], len(r_ev)),
    }


def main() -> None:
    for network in LOAD:
        print(network, real_split(network), flush=True)
    print("simulation", simulated_split())


if __name__ == "__main__":
    main()
