"""Does density clustering add anything to the anchors?

    python -m analysis.han_ablation otc       # real fraud and one injected attack
    python -m analysis.han_ablation --sweep   # 2 networks x 3 attacks x 2 q cuts

`HanDPCFilter.score` is the distance to the nearest anchor, a trivial statistic
that does not use DPC; `flag` is the actual algorithm. Comparing both at the
same discard rate isolates the contribution of clustering: if the trivial cut
ties or wins, DPC adds nothing and what works are the anchors.
"""
import sys

import numpy as np

from dpcfog.filters.dpc_han import HanDPCFilter
from realdata.attacks import Attack, inject
from realdata.networks import load

Q_CUTS = (5, 20, 50)


def compare(net, q_cut: int) -> dict | None:
    f = HanDPCFilter()
    q = np.bincount(net.target, minlength=net.n_targets)
    pred = np.zeros(len(net.value), dtype=bool)
    used = np.zeros(len(net.value), dtype=bool)
    for target in np.flatnonzero(q >= q_cut):
        m = net.target == target
        pred[m] = f._decide_one_target(net.value[m])
        used[m] = True
    ev = used & net.evaluable
    if ev.sum() < 50:
        return None
    y = net.label[ev]
    rate = float(pred[ev].mean())
    prec_dpc = float(y[pred[ev]].mean()) if pred[ev].any() else float("nan")

    # trivial cut: discards the same fraction, closest to the anchors first
    sc = -np.minimum(np.abs(net.value - 0.0), np.abs(net.value - 1.0))[ev]
    k = max(1, int(round(rate * sc.size)))
    trivial = np.zeros(sc.size, dtype=bool)
    trivial[np.argsort(-sc)[:k]] = True
    prec_tri = float(y[trivial].mean())

    return {"q": q_cut, "n": int(ev.sum()), "rate": rate, "prev": float(y.mean()),
            "dpc": prec_dpc, "trivial": prec_tri}


def main(network: str = "otc") -> None:
    clean = load(network)
    cases = {
        "real fraud": clean,
        "bad_mouthing mu=0.20": inject(clean, Attack(
            model="random", direction="nuke", attack_size=0.20,
            realistic_filler=True, seed=0)),
    }
    for title, net in cases.items():
        print(f"\n{network} -- {title}   (precision at the same discard rate)")
        print(f"{'q>=':>5} {'recs':>8} {'discard':>8} {'prev':>6} "
              f"{'DPC+anchor':>11} {'anchor only':>12} {'difference':>11}")
        for c in Q_CUTS:
            r = compare(net, c)
            if r:
                print(f"{r['q']:>5} {r['n']:>8} {r['rate']:>8.3f} "
                      f"{r['prev']:>6.3f} {r['dpc']:>11.3f} {r['trivial']:>12.3f} "
                      f"{r['dpc'] - r['trivial']:>+11.3f}")


SWEEP_ATTACKS = (("bad_mouthing", "random", "nuke"),
                 ("ballot_stuffing", "random", "push"),
                 ("bm_stealthy", "average", "nuke"))


def sweep(mu: float = 0.20, seeds=(0, 1, 2), cuts=(5, 20)) -> list[dict]:
    """Layer 2: 2 networks x 3 attacks x 2 q cuts, mean of 3 seeds."""
    rows = []
    for network in ("otc", "alpha"):
        clean = load(network)
        for name, model, direction in SWEEP_ATTACKS:
            nets = [inject(clean, Attack(
                model=model, direction=direction, attack_size=mu,
                realistic_filler=True, seed=s)) for s in seeds]
            for c in cuts:
                rs = [r for r in (compare(net, c) for net in nets) if r]
                if not rs:
                    continue
                dpc = float(np.mean([r["dpc"] for r in rs]))
                tri = float(np.mean([r["trivial"] for r in rs]))
                rows.append({"network": network, "attack": name, "q": c,
                             "dpc": dpc, "trivial": tri, "dif": dpc - tri,
                             "n_seeds": len(rs)})
    return rows


def main_sweep() -> None:
    rows = sweep()
    print("layer 2, mu=0.20, precision at the same discard rate, mean of 3 seeds")
    print(f"{'network':>8} {'attack':>16} {'q>=':>4} {'DPC+anchor':>11} "
          f"{'anchor only':>12} {'difference':>11}")
    for r in rows:
        print(f"{r['network']:>8} {r['attack']:>16} {r['q']:>4} {r['dpc']:>11.3f} "
              f"{r['trivial']:>12.3f} {r['dif']:>+11.3f}")
    difs = [r["dif"] for r in rows]
    print(f"\n{len(rows)} conditions; the trivial cut wins in "
          f"{sum(d < 0 for d in difs)}; difference from {min(difs):+.3f} to "
          f"{max(difs):+.3f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--sweep":
        main_sweep()
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else "otc")
