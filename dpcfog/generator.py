"""Synthetic Fog-IoT data with per-recommendation ground truth.

No public dataset labels recommendation attacks over Trust Scores (CICIoT2023,
for instance, labels intrusion traffic), so a controlled simulation is the only
way to obtain ground truth per recommendation.

Label convention: the classified unit is the recommendation, not the node. A
recommendation issued by a malicious node whose value was not distorted is
labelled benign. With attack_prob = 1.0 (the default) both criteria coincide;
with attack_prob < 1.0 the scenario becomes an intermittent attack and the
distinction matters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Params


@dataclass
class Simulation:
    """One complete run: the network, the recommendations and the ground truth."""

    params: Params
    # --- per node (N,)
    t_ref: np.ndarray        # T*_j, true trustworthiness
    malicious: np.ndarray    # bool
    collusion: np.ndarray    # bool -- coordinated subset
    fog: np.ndarray          # int  -- Fog node in charge
    strategy: np.ndarray     # str  -- attack of each node, empty for benign nodes
    # --- per recommendation (M,)
    rater: np.ndarray        # i
    target: np.ndarray       # j
    value: np.ndarray        # v in [0,1]
    round_idx: np.ndarray    # t
    label: np.ndarray        # bool -- ground truth, True = malicious

    def __len__(self) -> int:
        return int(self.value.size)

    @property
    def malicious_rate(self) -> float:
        return float(self.label.mean())

    def batch_indices(self, fog: int, round_no: int) -> np.ndarray:
        """Recommendations reaching a Fog node in one round: the filter's batch."""
        return np.flatnonzero((self.fog[self.rater] == fog) & (self.round_idx == round_no))

    def window_indices(self, fog: int, round_no: int, window: int) -> np.ndarray:
        """History visible to that Fog node: the last `window` rounds."""
        start = max(0, round_no - window + 1)
        return np.flatnonzero(
            (self.fog[self.rater] == fog)
            & (self.round_idx >= start)
            & (self.round_idx <= round_no)
        )


def _sample_targets(rng, rater, candidates, k):
    """K distinct targets, no self-recommendation."""
    pool = candidates[candidates != rater]
    if pool.size <= k:
        return pool
    return rng.choice(pool, size=k, replace=False)


def _badmouth(rng, p, t_target, sigma):
    """Bad-mouthing. The stealthy mode deviates by a bounded margin from the true
    value, which brings the attack tail close to the honest noise tail."""
    n = t_target.size
    if p.stealthy:
        margin = rng.uniform(p.stealthy_deviation[0], p.stealthy_deviation[1], size=n)
        base = t_target - margin
    else:
        base = np.full(n, p.bad_mouthing_value)
    return np.clip(base + rng.normal(0.0, sigma, size=n), 0.0, 1.0)


def _ballot_stuff(rng, p, t_target, sigma):
    """Ballot-stuffing, symmetric to the above."""
    n = t_target.size
    if p.stealthy:
        margin = rng.uniform(p.stealthy_deviation[0], p.stealthy_deviation[1], size=n)
        base = t_target + margin
    else:
        base = np.full(n, p.ballot_stuffing_value)
    return np.clip(base + rng.normal(0.0, sigma, size=n), 0.0, 1.0)


def generate(params: Params) -> Simulation:
    params.validate()
    p = params
    rng = np.random.default_rng(p.seed)
    n = p.n_nodes

    # --- roles
    n_mal = int(round(p.mu * n))
    malicious = np.zeros(n, dtype=bool)
    malicious[rng.choice(n, size=n_mal, replace=False)] = True
    idx_mal = np.flatnonzero(malicious)
    idx_ben = np.flatnonzero(~malicious)

    # --- reference Trust Score
    t_ref = np.empty(n)
    t_ref[idx_ben] = rng.uniform(p.t_benign[0], p.t_benign[1], size=idx_ben.size)
    t_ref[idx_mal] = rng.uniform(p.t_malicious[0], p.t_malicious[1], size=idx_mal.size)

    # --- strategy of each attacker
    strategy = np.full(n, "", dtype=object)
    if p.attack == "mixed":
        strategy[idx_mal] = rng.choice(
            ["bad_mouthing", "ballot_stuffing", "collusion"], size=idx_mal.size
        )
    else:
        strategy[idx_mal] = p.attack

    collusion = np.zeros(n, dtype=bool)
    candidates = np.array([i for i in idx_mal if strategy[i] == "collusion"], dtype=int)
    if p.attack == "collusion" and candidates.size:
        n_coord = min(
            candidates.size, max(2, int(round(p.collusion_size * candidates.size)))
        )
        coord = rng.choice(candidates, size=n_coord, replace=False)
        collusion[coord] = True
        strategy[np.setdiff1d(candidates, coord)] = "bad_mouthing"
    elif candidates.size:
        collusion[candidates] = True

    # Shared victims: what sets collusion apart from isolated bad-mouthing is
    # the concentration on the same target, able to shift the median itself.
    idx_collusion = np.flatnonzero(collusion)
    if idx_collusion.size and idx_ben.size:
        victims = rng.choice(
            idx_ben,
            size=max(1, int(round(p.victim_fraction * idx_ben.size))),
            replace=False,
        )
    else:
        victims = np.array([], dtype=int)

    # --- On-Off phase, one per attacker
    # Without a phase offset all attackers would switch on and off together,
    # shifting the consensus as a block, which is synchronised collusion and a
    # different attack. The random phase keeps On-Off an individual behaviour.
    phase = rng.integers(0, max(p.on_off_period, 1), size=n)

    # --- observation noise per node
    if p.heterogeneous_sigma:
        node_sigma = rng.uniform(p.sigma_range[0], p.sigma_range[1], size=n)
    else:
        node_sigma = np.full(n, p.noise_sigma)

    # --- Fog partition: balanced and shuffled
    fog = np.array([i % p.n_fog for i in range(n)])
    rng.shuffle(fog)

    everyone = np.arange(n)
    rater_l, target_l, value_l, round_l, label_l = [], [], [], [], []

    for t in range(p.n_rounds):
        for i in range(n):
            st = strategy[i]
            if st == "on_off":
                # on phase: attacks as bad-mouthing. Off phase: issues an honest
                # recommendation, indistinguishable from a benign node.
                on = (
                    (t + phase[i]) % max(p.on_off_period, 1)
                ) < p.active_fraction * max(p.on_off_period, 1)
                st = "bad_mouthing" if on else ""
            if collusion[i] and victims.size:
                k_v = max(1, int(round(p.collusion_focus * p.k_targets)))
                targets_v = rng.choice(
                    victims, size=min(k_v, victims.size), replace=False
                )
                remaining = p.k_targets - targets_v.size
                pool_c = idx_collusion[idx_collusion != i]
                if pool_c.size and remaining > 0:
                    targets_c = rng.choice(
                        pool_c, size=min(remaining, pool_c.size), replace=False
                    )
                else:
                    targets_c = np.array([], dtype=int)
                targets = np.concatenate([targets_v, targets_c]).astype(int)
                missing = p.k_targets - targets.size
                if missing > 0:
                    rest = np.setdiff1d(everyone[everyone != i], targets)
                    targets = np.concatenate(
                        [targets, rng.choice(rest, size=missing, replace=False)]
                    )
            else:
                targets = _sample_targets(rng, i, everyone, p.k_targets)

            # honest value: a noisy observation of T*
            v = np.clip(
                t_ref[targets] + rng.normal(0.0, node_sigma[i], size=targets.size), 0.0, 1.0
            )
            y = np.zeros(targets.size, dtype=bool)

            if st:
                acts = rng.random(targets.size) < p.attack_prob
                benign_target = ~malicious[targets]
                malicious_target = malicious[targets]
                if st == "bad_mouthing":
                    m = acts & benign_target
                    v[m] = _badmouth(rng, p, t_ref[targets][m], node_sigma[i])
                    y |= m
                elif st == "ballot_stuffing":
                    m = acts & malicious_target
                    v[m] = _ballot_stuff(rng, p, t_ref[targets][m], node_sigma[i])
                    y |= m
                elif st == "collusion":
                    # synchronised: lower noise, to mimic a legitimate consensus
                    s = min(node_sigma[i], p.noise_sigma) / 3.0
                    mb = acts & benign_target
                    v[mb] = _badmouth(rng, p, t_ref[targets][mb], s)
                    mm = acts & malicious_target
                    v[mm] = _ballot_stuff(rng, p, t_ref[targets][mm], s)
                    y |= mb | mm

            rater_l.append(np.full(targets.size, i))
            target_l.append(targets)
            value_l.append(v)
            round_l.append(np.full(targets.size, t))
            label_l.append(y)

    return Simulation(
        params=p,
        t_ref=t_ref,
        malicious=malicious,
        collusion=collusion,
        fog=fog,
        strategy=strategy,
        rater=np.concatenate(rater_l).astype(np.int32),
        target=np.concatenate(target_l).astype(np.int32),
        value=np.concatenate(value_l),
        round_idx=np.concatenate(round_l).astype(np.int32),
        label=np.concatenate(label_l),
    )
