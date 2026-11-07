"""Tests of the prototype.

They do not test whether DPC "gets it right": that is an experimental result,
and a test demanding F1 above a value would push towards tuning the algorithm
until the test passed. They test the properties that, if broken, invalidate any
number produced: reproducibility, ground-truth consistency, degenerate cases,
DPC's mathematical identities, and a regression test for each defect found.

    python -m pytest tests -q

Tests that need the real networks are skipped when the data is not downloaded.
"""

from __future__ import annotations

import numpy as np
import pytest

from dpcfog import metrics, trust
from dpcfog.config import Params
from dpcfog.experiment import build_batches, run_one, split
from dpcfog.features import extract, standardize
from dpcfog.filters import Batch, ConsensusFilter, DPCFilter, MLPFilter, ReputationFilter
from dpcfog.filters.dpc import (
    assign,
    choose_centers,
    dc_entropy,
    dc_heuristic,
    delta_and_parent,
    density,
    distance_matrix,
)
from dpcfog.filters.dpc_han import HanDPCFilter
from dpcfog.generator import generate
from dpcfog.scenarios import SCENARIOS, apply

SMALL = dict(n_nodes=40, k_targets=6, n_rounds=12, warmup_rounds=3)


def _real_network_or_skip(network: str = "otc"):
    from realdata.networks import load

    try:
        return load(network)
    except (FileNotFoundError, OSError):
        pytest.skip("real network not downloaded (python -m realdata.download)")


# ------------------------------------------------------------- generator


def test_same_seed_same_simulation():
    a = generate(Params(seed=7, **SMALL))
    b = generate(Params(seed=7, **SMALL))
    assert np.array_equal(a.value, b.value)
    assert np.array_equal(a.label, b.label)
    assert np.array_equal(a.t_ref, b.t_ref)


def test_different_seeds_diverge():
    a = generate(Params(seed=1, **SMALL))
    b = generate(Params(seed=2, **SMALL))
    assert not np.array_equal(a.value, b.value)


def test_no_attacker_no_malicious_recommendation():
    s = generate(Params(mu=0.0, **SMALL))
    assert s.label.sum() == 0
    assert s.malicious.sum() == 0


def test_malicious_label_implies_malicious_rater():
    """The ground truth must never mark an honest node's recommendation as an attack."""
    for attack in ("bad_mouthing", "ballot_stuffing", "collusion", "mixed"):
        s = generate(Params(attack=attack, mu=0.3, seed=3, **SMALL))
        assert s.malicious[s.rater[s.label]].all()


def test_values_in_range_and_no_self_recommendation():
    s = generate(Params(mu=0.3, seed=4, **SMALL))
    assert s.value.min() >= 0.0 and s.value.max() <= 1.0
    assert (s.rater != s.target).all()


def test_bad_mouthing_lowers_and_ballot_stuffing_raises():
    bm = generate(Params(attack="bad_mouthing", mu=0.3, seed=5, **SMALL))
    assert bm.value[bm.label].mean() < bm.value[~bm.label].mean()
    bs = generate(Params(attack="ballot_stuffing", mu=0.3, seed=5, **SMALL))
    assert bs.value[bs.label].mean() > bs.t_ref[bs.target[bs.label]].mean()


def test_mu_controls_the_fraction_of_attackers():
    for mu in (0.1, 0.2, 0.3, 0.4):
        s = generate(Params(mu=mu, seed=6, **SMALL))
        assert s.malicious.mean() == pytest.approx(mu, abs=0.03)


def test_attack_prob_below_one_yields_honest_recommendations_from_malicious_nodes():
    s = generate(Params(mu=0.4, attack_prob=0.5, seed=8, **SMALL))
    from_malicious = s.malicious[s.rater]
    assert (from_malicious & ~s.label).any()


def test_heterogeneous_scenario_widens_honest_noise():
    base = Params(attack="bad_mouthing", mu=0.2, seed=9, **SMALL)
    hom = generate(base)
    het = generate(apply(base, "heterogeneous"))
    honest_h = hom.value[~hom.label] - hom.t_ref[hom.target[~hom.label]]
    honest_t = het.value[~het.label] - het.t_ref[het.target[~het.label]]
    assert honest_t.std() > honest_h.std()


def test_every_scenario_generates():
    for name in SCENARIOS:
        if name == "calibrated":
            _real_network_or_skip()
        s = generate(apply(Params(seed=0, **SMALL), name))
        assert len(s) > 0


# ------------------------------------------------------------- features


def test_extract_returns_five_finite_columns():
    p = Params(seed=11, **SMALL)
    s = generate(p)
    window = s.window_indices(0, 5, p.consensus_window)
    batch = s.batch_indices(0, 5)
    x = extract(s.rater, s.target, s.value, s.round_idx, window, batch)
    assert x.shape == (batch.size, 5)
    assert np.isfinite(x).all()


def test_standardize_survives_a_constant_column():
    x = np.ones((10, 5))
    z = standardize(x)
    assert np.isfinite(z).all()
    assert np.allclose(z, 0.0)


def test_deviation_is_signed():
    """Bad-mouthing must give a negative deviation; a positive one means column 0
    is an absolute value and the decision graph loses the attack separation."""
    p = Params(attack="bad_mouthing", mu=0.3, seed=12, **SMALL)
    s = generate(p)
    batches = build_batches(s)
    x = np.vstack([b.x for b in batches])
    y = np.concatenate([b.label for b in batches])
    assert x[y, 0].mean() < 0


# ---------------------------------------------------------------- DPC


def _three_blobs(seed=0):
    rng = np.random.default_rng(seed)
    return np.vstack(
        [
            rng.normal((0, 0), 0.10, size=(60, 2)),
            rng.normal((4, 4), 0.10, size=(40, 2)),
            rng.normal((8, 0), 0.10, size=(20, 2)),
        ]
    )


def test_dpc_finds_the_three_blob_structure():
    x = _three_blobs()
    d = distance_matrix(x)
    dc = dc_heuristic(d, 0.02)
    rho = density(d, dc)
    delta, parent = delta_and_parent(d, rho)
    centres = choose_centers(rho, delta)
    cluster = assign(rho, parent, centres)
    assert centres.size == 3
    assert len(set(cluster[:60])) == 1
    assert len(set(cluster[60:100])) == 1
    assert len(set(cluster[100:])) == 1


def test_delta_of_the_densest_point_is_the_maximum():
    x = _three_blobs(1)
    d = distance_matrix(x)
    rho = density(d, dc_heuristic(d))
    delta, parent = delta_and_parent(d, rho)
    top = int(np.argmax(rho))
    assert delta[top] == pytest.approx(d[top].max())
    assert parent[top] == top


def test_delta_non_negative_and_parent_denser():
    x = _three_blobs(2)
    d = distance_matrix(x)
    rho = density(d, dc_heuristic(d))
    delta, parent = delta_and_parent(d, rho)
    assert (delta >= 0).all()
    others = np.arange(rho.size) != int(np.argmax(rho))
    assert (rho[parent[others]] >= rho[others]).all()


def test_centres_never_collapse_to_one():
    """Regression: the highest-rho point has the largest gamma by construction,
    and using that jump made k always end at 1."""
    x = _three_blobs(3)
    d = distance_matrix(x)
    rho = density(d, dc_heuristic(d))
    delta, _ = delta_and_parent(d, rho)
    assert choose_centers(rho, delta).size >= 2


def test_entropy_dc_is_on_the_distance_scale():
    x = _three_blobs(4)
    d = distance_matrix(x)
    dc = dc_entropy(d)
    assert 0 < dc < d.max()


def test_condensed_dc_matches_the_square_matrix():
    """The minimal-memory path must not change d_c."""
    from dpcfog.filters.dpc import condensed_distances, dc_from_condensed

    x = _three_blobs(6)
    d = distance_matrix(x)
    for fraction in (0.005, 0.02, 0.05):
        assert dc_from_condensed(condensed_distances(x), fraction) == pytest.approx(
            dc_heuristic(d, fraction), rel=1e-5
        )


def test_float32_does_not_change_the_decision():
    """float32 halves the memory of the distance matrix; the test guards that
    the saving is free."""
    from scipy.spatial.distance import pdist, squareform

    from dpcfog.filters import dpc as mod

    x = standardize(np.random.default_rng(9).normal(size=(400, 5)))
    outputs = []
    for dtype in (np.float64, np.float32):
        d = squareform(pdist(x).astype(dtype, copy=False))
        rho = mod.density(d, mod.dc_heuristic(d, 0.02))
        delta, parent = mod.delta_and_parent(d, rho)
        centres = mod.choose_centers(rho, delta)
        outputs.append(mod.assign(rho, parent, centres))
    assert np.array_equal(outputs[0], outputs[1])


def test_dpc_memory_stays_near_the_theoretical_minimum():
    """Regression: an earlier version used about twice the algorithm's minimum
    (index arrays from `triu_indices_from` and four chained n x n temporaries in
    the density). The 1.6x bound leaves room for platform variation."""
    n = 600
    batch = _synthetic_batch(n=n)
    _, peak = metrics.measure_memory(lambda: DPCFilter().flag(batch))
    minimum = n * n * 4 + n * (n - 1) / 2 * 4
    assert peak < 1.6 * minimum, f"{peak / minimum:.2f}x the theoretical minimum"


def test_dc_grows_with_the_fraction():
    d = distance_matrix(_three_blobs(5))
    assert dc_heuristic(d, 0.005) < dc_heuristic(d, 0.05)


# ------------------------------------------------------------ filters


def _synthetic_batch(n=60, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros((n, 5))
    y = np.zeros(n, dtype=bool)
    y[: n // 5] = True
    x[:, 0] = np.where(y, -0.6, 0.0) + rng.normal(0, 0.02, n)
    x[:, 1] = np.where(y, 0.3, 0.8)
    x[:, 3] = np.where(y, 0.2, 0.95)
    return Batch(
        x=x,
        value=rng.uniform(0, 1, n),
        rater=np.arange(n) % 12,
        target=(np.arange(n) * 7) % 20,
        round_no=5,
        fog=0,
        label=y,
    )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: DPCFilter(),
        lambda: DPCFilter(dc_criterion="entropy"),
        lambda: DPCFilter(granularity="recommender"),
        lambda: DPCFilter(legitimate_criterion="cardinality"),
        lambda: ConsensusFilter(),
        lambda: ReputationFilter(),
    ],
)
def test_filter_returns_boolean_of_batch_size(factory):
    batch = _synthetic_batch()
    out = factory().flag(batch)
    assert out.dtype == bool
    assert out.shape == (batch.x.shape[0],)


def test_filters_separate_the_easy_case():
    """A case built to be trivial: failing here is an implementation defect."""
    batch = _synthetic_batch()
    for f in (DPCFilter(), ConsensusFilter(threshold=0.25)):
        d = metrics.evaluate(f.flag(batch), batch.label)
        assert d.f1 > 0.7, f.name


def test_dpc_with_a_tiny_batch_does_not_break():
    for n in (1, 2, 3, 4):
        batch = _synthetic_batch(n=max(n, 1))
        assert DPCFilter().flag(batch).shape == (max(n, 1),)


def test_dpc_with_identical_points_does_not_break():
    n = 20
    batch = Batch(
        x=np.ones((n, 5)),
        value=np.full(n, 0.5),
        rater=np.zeros(n, dtype=int),
        target=np.arange(n),
        round_no=1,
        fog=0,
        label=np.zeros(n, dtype=bool),
    )
    assert np.isfinite(DPCFilter().flag(batch)).all()


def test_dpc_is_deterministic():
    batch = _synthetic_batch(seed=3)
    a = DPCFilter().flag(batch)
    b = DPCFilter().flag(batch)
    assert np.array_equal(a, b)


def test_reputation_accumulates_across_rounds():
    """New evidence moves reputation in its direction without erasing the past."""
    f = ReputationFilter(alpha=0.5)
    good = _synthetic_batch()
    good.x[:, 0] = 0.0            # everybody agrees with the consensus
    f.flag(good)
    before = dict(f.reputation)
    assert all(v == pytest.approx(1.0) for v in before.values())

    bad = _synthetic_batch()
    bad.x[:, 0] = 0.9             # the same raters now disagree
    f.flag(bad)
    for k, v in before.items():
        assert f.reputation[k] < v          # it fell
        assert f.reputation[k] > 0.0        # but kept the past


def test_reputation_without_optimistic_prior_judges_in_the_first_round():
    """Regression seen on Bitcoin OTC: with prior 1.0 a rater appearing once was
    never judged, and the filter returned F1 = 0.000."""
    batch = _synthetic_batch()
    batch.x[:, 0] = 0.9
    assert ReputationFilter().flag(batch).all()
    assert not ReputationFilter(optimistic_prior=True).flag(batch).any()


def test_mlp_without_both_classes_does_not_break():
    batch = _synthetic_batch()
    batch.label[:] = False
    f = MLPFilter()
    f.fit([batch])
    assert f.flag(batch).shape == (batch.x.shape[0],)


# ------------------------------------------------- island versus ramp


def test_bimodality_separates_a_ramp_from_two_modes():
    """If the discriminant feature is unimodal, there is no valley for a density
    method to find."""
    from dpcfog.features import bimodality_coefficient

    rng = np.random.default_rng(0)
    unimodal = rng.normal(0, 1, 4000)
    two_islands = np.concatenate([rng.normal(-3, 0.4, 2000), rng.normal(3, 0.4, 2000)])
    assert bimodality_coefficient(unimodal) < 0.5
    assert bimodality_coefficient(two_islands) > 0.7


def test_ties_degrade_dpc():
    """Same structure, same malicious fraction, same separation; only the
    resolution changes. Rounded to few levels, as integer ratings in a real
    network, density measures how many recommendations share a value instead of
    how anomalous the value is."""
    rng = np.random.default_rng(2)
    n, n_mal = 500, 100
    raw = np.concatenate(
        [rng.normal(0.0, 1.0, n - n_mal), rng.normal(2.5, 1.0, n_mal)]
    )
    y = np.zeros(n, dtype=bool)
    y[-n_mal:] = True

    def run(values):
        x = np.zeros((n, 5))
        x[:, 0] = values
        batch = Batch(
            x=x, value=np.clip((values - values.min()) / np.ptp(values), 0, 1),
            rater=np.arange(n) % 50, target=np.arange(n) % 30,
            round_no=1, fog=0, label=y,
        )
        return metrics.evaluate(DPCFilter().flag(batch), y).f1

    continuous = run(raw)
    discrete = run(np.round(raw * 2) / 2)  # few distinct levels

    assert continuous > discrete, (continuous, discrete)


def test_dc_ignores_zero_distances():
    """With heavily tied data the 2% percentile falls on zero and density divides
    by zero. Regression from Bitcoin OTC, where a 700-point batch had 51
    distinct values."""
    from dpcfog.filters.dpc import dc_from_condensed

    condensed = np.concatenate([np.zeros(900), np.linspace(0.5, 3.0, 100)])
    dc = dc_from_condensed(condensed, 0.02)
    assert dc > 0, "a null d_c would blow up the density"
    assert dc >= 0.5


# ----------------------------------------------------------- metrics


def test_metrics_in_the_perfect_case():
    y = np.array([True, True, False, False])
    d = metrics.evaluate(y, y)
    assert d.f1 == 1.0 and d.precision == 1.0 and d.recall == 1.0


def test_metrics_without_predicted_positives():
    d = metrics.evaluate(np.zeros(4, dtype=bool), np.array([1, 0, 0, 0], dtype=bool))
    assert d.precision == 0.0 and d.recall == 0.0 and d.f1 == 0.0


def test_f1_is_the_harmonic_mean():
    d = metrics.evaluate(
        np.array([1, 1, 1, 0], dtype=bool), np.array([1, 1, 0, 0], dtype=bool)
    )
    expected = 2 * d.precision * d.recall / (d.precision + d.recall)
    assert d.f1 == pytest.approx(expected)


def test_measure_time_refuses_to_run_under_tracemalloc():
    """Time measured with the profiler on is not time."""
    import tracemalloc

    tracemalloc.start()
    try:
        with pytest.raises(RuntimeError):
            metrics.measure_time(lambda: sum(range(10)))
    finally:
        tracemalloc.stop()


def test_measure_memory_reports_a_positive_peak():
    _, peak = metrics.measure_memory(lambda: [0.0] * 200000)
    assert peak > 0


# ---------------------------------------------------------- trust


def test_trust_score_recovers_the_reference_without_attack():
    p = Params(mu=0.0, seed=13, **SMALL)
    s = generate(p)
    t, no_evid = trust.trust_score(
        s.target, s.value, np.ones(len(s), dtype=bool), p.n_nodes
    )
    assert trust.mae(t, s.t_ref) < 0.02
    assert not no_evid.any()


def test_node_without_evidence_gets_the_prior_and_is_flagged():
    target = np.array([0, 0, 1])
    value = np.array([0.9, 0.8, 0.1])
    accepted = np.array([True, True, False])
    t, no_evid = trust.trust_score(target, value, accepted, n_nodes=3)
    assert t[1] == trust.NEUTRAL_PRIOR
    assert no_evid[1] and no_evid[2] and not no_evid[0]


def test_perfect_filtering_reduces_the_trust_score_error():
    p = Params(attack="bad_mouthing", mu=0.3, seed=14, **SMALL)
    s = generate(p)
    t_unf, _ = trust.trust_score(
        s.target, s.value, np.ones(len(s), dtype=bool), p.n_nodes
    )
    t_f, _ = trust.trust_score(s.target, s.value, ~s.label, p.n_nodes)
    assert trust.mae(t_f, s.t_ref) < trust.mae(t_unf, s.t_ref)


# --------------------------------------------------------- experiment


def test_temporal_split_does_not_leak_the_future():
    p = Params(seed=15, **SMALL)
    train, evaluation = split(build_batches(generate(p)), p)
    assert train and evaluation
    assert max(b.round_no for b in train) < min(b.round_no for b in evaluation)
    assert min(b.round_no for b in train) >= p.warmup_rounds


def test_a_run_yields_one_row_per_filter_with_the_csv_columns():
    from dpcfog.experiment import COLUMNS, filter_catalog

    p = Params(seed=16, **SMALL)
    rows = run_one(p, measure_ram=False)
    assert len(rows) == len(filter_catalog(0))
    for row in rows:
        assert set(row) == set(COLUMNS)


def test_unfiltered_mae_is_the_same_for_every_filter():
    """It is the H2 reference condition: if it varied across rows, the
    comparison would compare different baselines."""
    rows = run_one(Params(seed=17, **SMALL), measure_ram=False)
    assert len({r["mae_unfiltered"] for r in rows}) == 1


def test_confusion_counts_add_up_to_the_total():
    for r in run_one(Params(seed=18, **SMALL), measure_ram=False):
        total = r["tp"] + r["fp"] + r["fn"] + r["tn"]
        assert total == r["n_evaluated"]


def test_a_run_is_reproducible():
    a = run_one(Params(seed=19, **SMALL), measure_ram=False)
    b = run_one(Params(seed=19, **SMALL), measure_ram=False)
    for x, y in zip(a, b):
        assert x["f1"] == y["f1"] and x["tp"] == y["tp"]


def test_han_score_ignores_the_x_coordinate():
    """`score` is the distance to the nearest anchor, a function of the value
    only, so `use_id_as_x` changes `flag` but not AUC nor average precision.
    The ranking columns of `dpc_han` and `dpc_han_no_id` are therefore identical
    in the sweep CSV, by property and not by defect."""
    batch = _synthetic_batch()
    with_id = HanDPCFilter(use_id_as_x=True).score(batch)
    without = HanDPCFilter(use_id_as_x=False).score(batch)
    assert np.array_equal(with_id, without)


def test_consensus_capture_inverts_the_deviation():
    """When the attack is the majority about a target, the attacker becomes the
    consensus: 20 attackers and 2 honest raters on the same target, the median
    goes to the attack value, the attacker's deviation is zero and the honest
    rater's is large. The inversion raises no error, it only flips the sign."""
    n_at, n_hon = 20, 2
    rater = np.arange(n_at + n_hon)
    target = np.zeros(n_at + n_hon, dtype=int)
    value = np.concatenate([np.zeros(n_at), np.full(n_hon, 0.8)])
    round_idx = np.zeros(n_at + n_hon, dtype=int)
    idx = np.arange(n_at + n_hon)
    x = extract(rater, target, value, round_idx, idx, idx)

    attacker_dev = np.abs(x[:n_at, 0]).mean()
    honest_dev = np.abs(x[n_at:, 0]).mean()
    assert attacker_dev == pytest.approx(0.0, abs=1e-9)
    assert honest_dev > 0.5
    assert honest_dev > attacker_dev


def test_fixed_anchor_is_immune_to_consensus_capture():
    """Same scenario: the distance to the nearest anchor puts the attacker
    (value 0.0, on the anchor) above the honest rater (0.8), the right order,
    exactly what the deviation inverts."""
    n_at, n_hon = 20, 2
    value = np.concatenate([np.zeros(n_at), np.full(n_hon, 0.8)])
    batch = Batch(
        x=np.zeros((n_at + n_hon, 5)),
        value=value,
        rater=np.arange(n_at + n_hon),
        target=np.zeros(n_at + n_hon, dtype=int),
        round_no=0,
        fog=0,
        label=np.concatenate([np.ones(n_at, bool), np.zeros(n_hon, bool)]),
    )
    sc = HanDPCFilter().score(batch)
    assert sc[:n_at].mean() > sc[n_at:].mean()


def test_rater_and_target_are_independent_index_spaces():
    """In `RealNetwork` rater 5 and target 5 are not the same user. Crossing a
    rater label with a target statistic without the original id gives a
    plausible and meaningless number (benign T* 0.585 against fraudster 0.556,
    when the correct values are 0.628 and 0.268)."""
    net = _real_network_or_skip("otc")

    assert "rater_code" in net.meta and "target_code" in net.meta
    code_r, code_t = net.meta["rater_code"], net.meta["target_code"]
    assert len(code_r) != len(code_t)
    common = set(code_r) & set(code_t)
    assert common, "no node appears both as rater and as target"
    assert any(code_r[i] != code_t[i] for i in common)


def test_on_off_alternates_phases_instead_of_drawing_per_recommendation():
    """On-Off must produce a temporal pattern: an On-Off attacker has whole rounds
    without any attack, alternating with attack rounds, and the phases are not
    synchronised across attackers (that would be collusion)."""
    from dataclasses import replace

    p = replace(Params(), attack="on_off", seed=0)
    sim = generate(p)
    attackers = np.flatnonzero(sim.malicious)
    assert attackers.size > 0

    with_pattern = 0
    for i in attackers[:10]:
        mine = sim.rater == i
        attack_rounds = set(sim.round_idx[mine & sim.label].tolist())
        clean_rounds = set(sim.round_idx[mine].tolist()) - attack_rounds
        if attack_rounds and clean_rounds:
            with_pattern += 1
    assert with_pattern >= 8, "On-Off without an off phase is not On-Off"

    first = [
        min(sim.round_idx[(sim.rater == i) & sim.label].tolist(), default=-1)
        for i in attackers
    ]
    assert len({r for r in first if r >= 0}) > 1


def test_stale_csv_detector_fails_when_a_value_changes(tmp_path):
    """Perturbing an AUC in the CSV must make `validation.stale_csv` fail. A
    checker never seen failing is evidence of nothing, so the test demands the
    failing behaviour."""
    import pathlib
    import shutil

    import pandas as pd

    from validation import stale_csv

    source = pathlib.Path("results/real.csv")
    if not source.exists():
        pytest.skip("results/real.csv has not been generated")
    _real_network_or_skip("otc")

    copy = tmp_path / "real.csv"
    shutil.copy(source, copy)
    d = pd.read_csv(copy)
    target = (d.scenario == "real_otc") & (d["filter"] == "dpc")
    if not target.any():
        pytest.skip("the CSV has no real_otc/dpc row")
    d.loc[target, "auc"] = float(d.loc[target, "auc"].iloc[0]) - 0.05
    d.to_csv(copy, index=False)

    previous = stale_csv.CSV
    try:
        stale_csv.CSV = str(copy)
        assert stale_csv.main([]) == 1, (
            "the detector passed a CSV that differs from the code by 0.05 AUC"
        )
    finally:
        stale_csv.CSV = previous


def test_layer1_interval_resamples_raters():
    """The layer-1 label belongs to the user, and the interval must resample
    users. 20 raters with 200 recommendations each and a per-rater effect on the
    score: resampling raters must give a much wider interval than resampling
    recommendations, and contain the AUC."""
    from sklearn.metrics import roc_auc_score

    import paper_tables as pt

    rng = np.random.default_rng(0)
    rater = np.repeat(np.arange(20), 200)
    y = np.repeat(np.arange(20) % 2 == 0, 200)
    effect = rng.normal(0.0, 1.0, 20)[rater]
    score = 0.3 * y + effect + rng.normal(0.0, 0.3, rater.size)

    lo, hi = pt.bootstrap(score, y, rater, n=200)
    auc = roc_auc_score(y, score)
    assert lo < auc < hi

    widths = []
    for _ in range(200):
        idx = rng.integers(0, y.size, y.size)
        widths.append(roc_auc_score(y[idx], score[idx]))
    width_per_recommendation = np.percentile(widths, 97.5) - np.percentile(widths, 2.5)
    assert hi - lo > 3 * width_per_recommendation, (
        "the interval is not wider than resampling recommendations: the "
        "bootstrap is ignoring the per-rater grouping again"
    )
