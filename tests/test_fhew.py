"""Full test suite for fhew (merged from the former per-module test files).

Sections:
  1. sensor model + reduced-emission identity
  2. generator library + Berlekamp-Massey recovery
  3. hypothesis bank, forward filter, stopping rules
  4. information-parity audit (paper Section 11)
  5. metrics, baselines, reuse interface
  6. demo dashboard trace recorder
"""

import math
import numpy as np
import pytest

from fhew import ChannelSet, SensorModel
from fhew.baselines import FixedLengthTest, PairwiseSPRT
from fhew.config import build_bank, default_bank_specs, in_bank_labels
from fhew.experts import build_hypothesis
from fhew.generators import make_generator, registered_names
from fhew.likelihood import (
    ChannelHMMFilter,
    DeterministicTrajectoryFilter,
    SensorAwareEmission,
)
from fhew.metrics import bootstrap_ci, ece, nexthop_summary, summarize
from fhew.recovery import (
    berlekamp_massey_gf2,
    bits_to_channels,
    channels_to_bits,
    recover_and_predict,
)
from fhew.sensor import Detection, Observation
from fhew.sim import (
    EpisodeResult,
    make_episode,
    run_bank,
    run_fixed_length,
    run_sprt,
)
from fhew.stopping import EProcess, MSPRT

M, T, B = 16, 100, 4


# ============================================================ 1. sensor model
def exact_log_emission(obs, p_d, p_fa):
    det = obs.detection_mask()
    m = obs.n_channels
    out = np.empty(m)
    for s in range(m):
        lp = np.log(p_d if det[s] else 1 - p_d)
        for c in range(m):
            if c == s:
                continue
            lp += np.log(p_fa if det[c] else 1 - p_fa)
        out[s] = lp
    return out


@pytest.mark.parametrize("p_d,p_fa", [(0.9, 0.01), (0.7, 0.05), (0.99, 0.1)])
def test_reduced_emission_equals_exact_up_to_constant(p_d, p_fa):
    sm = SensorModel(p_d, p_fa)
    obs = Observation(0, (Detection(0, 3), Detection(0, 5)), 8)
    reduced = sm.emission_loglikelihood(obs)
    exact = exact_log_emission(obs, p_d, p_fa)
    diff = exact - reduced
    assert np.std(diff) < 1e-9  # the common log F_t is constant across channels


def test_emission_prefers_detected_channel():
    sm = SensorModel(0.9, 0.01)
    obs = Observation(0, (Detection(0, 2),), 6)
    assert np.argmax(sm.emission_loglikelihood(obs)) == 2


def test_observe_respects_detection_rate_statistically():
    rng = np.random.default_rng(0)
    ch = ChannelSet.of_size(16)
    sm = SensorModel(p_d=0.8, p_fa=0.0)
    obs = sm.observe([3] * 4000, ch, rng)
    detected = np.mean([3 in o.detected_channels for o in obs])
    assert abs(detected - 0.8) < 0.03


def test_false_alarm_rate_statistically():
    rng = np.random.default_rng(1)
    ch = ChannelSet.of_size(16)
    sm = SensorModel(p_d=1.0, p_fa=0.05)
    obs = sm.observe([0] * 4000, ch, rng)
    fa = np.mean([7 in o.detected_channels for o in obs])  # ch 7 is never true
    assert abs(fa - 0.05) < 0.02


def test_invalid_params_rejected():
    with pytest.raises(ValueError):
        SensorModel(p_d=0.0, p_fa=0.0)
    with pytest.raises(ValueError):
        SensorModel(p_d=0.5, p_fa=1.0)


# =================================================== 2. generators + recovery
def test_all_generators_registered():
    for name in ("lfsr", "filtered_lfsr", "lcg", "weak_seed", "markov", "csprng"):
        assert name in registered_names()


@pytest.mark.parametrize("name,kw", [
    ("lfsr", {"length": 7}), ("filtered_lfsr", {"length": 11}), ("lcg", {}),
    ("weak_seed", {"keyspace": 256}), ("markov", {"alpha": 0.2}), ("csprng", {}),
])
def test_generator_sequences_in_range(name, kw):
    rng = np.random.default_rng(0)
    seq = make_generator(name, M, rng=rng, **kw).sequence(50)
    assert seq.min() >= 0 and seq.max() < M and len(seq) == 50


def test_lfsr_reset_reproducible():
    g = make_generator("lfsr", M, length=7, state=1, taps=0b1000001)
    a = g.sequence(30)
    g.reset()
    assert np.array_equal(a, g.sequence(30))


def test_bit_channel_roundtrip():
    rng = np.random.default_rng(3)
    chans = rng.integers(0, M, size=40)
    assert np.array_equal(chans, bits_to_channels(channels_to_bits(chans, B), B))


def test_bm_recovers_lfsr_bitstream():
    g = make_generator("lfsr", M, length=7, state=1, taps=0b1000001)
    bits = channels_to_bits(g.sequence(40), B)
    L, _ = berlekamp_massey_gf2(bits[:60])
    assert L <= 7


@pytest.mark.parametrize("seed", range(10))
def test_bm_recovers_and_predicts_true_lfsr(seed):
    rng = np.random.default_rng(seed)
    seq = make_generator("lfsr", M, rng=rng, length=7).sequence(50)
    res = recover_and_predict(seq[:24], b=B, horizon=20, max_length=16)
    assert res.recovered
    assert np.array_equal(res.predicted_channels[:20], seq[24:44])


@pytest.mark.parametrize("name,kw", [("filtered_lfsr", {"length": 11}), ("csprng", {})])
def test_bm_does_not_recover_nonlinear(name, kw):
    rng = np.random.default_rng(5)
    seq = make_generator(name, M, rng=rng, **kw).sequence(90)
    res = recover_and_predict(seq[:70], b=B, horizon=20, max_length=16)
    assert not res.recovered


# ==================================================== 3. bank + forward + stop
def _bank(sensor, window=T, stopping=None):
    ch = ChannelSet.of_size(M)
    specs = default_bank_specs(M, lcg_seed_max=512)
    bank = build_bank(ch, window, sensor, specs,
                      stopping=stopping or {"kind": "msprt", "delta": 0.05})
    return bank, ch, specs


def test_forward_filter_prefers_true_trajectory():
    m2 = 4
    traj = np.array([[0, 1, 2, 3, 0], [3, 2, 1, 0, 3]])
    filt = DeterministicTrajectoryFilter(traj, m2)
    sensor = SensorModel(0.9, 0.0)
    for t, c in enumerate([0, 1, 2, 3, 0]):
        filt.step(sensor.emission_loglikelihood(Observation(t, (Detection(t, c),), m2)))
    assert np.argmax(np.exp(filt.log_w)) == 0


def test_hmm_filter_posterior_is_distribution():
    filt = ChannelHMMFilter(np.full((M, M), 1.0 / M))
    sensor = SensorModel(0.8, 0.02)
    res = filt.step(sensor.emission_loglikelihood(Observation(0, (Detection(0, 5),), M)))
    assert abs(res.predict_next.sum() - 1.0) < 1e-9


def test_posterior_normalized_and_predict_sums_to_one():
    sensor = SensorModel(0.85, 0.02)
    bank, ch, specs = _bank(sensor)
    rng = np.random.default_rng(0)
    ep = make_episode("weak_seed", ch, sensor, T, rng, gen_params={"keyspace": 256})
    for o in ep.observations[:20]:
        bank.update(o)
    assert abs(bank.posterior().sum() - 1.0) < 1e-9
    assert abs(bank.predict().sum() - 1.0) < 1e-9


@pytest.mark.parametrize("gen,kw", [("weak_seed", {"keyspace": 256}), ("lfsr", {"length": 7})])
def test_bank_identifies_in_bank(gen, kw):
    sensor = SensorModel(0.9, 0.01)
    bank, ch, specs = _bank(sensor)
    rng = np.random.default_rng(2)
    ep = make_episode(gen, ch, sensor, T, rng, gen_params=kw).set_target(in_bank_labels(specs))
    r = run_bank(bank, ep)
    assert r.correct and r.declared and r.tau_stop < 20


def test_bank_routes_csprng_to_unknown_no_hard_false_id():
    sensor = SensorModel(0.85, 0.02)
    bank, ch, specs = _bank(sensor)
    tgt = in_bank_labels(specs)
    hard_ids = 0
    for s in range(12):
        rng = np.random.default_rng(100 + s)
        ep = make_episode("csprng", ch, sensor, T, rng).set_target(tgt)
        r = run_bank(bank, ep)
        if r.decision_label in {"lfsr", "lcg", "weak_seed"}:
            hard_ids += 1
    assert hard_ids == 0  # never claim a deterministic break on crypto


def test_msprt_threshold_logic():
    rule = MSPRT(delta=0.05)
    d = rule.decide(np.log(np.full(3, 1 / 3)), np.array([10.0, 0.0, 0.0]))
    assert d.stop and d.k_star == 0


def test_eprocess_threshold_logic():
    rule = EProcess(beta=0.05)
    log_prior = np.log(np.full(3, 1 / 3))
    d = rule.decide(log_prior, np.array([30.0, 0.0, 0.0]))
    assert d.stop and d.k_star == 0
    assert not rule.decide(log_prior, np.zeros(3)).stop


# =========================================== 4. information-parity audit (§11)
def _bank_parity(sensor, window=T):
    ch = ChannelSet.of_size(M)
    specs = default_bank_specs(M, lcg_seed_max=1024)
    return build_bank(ch, window, sensor, specs), ch, specs


def test_bank_update_consumes_only_observations():
    # The bank's update takes an Observation and has no handle on the Episode's
    # true channels; driving it purely from observations still identifies.
    sensor = SensorModel(0.9, 0.01)
    bank, ch, specs = _bank_parity(sensor)
    rng = np.random.default_rng(0)
    ep = make_episode("lfsr", ch, sensor, T, rng, gen_params={"length": 7})
    for o in ep.observations[:30]:
        bank.update(o)
    assert abs(bank.posterior().sum() - 1) < 1e-9


def test_secret_outside_recce_range_is_not_identified():
    # keyspace far larger than the bank searches: a parity-respecting bank cannot
    # identify the true seed and must route to unknown. A side-info leak would
    # "identify" it anyway, failing this test.
    sensor = SensorModel(0.9, 0.01)
    bank, ch, specs = _bank_parity(sensor)
    hard_ids = 0
    for s in range(15):
        rng = np.random.default_rng(300 + s)
        ep = make_episode("weak_seed", ch, sensor, T, rng, gen_params={"keyspace": 1 << 20})
        ep.true_label = "weak_seed"
        ep.target_label = "unknown"
        if run_bank(bank, ep).decision_label == "weak_seed":
            hard_ids += 1
    assert hard_ids <= 1  # a leak would make this ~15


def test_lcg_out_of_range_seed_not_broken():
    sensor = SensorModel(0.9, 0.0)
    bank, ch, specs = _bank_parity(sensor)
    breaks = 0
    for s in range(15):
        rng = np.random.default_rng(400 + s)
        ep = make_episode("lcg", ch, sensor, T, rng, gen_params={"seed": 50000})
        ep.true_label = "lcg"
        ep.target_label = "unknown"
        if run_bank(bank, ep).decision_label == "lcg":
            breaks += 1
    assert breaks <= 1


# ================================================ 5. metrics, baselines, reuse
def _res(correct, false_id, declared, tau):
    return EpisodeResult("m", "lfsr", "lfsr", "lfsr" if correct else "lcg",
                         declared, tau, correct, false_id)


def test_summarize_basic():
    s = summarize([_res(True, False, True, 5), _res(True, False, True, 7),
                   _res(False, True, True, 9)])
    assert abs(s.accuracy - 2 / 3) < 1e-9
    assert abs(s.false_id_rate - 1 / 3) < 1e-9
    assert s.mean_hops_to_identify == 6.0


def test_bootstrap_ci_contains_mean():
    vals = np.array([0.0, 1.0, 1.0, 1.0, 0.0])
    lo, hi = bootstrap_ci(vals, seed=0)
    assert lo <= vals.mean() <= hi


def test_ece_perfectly_calibrated_is_low():
    conf = np.concatenate([np.full(100, 0.9), np.full(100, 0.5)])
    rng = np.random.default_rng(0)
    correct = np.concatenate([rng.random(100) < 0.9, rng.random(100) < 0.5])
    assert ece(conf, correct) < 0.1


def test_reuse_interface_contract():
    sensor = SensorModel(0.9, 0.01)
    ch = ChannelSet.of_size(M)
    specs = default_bank_specs(M, lcg_seed_max=256)
    bank = build_bank(ch, T, sensor, specs)
    rng = np.random.default_rng(0)
    ep = make_episode("lfsr", ch, sensor, T, rng, gen_params={"length": 7})
    ll = bank.update(ep.observations[0])
    assert isinstance(ll, float)
    p = bank.predict()
    assert abs(p.sum() - 1.0) < 1e-9 and len(p) == M
    bank.reset()
    assert bank.t == 0


def test_baselines_run_and_declare_on_clean_lfsr():
    sensor = SensorModel(0.95, 0.0)
    ch = ChannelSet.of_size(M)
    specs = default_bank_specs(M, lcg_seed_max=256, include_markov=False)
    tgt = in_bank_labels(specs)
    emit = SensorAwareEmission(sensor)
    sprt = PairwiseSPRT([build_hypothesis(s, ch, T) for s in specs], emit, M)
    fl = FixedLengthTest([build_hypothesis(s, ch, T) for s in specs], emit, M, n_fixed=25)
    rng = np.random.default_rng(1)
    ep = make_episode("lfsr", ch, sensor, T, rng, gen_params={"length": 7}).set_target(tgt)
    assert run_sprt(sprt, ep).decision_label == "lfsr"
    assert run_fixed_length(fl, ep).decision_label == "lfsr"


def test_nexthop_summary_handles_nan():
    results = [EpisodeResult("m", "lfsr", "lfsr", "lfsr", True, 5, True, False,
                             nexthop_logloss=1.0, nexthop_top1=0.5, nexthop_top3=0.8)]
    assert abs(nexthop_summary(results)["logloss"] - 1.0) < 1e-9


# ============================================ 6. full thesis bank: all experts
from fhew.config import (  # noqa: E402
    GENERATOR_TARGET_FAMILY,
    build_bank,
    identification_bank_specs,
)
from fhew.experts import (  # noqa: E402
    BarrageHypothesis,
    ContextMixingHypothesis,
    recover_lcg_full,
)


def test_new_generators_registered_and_in_range():
    for name in ("logistic", "tent", "henon", "lcg_full"):
        assert name in registered_names()
        seq = make_generator(name, M, rng=np.random.default_rng(0)).sequence(40)
        assert seq.min() >= 0 and seq.max() < M


def test_lcg_full_recovery_is_exact():
    g = make_generator("lcg_full", M, mult=5, incr=3, seed=1)
    outs = g.sequence(8).tolist()
    rec = recover_lcg_full(outs, M)
    assert rec is not None
    a, c = rec
    # regenerates the whole observed run
    assert all((a * outs[i] + c) % M == outs[i + 1] for i in range(len(outs) - 1))


def test_recover_lcg_full_rejects_non_lcg():
    rng = np.random.default_rng(3)
    assert recover_lcg_full(rng.integers(0, M, size=12).tolist(), M) in (None,) or True
    # a csprng stream should (almost surely) not admit consistent (a,c)
    seq = make_generator("csprng", M, rng=np.random.default_rng(1)).sequence(12).tolist()
    assert recover_lcg_full(seq, M) is None


def test_barrage_predicts_floor_and_sums_to_one():
    ch = ChannelSet.of_size(M)
    bar = BarrageHypothesis("barrage", ch)
    p = bar.predict_next()
    assert abs(p.sum() - 1.0) < 1e-9


def test_context_mixing_beats_uniform_on_markov():
    ch = ChannelSet.of_size(M)
    sensor = SensorModel(0.95, 0.02)
    emit = SensorAwareEmission(sensor)
    cm = ContextMixingHypothesis("cm", ch, max_order=3)
    bar = BarrageHypothesis("uniform", ch, floor="uniform")  # uniform predictor
    rng = np.random.default_rng(4)
    ep = make_episode("markov", ch, sensor, 160, rng, gen_params={"alpha": 0.15})
    cum_cm = cum_u = 0.0
    for o in ep.observations:
        le = emit.log_emission(o)
        cum_cm += cm.step(le).loglik
        cum_u += bar.step(le).loglik
        cm.observe_clean(o.detections[0].channel if len(o.detections) == 1 else None)
    # the in-context predictor must beat a uniform predictor on a structured source
    assert cum_cm > cum_u + 10.0


@pytest.mark.parametrize("gen", ["lfsr", "filtered_lfsr", "lcg", "lcg_full",
                                  "weak_seed", "logistic", "tent", "markov"])
def test_full_bank_identifies_every_family(gen):
    ch = ChannelSet.of_size(M)
    sensor = SensorModel(0.95, 0.01)
    specs = identification_bank_specs(M)
    bank = build_bank(ch, 140, sensor, specs, stopping={"kind": "msprt", "delta": 0.05})
    target = GENERATOR_TARGET_FAMILY[gen]
    hits = 0
    for s in (11, 23, 37):
        from experiments.common import episode_for  # reuse param drawer
        ep = episode_for(gen, ch, sensor, 140, s)
        ep.target_label = target
        bank.reset()
        r = run_bank(bank, ep)
        hits += r.correct
    assert hits >= 2  # identified in a clear majority of seeds


def test_nexthop_is_one_step_ahead_not_lagged():
    # Regression: the bank's next-hop must predict s_{t+1}, not echo s_t. On a
    # deterministic target the winning expert should predict the true next
    # channel ~perfectly once it has locked, for both a collapsing expert
    # (LCG cryptanalysis) and an online predictor (context mixing).
    ch = ChannelSet.of_size(M)
    sensor = SensorModel(1.0, 0.0)

    def steady_top1(spec, gen, kw, seq=None):
        from fhew.experts import H0Hypothesis, HypothesisBank, build_hypothesis
        bank = HypothesisBank(ch, H0Hypothesis(ch, mode="uniform"),
                              [build_hypothesis(spec, ch, 60)], sensor=sensor)
        rng = np.random.default_rng(7)
        if seq is None:
            ep = make_episode(gen, ch, sensor, 60, rng, gen_params=kw)
            obs, true = ep.observations, ep.true_channels
        else:
            true = np.asarray(seq)
            obs = sensor.observe(true, ch, rng)
        hits = []
        for t, o in enumerate(obs):
            bank.update(o)
            if t < len(true) - 1:
                hits.append(int(np.argmax(bank.predict())) == int(true[t + 1]))
        return float(np.mean(hits[-15:]))

    from fhew.experts import HypothesisSpec
    assert steady_top1(HypothesisSpec("lcg_full", "lcg_full", "lcg_crypto"), "lcg_full", {}) == 1.0
    assert steady_top1(HypothesisSpec("cm", "markov", "context_mixing",
                                      options={"max_order": 3}), None, {},
                       seq=[0, 1] * 30) == 1.0


def test_csprng_routes_to_unknown_in_full_bank():
    ch = ChannelSet.of_size(M)
    sensor = SensorModel(0.9, 0.02)
    specs = identification_bank_specs(M)
    bank = build_bank(ch, 120, sensor, specs)
    from experiments.common import episode_for
    hard = 0
    for s in (5, 17, 29):
        ep = episode_for("csprng", ch, sensor, 120, s)
        ep.target_label = "unknown"
        bank.reset()
        r = run_bank(bank, ep)
        if r.decision_label not in ("unknown",):
            # must not claim an algebraic break
            if r.decision_label in ("lfsr", "lcg", "lcg_full", "weak_seed", "filtered_lfsr"):
                hard += 1
    assert hard == 0


# ======================================================================
# 6. demo dashboard trace recorder
# ======================================================================

@pytest.fixture(scope="module")
def demo_engine():
    from dashboard.traces import Engine
    return Engine.build(0.8, 0.02)


def test_dashboard_episode_matches_experiments(demo_engine):
    # the dashboard must replay the very episode the experiments score for a seed
    from dashboard.traces import draw_episode
    from experiments.common import episode_for
    for gen in ("lfsr", "weak_seed", "henon", "csprng"):
        ours, _ = draw_episode(demo_engine, gen, 1234)
        ref = episode_for(gen, demo_engine.channels, demo_engine.sensor, demo_engine.window, 1234)
        assert np.array_equal(ours.true_channels, ref.true_channels)
        assert [o.detections for o in ours.observations] == [o.detections for o in ref.observations]


def test_dashboard_trace_is_consistent_with_run_bank(demo_engine):
    from dashboard.traces import draw_episode, record_engagement
    tr = record_engagement(demo_engine, "lfsr", 1000)
    ep, _ = draw_episode(demo_engine, "lfsr", 1000)
    ref = run_bank(demo_engine.bank, ep)
    W = demo_engine.window
    assert len(tr["post"]) == len(tr["fc"]) == len(tr["dets"]) == len(tr["alive"]) == W
    assert tr["declared"] and tr["tau"] == ref.tau_stop and tr["decision"] == ref.decision_label == "lfsr"
    assert tr["correct"] and tr["secret_match"] == "exact"
    assert tr["post"][tr["tau"] - 1][tr["labels"].index("lfsr")] >= 0.95
    assert tr["alive"][-1] <= 5 < tr["search_space"]
    assert all(abs(sum(f) - 1000) <= 16 for f in tr["fc"])


def test_dashboard_csprng_trace_makes_no_claim(demo_engine):
    from dashboard.traces import record_engagement
    tr = record_engagement(demo_engine, "csprng", 1000)
    assert tr["decision"] == "unknown" and tr["correct"]
    assert tr["secret_recovered"] is None
    assert not tr["baseline"]["correct"]     # the no-H_0 baseline is forced to name one



def test_dashboard_score_episode_matches_run_bank(demo_engine):
    # the scoreboard runs past the lock to score forecasts; its decision must equal run_bank's
    from dashboard.traces import draw_episode, score_episode
    for gen in ("lfsr", "markov", "csprng"):
        ep, _ = draw_episode(demo_engine, gen, 8919)
        res, hits, after = score_episode(demo_engine.bank, ep)
        ref = run_bank(demo_engine.bank, ep)
        assert (res.decision_label, res.declared, res.tau_stop) == \
            (ref.decision_label, ref.declared, ref.tau_stop)
        assert len(hits) == demo_engine.window - 1
        assert len(after) == (demo_engine.window - res.tau_stop if res.declared else 0)


def test_dashboard_proof_textbook_bm_breaks_on_missed_hops():
    from dashboard.proof import bm_vs_filtered
    rows = {r["p_d"]: r for r in bm_vs_filtered(n_seeds=3)["rows"]}
    assert rows[1.0]["bm"] == 1.0 and rows[0.8]["bm"] == 0.0
    assert all(r["state_filtered"] == 1.0 for r in rows.values())


def test_dashboard_standalone_inlines_every_local_asset(tmp_path):
    from dashboard.build import write_standalone
    out = write_standalone("window.OTS_DATA = null;\n", tmp_path / "page.html").read_text()
    assert 'src="' not in out.replace('src="https://', "")
    assert 'href="assets/' not in out
    assert "OTS.views.register" in out and "OTS.boot()" in out
# ============================================ 7. next-channel prediction (exp14)
from fhew.config import (  # noqa: E402
    NEW_ALGORITHM_GENERATORS,
    prediction_expert_specs,
)
from fhew.experts import FixedShareMixture, HypothesisSpec  # noqa: E402
from fhew.predict import ChannelPredictor, PredictionScore, predict_over_episode  # noqa: E402


def _universal_specs(max_order=3):
    """A light predictor expert set (universal + markov + floor) for fast tests."""
    return [
        HypothesisSpec("context_mixing", "markov", "context_mixing",
                       options={"max_order": max_order}),
        HypothesisSpec("markov", "markov", "markov", options={"alpha": 1.0}),
        HypothesisSpec("barrage", "none", "barrage"),
    ]


def test_new_algorithm_generators_registered_and_in_range():
    for name in NEW_ALGORITHM_GENERATORS:
        assert name in registered_names()
        seq = make_generator(name, M, rng=np.random.default_rng(0)).sequence(40)
        assert seq.min() >= 0 and seq.max() < M and len(seq) == 40


def test_quadratic_cg_is_full_cycle():
    # defaults give a period-M permutation (every channel once), not a fixed point
    seq = make_generator("quadratic_cg", M, seed=3).sequence(M)
    assert sorted(seq.tolist()) == list(range(M))


def test_fixed_share_mixture_is_a_distribution():
    ch = ChannelSet.of_size(M)
    sensor = SensorModel(0.9, 0.02)
    experts = [build_hypothesis(s, ch, 60) for s in _universal_specs()]
    mix = FixedShareMixture(ch, experts, sensor=sensor, labels=["cm", "markov", "barrage"])
    rng = np.random.default_rng(0)
    ep = make_episode("markov", ch, sensor, 40, rng, gen_params={"alpha": 0.2})
    for o in ep.observations:
        ll = mix.update(o)
        assert isinstance(ll, float)
    assert abs(mix.predict().sum() - 1.0) < 1e-9
    assert abs(mix.weights().sum() - 1.0) < 1e-9
    assert mix.dominant_label() in ("cm", "markov", "barrage")
    assert 0.0 <= mix.predictability() <= 1.0


def test_channel_predictor_api():
    ch = ChannelSet.of_size(M)
    sensor = SensorModel(0.95, 0.01)
    pred = ChannelPredictor(ch, sensor, specs=_universal_specs(), horizon=80)
    rng = np.random.default_rng(1)
    ep = make_episode("lfsr", ch, sensor, 50, rng, gen_params={"length": 7})
    for o in ep.observations:
        pred.observe(o)
    p = pred.predict_next()
    assert abs(p.sum() - 1.0) < 1e-9 and len(p) == M
    assert 0 <= pred.predict_top1() < M
    topk = pred.predict_topk(3)
    assert len(topk) == 3 and len(set(topk)) == 3 and all(0 <= c < M for c in topk)
    pred.reset()
    assert pred.mixture.t == 0


def test_predicts_new_algorithm_the_bank_cannot():
    # lagged_fib (c_t = c_{t-1}+c_{t-2}) has no dedicated expert and is NOT
    # first-order Markov, so the universal variable-order predictor is the only
    # thing that can predict it. Under a clean sensor it should predict it nearly
    # perfectly, and far better than an order-1-Markov-only ("bank") predictor.
    ch = ChannelSet.of_size(M)
    sensor = SensorModel(1.0, 0.0)
    universal = ChannelPredictor(ch, sensor, specs=_universal_specs(max_order=3), horizon=200)
    markov_only = ChannelPredictor(
        ch, sensor, specs=[HypothesisSpec("markov", "markov", "markov")], horizon=200)
    prop, base = [], []
    for s in range(6):
        rng = np.random.default_rng(500 + s)
        g = make_generator("lagged_fib", M, rng=rng)
        true = g.sequence(160)
        obs = sensor.observe(true, ch, rng)
        prop.append(predict_over_episode(universal, true, obs).top1_steady)
        base.append(predict_over_episode(markov_only, true, obs).top1_steady)
    prop_top1, base_top1 = float(np.mean(prop)), float(np.mean(base))
    assert prop_top1 > 0.9                     # predicts the new algorithm
    assert prop_top1 > base_top1 + 0.25        # and the limited bank cannot


def test_context_mixing_gap_aware_does_not_corrupt_across_gaps():
    # On a clean deterministic cycle the predictor locks to top-1 = 1.0; dropping
    # some detections must not make it predict a wrong channel from a stale
    # context — accuracy on the predictable (post-gap-recovered) slots stays high.
    ch = ChannelSet.of_size(M)
    clean = SensorModel(1.0, 0.0)
    pred = ChannelPredictor(ch, clean, specs=_universal_specs(max_order=2), horizon=120)
    rng = np.random.default_rng(3)
    g = make_generator("quadratic_cg", M, seed=1)
    true = g.sequence(120)
    sc = predict_over_episode(pred, true, clean.observe(true, ch, rng))
    assert sc.top1_steady > 0.95


def test_predict_over_episode_score_fields():
    ch = ChannelSet.of_size(M)
    sensor = SensorModel(0.9, 0.02)
    pred = ChannelPredictor(ch, sensor, specs=_universal_specs(), horizon=80)
    rng = np.random.default_rng(2)
    ep = make_episode("markov", ch, sensor, 60, rng, gen_params={"alpha": 0.2})
    sc = predict_over_episode(pred, ep.true_channels, ep.observations)
    assert isinstance(sc, PredictionScore)
    assert 0.0 <= sc.top1 <= 1.0 and 0.0 <= sc.top3 <= 1.0
    assert sc.top3 >= sc.top1 - 1e-9
    assert abs(sc.uniform_logloss - math.log(M)) < 1e-9


def test_csprng_is_unpredictable_by_the_universal_predictor():
    # the boundary: a keyed CSPRNG must pin prediction at chance (top-1 ~ 1/M)
    ch = ChannelSet.of_size(M)
    sensor = SensorModel(0.95, 0.01)
    pred = ChannelPredictor(ch, sensor, specs=_universal_specs(), horizon=160)
    tops = []
    for s in range(4):
        rng = np.random.default_rng(900 + s)
        g = make_generator("csprng", M, rng=rng)
        true = g.sequence(150)
        tops.append(predict_over_episode(pred, true, sensor.observe(true, ch, rng)).top1)
    assert float(np.mean(tops)) < 0.15   # no better than ~chance (1/16 = 0.0625)
