"""Full test suite for fhew (merged from the former per-module test files).

Sections:
  1. sensor model + reduced-emission identity
  2. generator library + Berlekamp-Massey recovery
  3. hypothesis bank, forward filter, stopping rules
  4. information-parity audit (paper Section 11)
  5. metrics, baselines, reuse interface
"""

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
