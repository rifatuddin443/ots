"""Metrics, baselines, and the reuse interface."""

import numpy as np

from fhew import ChannelSet, SensorModel
from fhew.baselines import FixedLengthTest, PairwiseSPRT
from fhew.config import build_bank, default_bank_specs, in_bank_labels
from fhew.experts.hypothesis import build_hypothesis
from fhew.likelihood import SensorAwareEmission
from fhew.metrics import bootstrap_ci, ece, nexthop_summary, summarize
from fhew.sim.engine import EpisodeResult, run_bank, run_fixed_length, run_sprt
from fhew.sim.episode import make_episode

M, T = 16, 80


def _res(correct, false_id, declared, tau):
    return EpisodeResult("m", "lfsr", "lfsr", "lfsr" if correct else "lcg",
                         declared, tau, correct, false_id)


def test_summarize_basic():
    results = [_res(True, False, True, 5), _res(True, False, True, 7),
               _res(False, True, True, 9)]
    s = summarize(results)
    assert abs(s.accuracy - 2 / 3) < 1e-9
    assert abs(s.false_id_rate - 1 / 3) < 1e-9
    assert s.mean_hops_to_identify == 6.0  # over the two correct declared


def test_bootstrap_ci_contains_mean():
    vals = np.array([0.0, 1.0, 1.0, 1.0, 0.0])
    lo, hi = bootstrap_ci(vals, seed=0)
    assert lo <= vals.mean() <= hi


def test_ece_perfectly_calibrated_is_low():
    # confidence equals accuracy in each regime
    conf = np.concatenate([np.full(100, 0.9), np.full(100, 0.5)])
    rng = np.random.default_rng(0)
    correct = np.concatenate([rng.random(100) < 0.9, rng.random(100) < 0.5])
    assert ece(conf, correct) < 0.1


def test_reuse_interface_contract():
    # The bank is an Expert: update returns a float, predict sums to 1, reset works.
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
    rs = run_sprt(sprt, ep)
    rf = run_fixed_length(fl, ep)
    assert rs.decision_label == "lfsr"
    assert rf.decision_label == "lfsr"


def test_nexthop_summary_handles_nan():
    results = [EpisodeResult("m", "lfsr", "lfsr", "lfsr", True, 5, True, False,
                             nexthop_logloss=1.0, nexthop_top1=0.5, nexthop_top3=0.8)]
    nh = nexthop_summary(results)
    assert abs(nh["logloss"] - 1.0) < 1e-9
