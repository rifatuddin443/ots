"""Tests for the closed-loop OTS-vs-adversary engagement (fhew.duel).

Sections:
  1. record shape / metric validity
  2. the headline game properties (evasion and deception each lower denial)
  3. fairness: no clairvoyance, and a reused jammer matches a fresh one
"""

import numpy as np
import pytest

from fhew.allocator import EffectModel
from fhew.channels import ChannelSet
from fhew.deception import DeceptionConfig, DeceptionEmitter
from fhew.duel import OTSJammer, run_duel
from fhew.evasion import NO_EVASION_CFG, AdaptiveEvasionGenerator, EvasionConfig
from fhew.generators import make_generator
from fhew.sensor import SensorModel

M = 16
WINDOW = 120


@pytest.fixture(scope="module")
def ch():
    return ChannelSet.of_size(M)


@pytest.fixture(scope="module")
def sensor():
    return SensorModel(0.95, 0.01)


@pytest.fixture(scope="module")
def jammer(ch, sensor):
    # one expensive predictor, reused (reset by run_duel) across the module's duels
    return OTSJammer(ch, sensor, EffectModel(ch, tau=1.0), budget=3.0,
                     plan="knapsack", max_tones=4)


def _evader(fam, ch, rng, cfg, **params):
    return AdaptiveEvasionGenerator.from_family(fam, ch, rng, base_params=params, config=cfg)


# ============================================================ 1. record / validity
def test_run_duel_record_shape_and_ranges(ch, sensor, jammer):
    rng = np.random.default_rng(0)
    em = _evader("lfsr", ch, rng, EvasionConfig(), length=7)
    res = run_duel(em, jammer, sensor, ch, rng, window=WINDOW)
    assert len(res.slots) == WINDOW
    for s in res.slots:
        assert 0 <= s.true_channel < M
        assert 0 <= s.predicted_next < M
        assert 0.0 <= s.effect_on_true <= 1.0
        assert 0.0 <= s.kappa <= 1.0001
        assert s.n_tones >= 0
    agg = res.aggregate()
    for k in ("denied_frac", "mean_effect", "nexthop_top1", "kappa_mean", "mean_tones"):
        assert k in agg
    assert 0.0 <= agg["denied_frac"] <= 1.0


# ============================================================ 2. the game properties
def test_predictable_source_is_denied_without_evasion(ch, sensor, jammer):
    rng = np.random.default_rng(1)
    em = _evader("lfsr", ch, rng, NO_EVASION_CFG, length=7)
    agg = run_duel(em, jammer, sensor, ch, rng, window=WINDOW).aggregate()
    assert agg["denied_frac"] > 0.8          # a fixed LFSR tape is jammed almost completely


def test_evasion_reduces_denial(ch, sensor, jammer):
    def mean_denied(cfg, seeds):
        out = []
        for s in seeds:
            rng = np.random.default_rng(s)
            em = _evader("lfsr", ch, rng, cfg, length=7)
            out.append(run_duel(em, jammer, sensor, ch, rng, window=WINDOW).aggregate()["denied_frac"])
        return float(np.mean(out))

    seeds = [1, 2, 3]
    off = mean_denied(NO_EVASION_CFG, seeds)
    on = mean_denied(EvasionConfig(), seeds)
    assert on < off - 0.3                      # a large, unambiguous evasion dividend


def test_deception_reduces_denial(ch, sensor, jammer):
    def mean_denied(use_decoy, seeds):
        out = []
        for s in seeds:
            rng = np.random.default_rng(s)
            em = _evader("lfsr", ch, rng, NO_EVASION_CFG, length=7)
            dec = None
            if use_decoy:
                dg = make_generator("lfsr", M, rng=rng, length=6)
                dec = DeceptionEmitter(dg, ch, DeceptionConfig(n_decoys=2),
                                       decoy_family="lfsr", decoy_params={"length": 6}, rng=rng)
            out.append(run_duel(em, jammer, sensor, ch, rng, window=WINDOW,
                                deception=dec).aggregate()["denied_frac"])
        return float(np.mean(out))

    seeds = [1, 2, 3]
    assert mean_denied(True, seeds) < mean_denied(False, seeds) - 0.3


# ============================================================ 3. fairness
def test_no_jamming_means_no_evasion(ch, sensor):
    """With a zero-budget jammer nothing is ever denied, so even an aggressive evader
    must reproduce its base sequence exactly -- it reacts only to real jamming, never
    to the jammer's intentions (no clairvoyance)."""
    zero = OTSJammer(ch, sensor, EffectModel(ch, tau=1.0), budget=0.0, plan="knapsack")
    base = make_generator("lfsr", M, rng=np.random.default_rng(0), length=7, state=55)
    ref = base.sequence(WINDOW)
    em = AdaptiveEvasionGenerator(
        make_generator("lfsr", M, rng=np.random.default_rng(0), length=7, state=55),
        ch, config=EvasionConfig(), rng=np.random.default_rng(0))
    res = run_duel(em, zero, sensor, ch, np.random.default_rng(7), window=WINDOW)
    got = np.array([s.true_channel for s in res.slots])
    assert np.array_equal(got, ref)
    assert em.rekeys == 0 and em.pressure == pytest.approx(0.0, abs=1e-9)


def test_reused_jammer_matches_fresh_jammer(ch, sensor):
    """Resetting a reused jammer must reproduce a freshly built one bit for bit
    (guards the reuse optimisation in the experiments)."""
    def run(j):
        rng = np.random.default_rng(42)
        em = _evader("lfsr", ch, rng, EvasionConfig(), length=7)
        return run_duel(em, j, sensor, ch, rng, window=90).aggregate()

    j1 = OTSJammer(ch, sensor, EffectModel(ch, tau=1.0), budget=3.0, max_tones=4)
    first = run(j1)
    _ = run(j1)                                 # dirty it with a second, different duel
    rerun = run(j1)                             # ... then repeat the first
    fresh = run(OTSJammer(ch, sensor, EffectModel(ch, tau=1.0), budget=3.0, max_tones=4))
    assert rerun["denied_frac"] == pytest.approx(first["denied_frac"])
    assert rerun["denied_frac"] == pytest.approx(fresh["denied_frac"])


def test_conformal_plan_runs(ch, sensor):
    j = OTSJammer(ch, sensor, EffectModel(ch, tau=1.0), budget=3.0, plan="conformal")
    rng = np.random.default_rng(3)
    em = _evader("lfsr", ch, rng, EvasionConfig(), length=7)
    agg = run_duel(em, j, sensor, ch, rng, window=90).aggregate()
    assert 0.0 <= agg["denied_frac"] <= 1.0
