"""Structured sequence coverage: the new generators and the experts that predict them.

Covers the rule-search expert (arithmetic, polynomial, linear / nonlinear
recurrences, lookup tables, hybrids, noisy rules), the full-keyspace seed beam
(weak-seed PRNG, truncated LCG) and the adaptive Berlekamp-Massey expert (long
LFSRs). Each "locks within N hops" test pins the information-theoretic budget
the expert is designed to meet, so a regression in lock speed fails loudly.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fhew import ChannelSet, SensorModel
from fhew.config import STRUCTURED_GENERATORS, prediction_expert_specs
from fhew.experts import (
    AdaptiveBMHypothesis,
    HypothesisSpec,
    RuleSearchHypothesis,
    SeedBeamHypothesis,
    build_hypothesis,
    default_rule_library,
)
from fhew.generators import make_generator, registered_names
from fhew.likelihood import SensorAwareEmission
from fhew.predict import ChannelPredictor, predict_over_episode
from fhew.sim import make_episode

M = 16
CH = ChannelSet.of_size(M)
PERFECT = SensorModel(p_d=1.0, p_fa=0.0)


def _rule_specs(**opts):
    return [HypothesisSpec("rule_search", "structured", "rule_search", options=opts),
            HypothesisSpec("barrage", "none", "barrage")]


def _drive(h, sensor, true_channels, rng):
    """Feed one hypothesis a sensed stream; return its top-1 hit per next hop."""
    em = SensorAwareEmission(sensor)
    obs = sensor.observe(np.asarray(true_channels), CH, rng)
    hits = []
    for t, o in enumerate(obs[:-1]):
        h.step(em.log_emission(o))
        oc = getattr(h, "observe_clean", None)
        if oc is not None:
            oc(o.detections[0].channel if len(o.detections) == 1 else None)
        hits.append(int(np.argmax(h.predict_next()) == true_channels[t + 1]))
    return hits


# ----------------------------------------------------------------- generators
def test_structured_generators_registered_and_in_range():
    for name in STRUCTURED_GENERATORS:
        assert name in registered_names()
        seq = make_generator(name, M, rng=np.random.default_rng(0)).sequence(80)
        assert seq.min() >= 0 and seq.max() < M


def test_polynomial_obeys_its_finite_difference_rule():
    for deg in (1, 2, 3):
        seq = make_generator("polynomial", M, rng=np.random.default_rng(deg), degree=deg).sequence(30)
        w = [((-1) ** (i + 1)) * math.comb(deg + 1, i) for i in range(1, deg + 2)]
        for t in range(deg + 1, len(seq)):
            assert seq[t] == sum(wi * seq[t - i] for i, wi in enumerate(w, start=1)) % M


def test_lookup_table_and_hybrid_sum_obey_the_lag_rule():
    tab = make_generator("lookup_table", M, rng=np.random.default_rng(1), period=11).sequence(60)
    assert all(tab[t] == tab[t - 11] for t in range(11, 60))
    hyb = make_generator("hybrid_sum", M, rng=np.random.default_rng(1), period=9, drift=3).sequence(60)
    assert all(hyb[t] == (hyb[t - 9] + 27) % M for t in range(9, 60))


def test_generators_reset_reproduces_the_sequence():
    for name in STRUCTURED_GENERATORS:
        g = make_generator(name, M, rng=np.random.default_rng(5))
        a = g.sequence(50)
        g.reset()
        assert np.array_equal(a, g.sequence(50)), name


# ----------------------------------------------------------------- rule search
def test_rule_library_size_and_lite_variant():
    heavy = sum(f.K for f in default_rule_library(M))
    lite = sum(f.K for f in default_rule_library(M, max_lag=32, heavy=False))
    assert 100_000 < heavy < 200_000
    assert lite < 15_000


@pytest.mark.parametrize("gen,params,budget", [
    ("arithmetic", {}, 3),                 # two steps pin c_t = c_{t-1} + d among a c + b
    ("polynomial", {"degree": 2}, 5),      # d + 3 channels
    ("polynomial", {"degree": 3}, 6),
    ("lookup_table", {"period": 12}, 13),  # P + 1: a random table needs a full period
])
def test_rule_search_locks_within_the_minimum_hops(gen, params, budget):
    for s in range(3):
        rng = np.random.default_rng(10 + s)
        true = make_generator(gen, M, rng=rng, **params).sequence(60)
        h = RuleSearchHypothesis("rs", CH)
        hits = _drive(h, PERFECT, true, rng)
        # hits[t] scores the prediction made after observing t + 1 channels
        assert all(hits[budget - 1:]), (gen, s, hits[:budget + 4])


@pytest.mark.parametrize("gen", ["linear_rec", "nonlinear_rec", "lagged_fib", "quadratic_cg",
                                 "hybrid_sum"])
def test_rule_search_predicts_structured_families_through_a_lossy_sensor(gen):
    sensor = SensorModel(p_d=0.9, p_fa=0.01)
    pred = ChannelPredictor(CH, sensor, specs=_rule_specs(), horizon=128)
    scores = [predict_over_episode(pred, ep.true_channels, ep.observations).top1_steady
              for ep in (make_episode(gen, CH, sensor, 100, np.random.default_rng(20 + s))
                         for s in range(3))]
    assert np.mean(scores) >= 0.85, scores


def test_rule_search_relocks_after_a_rule_switch():
    pred = ChannelPredictor(CH, PERFECT, specs=_rule_specs(), horizon=128)
    ep = make_episode("hybrid_switch", CH, PERFECT, 120, np.random.default_rng(3),
                      gen_params={"segment": 40})
    pred.reset()
    hits = []
    for t, o in enumerate(ep.observations[:-1]):
        pred.observe(o)
        hits.append(int(pred.predict_top1() == ep.true_channels[t + 1]))
    # every segment is predicted again well before it ends
    for start in (40, 80):
        assert np.mean(hits[start + 15:start + 39]) >= 0.9, hits[start:start + 40]


def test_rule_search_survives_random_jumps():
    pred = ChannelPredictor(CH, PERFECT, specs=_rule_specs(), horizon=128)
    scores = [predict_over_episode(pred, ep.true_channels, ep.observations, steady_window=80).top1_steady
              for ep in (make_episode("noisy_rule", CH, PERFECT, 120, np.random.default_rng(s),
                                      gen_params={"eps": 0.1}) for s in range(3))]
    # ceiling ~ 1 - eps; each jump should cost about one hop, not a re-learn
    assert np.mean(scores) >= 0.78, scores


def test_rule_search_stays_humble_on_a_csprng():
    h = RuleSearchHypothesis("rs", CH)
    rng = np.random.default_rng(4)
    true = make_generator("csprng", M, rng=rng).sequence(100)
    hits = _drive(h, PERFECT, true, rng)
    assert np.mean(hits) < 0.2
    p = h.predict_next()
    assert abs(p.sum() - 1.0) < 1e-9


def test_rule_search_reports_the_rule_it_found():
    true = make_generator("arithmetic", M, rng=np.random.default_rng(0), step=5).sequence(20)
    h = RuleSearchHypothesis("rs", CH)
    _drive(h, PERFECT, true, np.random.default_rng(0))
    fam, params = h.map_rule()
    # _drive stops one channel short: the last prediction is for true[-1]
    assert np.argmax(h.predict_next()) == true[-1]
    assert fam in {f.name for f in h.families} and params


# ------------------------------------------------------------------ seed beam
def test_seed_beam_cracks_a_weak_seed_far_outside_the_recce_range():
    rng = np.random.default_rng(7)
    g = make_generator("weak_seed", M, rng=rng, keyspace=2 ** 16)
    true = g.sequence(40)
    h = SeedBeamHypothesis("beam", CH, family="weak_seed", keyspace=2 ** 16)
    hits = _drive(h, PERFECT, true, rng)
    assert all(hits[6:]), hits                  # log_16(65536) = 4 clean hops + slack
    assert h.n_particles <= 2


def test_seed_beam_cracks_any_lcg_seed():
    rng = np.random.default_rng(8)
    true = make_generator("lcg", M, rng=rng, seed=50_000).sequence(40)
    h = SeedBeamHypothesis("beam", CH, family="lcg")
    hits = _drive(h, SensorModel(p_d=0.9, p_fa=0.01), true, rng)
    assert np.mean(hits[10:]) >= 0.95, hits


# ---------------------------------------------------------------- adaptive BM
def test_adaptive_bm_predicts_a_long_lfsr():
    rng = np.random.default_rng(9)
    true = make_generator("lfsr", M, rng=rng, length=16).sequence(60)
    h = AdaptiveBMHypothesis("bm", CH, max_length=32)
    hits = _drive(h, PERFECT, true, rng)
    assert all(hits[14:]), hits                 # ~ (8/3) * 16 / 4 = 11 hops + slack


def test_adaptive_bm_drops_a_spurious_fit():
    rng = np.random.default_rng(11)
    true = make_generator("csprng", M, rng=rng).sequence(120)
    h = AdaptiveBMHypothesis("bm", CH, max_length=32)
    hits = _drive(h, PERFECT, true, rng)
    assert np.mean(hits) < 0.2


# -------------------------------------------------------------------- wiring
def test_prediction_specs_carry_the_structured_experts():
    labels = {s.label for s in prediction_expert_specs(M)}
    assert {"rule_search", "seed_beam_weak", "seed_beam_lcg", "bm_wide"} <= labels
    assert "bm_wide" not in {s.label for s in prediction_expert_specs(12)}   # BM needs 2**b
    assert "rule_search" not in {s.label for s in prediction_expert_specs(M, structured=False)}
    for s in prediction_expert_specs(M):
        if s.kind in ("rule_search", "seed_beam", "bm_adaptive"):
            assert build_hypothesis(s, CH, 64) is not None
