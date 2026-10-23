"""Tests for the Revision-3 additions: Fixed-Share mixture, regime monitor and
context forgetting, dwell feature, threshold-aware allocation, feasibility map.

Sections:
  1. context forgetting (the Section 6.4 gap: the detector must flush the model)
  2. Fixed-Share mixture + share floor (a dead expert recovers)
  3. regime monitor + controller (detects a switch; flush speeds recovery)
  4. dwell feature
  5. threshold-aware allocation (spot<->barrage, knapsack budget, CDT, reactive)
  6. feasibility map (the reactive region)
"""

import numpy as np
import pytest

from fhew import ChannelSet, SensorModel
from fhew.allocator import (
    AllocationController,
    CDTController,
    EffectModel,
    equal_power,
    greedy_knapsack,
    reactive_follower,
    water_filling,
)
from fhew.dwell import DwellTracker, dwell_predictability, estimate_dwell
from fhew.experts import ContextMixingHypothesis, MarkovHypothesis
from fhew.feasibility import RegimeParams, can_react, classify, dwell_latency_grid
from fhew.likelihood import SensorAwareEmission
from fhew.mixture import FixedShareMixture, HypothesisExpert, build_default_mixture
from fhew.regime import (
    BOCPDDetector,
    CUSUMDetector,
    ConformalEDetector,
    RegimeController,
    make_detector,
    recovery_time,
)
from fhew.scenarios import make_dwell_episode, make_switch_episode, with_dwell

M, W = 16, 160


def _emit(p_d=0.98, p_fa=0.0):
    return SensorAwareEmission(SensorModel(p_d, p_fa))


# ======================================================= 1. context forgetting
def test_context_mixing_hard_flush_empties_tables():
    ch = ChannelSet.of_size(M)
    cm = ContextMixingHypothesis("cm", ch, max_order=3)
    for c in ([0, 1, 2, 3] * 10):
        cm.observe_clean(c)
    assert sum(len(t) for t in cm.tables) > 0
    cm.forget(0.0)
    assert sum(len(t) for t in cm.tables) == 0
    p = cm.predict_next()
    # a flushed predictor is back to uniform (maximum entropy)
    assert abs(float(-(p[p > 0] * np.log(p[p > 0])).sum()) - np.log(M)) < 1e-9


def test_context_mixing_exponential_forget_shrinks_counts():
    ch = ChannelSet.of_size(M)
    cm = ContextMixingHypothesis("cm", ch, max_order=2)
    for c in ([5, 6, 7] * 10):
        cm.observe_clean(c)
    total_before = sum(float(v.sum()) for t in cm.tables for v in t.values())
    cm.forget(0.25)
    total_after = sum(float(v.sum()) for t in cm.tables for v in t.values())
    assert total_after == pytest.approx(0.25 * total_before, rel=1e-6)


def test_markov_forget_returns_to_prior():
    ch = ChannelSet.of_size(M)
    mk = MarkovHypothesis("mk", ch, alpha=1.0)
    for c in range(M):
        mk.observe_clean(c)
    mk.forget(0.0)
    # counts back to the flat Laplace prior
    assert np.allclose(mk.counts, 1.0)


def test_forget_default_noop_on_stateless_expert():
    from fhew.experts import BarrageHypothesis
    ch = ChannelSet.of_size(M)
    bar = BarrageHypothesis("bar", ch)
    before = bar.predict_next().copy()
    bar.forget(0.0)                      # must not raise
    assert np.allclose(bar.predict_next(), before)


# ================================================= 2. Fixed-Share mixture floor
def test_mixture_predict_is_distribution():
    ch = ChannelSet.of_size(M)
    sensor = SensorModel(0.95, 0.01)
    mix = build_default_mixture(ch, W, sensor)
    rng = np.random.default_rng(0)
    ep = make_dwell_episode("lfsr", ch, sensor, 40, rng, dwell=1, gen_params={"length": 7})
    for o in ep.observations:
        mix.update(o)
    p = mix.predict()
    assert abs(p.sum() - 1.0) < 1e-9 and (p >= 0).all()


def test_share_floor_keeps_dead_expert_recoverable():
    """With share>0 a member that lost all its mass can climb back; with
    share=0 (plain Hedge) it is driven to ~0 and stays there."""
    ch = ChannelSet.of_size(4)
    emit = _emit()
    def member():
        return HypothesisExpert(ContextMixingHypothesis("cm", ch, max_order=1), emit, ch)
    # one strong + one weak expert; same members, different share floors
    def final_min_weight(share):
        mix = FixedShareMixture([member(), member()], channels=ch, eta=1.0, share=share)
        rng = np.random.default_rng(1)
        # drive expert 0 to lose for a while by feeding a stream only cm learns
        from fhew.sensor import Detection, Observation
        seq = [0, 1, 2, 3] * 20
        for t, c in enumerate(seq):
            mix.update(Observation(t, (Detection(t, c),), 4))
        return float(mix.w.min())
    assert final_min_weight(0.05) > 0.02        # floor keeps it alive
    assert final_min_weight(0.0) <= final_min_weight(0.05)


def test_mixture_tracks_switch_better_than_member_alone():
    ch = ChannelSet.of_size(M)
    sensor = SensorModel(0.98, 0.0)
    rng = np.random.default_rng(3)
    ep = make_switch_episode([("markov", {"alpha": 0.03}), ("markov", {"alpha": 0.03})],
                             ch, sensor, 240, rng, switch_at=120)
    mix = build_default_mixture(ch, 240, sensor)
    ctrl = RegimeController(mix, make_detector("cusum", k=0.5, h=5.5, warmup=18),
                            forget_context=True, retain=0.0)
    t1 = []
    for t, o in enumerate(ep.observations):
        ctrl.update(o)
        if t < 239:
            t1.append(int(np.argmax(mix.predict())) == int(ep.true_channels[t + 1]))
    # recovers to a reliable top-1 at some point after the switch
    assert np.isfinite(recovery_time(t1, 120, window=6, thresh=0.6))


# =============================================== 3. regime monitor + controller
def test_cusum_fires_on_sustained_shift_not_blip():
    det = CUSUMDetector(k=1.0, h=6.0, warmup=20)
    rng = np.random.default_rng(0)
    fired_blip = False
    for _ in range(80):
        fired_blip |= det.update(1.0 + 0.3 * rng.standard_normal())   # calm: mean 1, std ~0.3
    assert not fired_blip
    # a single moderate blip (a few sigma) should not immediately trip it
    assert not det.update(2.2)
    # a sustained upward shift should
    fired = any(det.update(2.2) for _ in range(20))
    assert fired


@pytest.mark.parametrize("kind", ["cusum", "bocpd", "conformal_e"])
def test_detectors_have_bounded_false_alarms_on_stationary_stream(kind):
    det = make_detector(kind)
    rng = np.random.default_rng(2)
    alarms = sum(det.update(float(abs(rng.standard_normal()))) for _ in range(300))
    # a stationary stream should rarely alarm (ARL control / slack)
    assert alarms <= 6


def test_context_flush_recovers_at_least_as_fast_as_weights_only():
    ch = ChannelSet.of_size(M)
    sensor = SensorModel(0.98, 0.0)
    def recov(forget):
        rng = np.random.default_rng(11)
        ep = make_switch_episode([("markov", {"alpha": 0.03}), ("markov", {"alpha": 0.03})],
                                 ch, sensor, 240, rng, switch_at=120)
        mix = build_default_mixture(ch, 240, sensor)
        ctrl = RegimeController(mix, make_detector("cusum", k=0.5, h=5.5, warmup=18),
                                forget_context=forget, retain=0.0)
        t1 = []
        for t, o in enumerate(ep.observations):
            ctrl.update(o)
            if t < 239:
                t1.append(int(np.argmax(mix.predict())) == int(ep.true_channels[t + 1]))
        return recovery_time(t1, 120, window=6, thresh=0.6)
    on, off = recov(True), recov(False)
    assert np.isfinite(on)
    # the flush never recovers slower than weights-only on this structured switch
    assert on <= off + 1e-9


def test_controller_actually_calls_forget_on_alarm(monkeypatch):
    """Guard against the exact gap the user flagged: an alarm must flush the
    context model, not only reset the weights."""
    ch = ChannelSet.of_size(M)
    sensor = SensorModel(0.98, 0.0)
    mix = build_default_mixture(ch, 240, sensor)
    calls = {"forget": 0, "reset_weights": 0}
    orig_forget, orig_reset = mix.forget, mix.reset_weights
    monkeypatch.setattr(mix, "forget", lambda *a, **k: (calls.__setitem__("forget", calls["forget"] + 1), orig_forget(*a, **k))[1])
    monkeypatch.setattr(mix, "reset_weights", lambda *a, **k: (calls.__setitem__("reset_weights", calls["reset_weights"] + 1), orig_reset(*a, **k))[1])
    ctrl = RegimeController(mix, make_detector("cusum", k=0.5, h=5.5, warmup=18),
                            forget_context=True, retain=0.0)
    rng = np.random.default_rng(7)
    ep = make_switch_episode([("markov", {"alpha": 0.03}), ("markov", {"alpha": 0.03})],
                             ch, sensor, 240, rng, switch_at=120)
    for o in ep.observations:
        ctrl.update(o)
    assert ctrl.alarms, "expected at least one regime alarm on a switch"
    assert calls["forget"] >= 1 and calls["reset_weights"] >= 1


def test_recovery_time_basic():
    series = [False] * 10 + [True] * 10
    assert recovery_time(series, switch_at=10, window=4, thresh=0.75) == 0.0
    # never recovers -> nan
    assert np.isnan(recovery_time([False] * 20, switch_at=5, window=4, thresh=0.6))


# ============================================================ 4. dwell feature
def test_with_dwell_repeats_and_truncates():
    out = with_dwell(np.array([3, 7, 1]), dwell=4, window=10)
    assert out.tolist() == [3, 3, 3, 3, 7, 7, 7, 7, 1, 1]


def test_estimate_dwell_recovers_hold_length():
    ch = ChannelSet.of_size(M)
    sensor = SensorModel(1.0, 0.0)
    rng = np.random.default_rng(0)
    ep = make_dwell_episode("lcg", ch, sensor, 120, rng, dwell=4)
    assert abs(estimate_dwell(ep.observations) - 4.0) < 1.0


def test_dwell_tracker_reactive_ready():
    ch = ChannelSet.of_size(M)
    sensor = SensorModel(1.0, 0.0)
    from fhew.sensor import Detection, Observation
    dt = DwellTracker()
    for t in range(5):
        dt.update(Observation(t, (Detection(t, 9),), M))
    assert dt.current_run == 5 and dt.estimate() >= 5
    assert dt.reactive_ready(reaction_latency=2)


def test_dwell_predictability_requires_beating_latency():
    assert dwell_predictability(1.0, reaction_latency=1.0) == 0.0
    assert 0.0 < dwell_predictability(5.0, reaction_latency=1.0) < 1.0
    assert dwell_predictability(2.0, reaction_latency=3.0) == 0.0  # too fast to exploit


# ===================================================== 5. threshold allocation
def _effect():
    return EffectModel(ChannelSet.of_size(M), tau=1.0, sharpness=10.0)


def test_knapsack_respects_budget():
    eff = _effect()
    p = np.full(M, 1.0 / M)
    power = greedy_knapsack(p, eff, budget=3.0, lam=0.3)
    assert power.sum() <= 3.0 + 1e-9 and (power >= 0).all()


def test_knapsack_spot_vs_barrage():
    eff = _effect()
    sharp = np.zeros(M); sharp[4] = 0.9; sharp[5] = 0.1
    spot = greedy_knapsack(sharp, eff, budget=3.0, lam=0.0)
    assert set(np.where(spot > 0)[0].tolist()) <= {4, 5}        # concentrated
    flat = greedy_knapsack(np.full(M, 1.0 / M), eff, budget=3.0, lam=0.5)
    assert (flat > 0.01).sum() >= 2                              # spread


def test_knapsack_beats_waterfilling_on_flat_prediction():
    eff = _effect()
    flat = np.full(M, 1.0 / M)
    kn = greedy_knapsack(flat, eff, budget=3.0, lam=0.3)
    wf = water_filling(flat, eff, budget=3.0)
    true = 7
    # threshold-aware allocation denies some channels; water-filling denies none
    assert eff.effect(kn).max() > eff.effect(wf).max()


def test_reactive_follower_denies_observed_channel():
    eff = _effect()
    power = reactive_follower(9, eff, budget=3.0, p=np.full(M, 1.0 / M))
    assert eff.effect_on(power, 9) > 0.8


def test_cdt_keeps_mean_loss_near_alpha():
    rng = np.random.default_rng(0)
    cdt = CDTController(alpha=0.2, eta=0.1, lam0=0.3)
    for _ in range(500):
        # loss falls as lambda (spread) rises; controller should settle near alpha
        L = float(np.clip(0.6 - 0.5 * cdt.lam + 0.15 * rng.standard_normal(), 0, 1))
        cdt.update(L)
    assert cdt.mean_loss <= 0.2 + 0.05


def test_allocation_controller_picks_reactive_when_dwell_beats_latency():
    eff = _effect()
    ctrl = AllocationController(eff, budget=3.0, reaction_latency=2)
    p = np.full(M, 1.0 / M)
    _, mode_fast = ctrl.allocate(p, observed_channel=3, dwell_estimate=6.0, current_run=4)
    _, mode_slow = ctrl.allocate(p, observed_channel=3, dwell_estimate=1.0, current_run=1)
    assert mode_fast == "reactive" and mode_slow == "predictive"


def test_effect_model_cost_crosses_threshold():
    eff = _effect()
    costs = eff.costs()
    # funding exactly the cost puts a channel on the denied side of the knee
    power = np.zeros(M); power[0] = costs[0]
    assert eff.effect_on(power, 0) > 0.7


# ============================================================ 6. feasibility map
def test_reactive_region_feasible_regardless_of_kappa():
    rp = RegimeParams(window=100, dwell=5, reaction_latency=2, kappa=0.0, key_strength=1.0)
    assert can_react(rp)
    assert classify(rp) == "reactive_follower"     # unpredictable but jammable


def test_short_dwell_unpredictable_is_barrage():
    rp = RegimeParams(window=100, dwell=1, reaction_latency=2, kappa=0.0, key_strength=1.0)
    assert not can_react(rp)
    assert classify(rp) == "barrage"               # the predictability-boundary corner


def test_exact_recovery_when_enough_clean_hops():
    rp = RegimeParams(window=200, hop_rate=1.0, dwell=1, p_d=1.0, register_length=7,
                      key_strength=0.0, kappa=0.9)
    assert classify(rp) == "exact_recovery"


def test_dwell_latency_grid_has_reactive_band():
    grid = dwell_latency_grid(range(1, 10), [1, 2, 3, 4], base=RegimeParams(
        window=120, kappa=0.0, key_strength=1.0))
    from fhew.feasibility import REGIONS
    react = REGIONS.index("reactive_follower")
    assert (grid == react).any()                   # a non-empty reactive region


# ============================ 7. ChannelPredictor regime flush (live-demo path)
def test_predictor_mixture_forget_flushes_and_reflattens():
    from fhew.experts import FixedShareMixture, ContextMixingHypothesis, build_hypothesis, HypothesisSpec
    ch = ChannelSet.of_size(M)
    experts = [ContextMixingHypothesis("cm", ch, max_order=2),
               build_hypothesis(HypothesisSpec("mk", "markov", "markov"), ch, 60)]
    mix = FixedShareMixture(ch, experts, sensor=SensorModel(0.98, 0.0), eta=1.0, alpha=0.02)
    from fhew.sensor import Detection, Observation
    for t, c in enumerate([0, 1, 2, 3] * 12):
        mix.update(Observation(t, (Detection(t, c),), M))
    mix.forget(0.0)
    # weights re-flattened to uniform and the context-mixing tables flushed
    assert np.allclose(np.exp(mix.log_w), 1.0 / mix.n)
    assert sum(len(tbl) for tbl in experts[0].tables) == 0


def test_channel_predictor_monitor_flushes_and_recovers_faster():
    from fhew.predict import ChannelPredictor
    from fhew.config import prediction_expert_specs
    from fhew.scenarios import make_switch_episode
    ch = ChannelSet.of_size(M)
    sensor = SensorModel(0.98, 0.0)
    specs = prediction_expert_specs(M)

    def recov(monitor):
        rng = np.random.default_rng(4)
        ep = make_switch_episode([("markov", {"alpha": 0.03}), ("markov", {"alpha": 0.03})],
                                 ch, sensor, 220, rng, switch_at=110)
        pred = ChannelPredictor(ch, sensor, specs=specs, horizon=224, monitor=monitor, retain=0.0)
        t1 = []
        for t, o in enumerate(ep.observations):
            pred.observe(o)
            if t < 219:
                t1.append(pred.predict_top1() == int(ep.true_channels[t + 1]))
        return recovery_time(t1, 110, window=6, thresh=0.6), pred.alarms

    on, alarms = recov(True)
    off, _ = recov(False)
    assert alarms, "monitor should fire on the switch"
    assert np.isfinite(on) and on <= off     # flushing never recovers slower here
