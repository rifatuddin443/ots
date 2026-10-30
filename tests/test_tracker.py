"""Tests for the switch tracker (fhew.tracker) and the control-room session.

The transmitter switching generator mid-engagement is the case the time-anchored
identification bank cannot follow on its own; these check that the tracker
notices the switch, dates it, re-identifies the new generator (including a
counter-based one) and keeps predicting, without false alarms on a stationary
source -- and that the control-room session the dashboard drives respects the
channel-count constraint and produces serialisable records.
"""

import json

import numpy as np
import pytest

from experiments.common import draw_episode_params
from fhew import ChannelSet, SensorModel
from fhew.experts import DeterministicHypothesis
from fhew.generators import make_generator
from fhew.tracker import ExpertLibrary, SwitchTracker, default_protect, tracker_bank_specs

M = 16


@pytest.fixture(scope="module")
def library():
    return ExpertLibrary.build(ChannelSet.of_size(M), 256)


def _drive(tracker, plan, n, sensor, seed):
    """Run ``plan = [(algo, from_slot), ...]`` for ``n`` slots; return truth + steps."""
    rng = np.random.default_rng(seed)
    ch = tracker.channels
    starts = dict((s, a) for a, s in plan)
    gen, truth, steps, hits = None, [], [], []
    for t in range(n):
        if t in starts:
            a = starts[t]
            gen = make_generator(a, ch.size, rng=rng, **draw_episode_params(a, rng))
        c = gen.next_channel()
        if t:
            hits.append(tracker.predict_top1() == c)
        steps.append(tracker.observe(sensor.observe([c], ch, rng, start_slot=t)[0]))
        truth.append(c)
    return truth, steps, hits


def test_register_experts_need_power_of_two_channels():
    assert {s.family for s in tracker_bank_specs(16)} >= {"lfsr", "filtered_lfsr"}
    fams = {s.family for s in tracker_bank_specs(12)}
    assert "lfsr" not in fams and "filtered_lfsr" not in fams and "weak_seed" in fams


def test_protection_grows_as_the_sensor_worsens():
    assert default_protect(SensorModel(1.0, 0.0), M) <= default_protect(SensorModel(0.8, 0.02), M) \
        <= default_protect(SensorModel(0.6, 0.05), M)


def test_segments_share_tables_but_not_state(library):
    tr = SwitchTracker(ChannelSet.of_size(M), SensorModel(1.0, 0.0), library=library)
    a, b = tr._spawn(), tr._spawn()
    ha = [h for h in a.bank.hypotheses if isinstance(h, DeterministicHypothesis)]
    hb = [h for h in b.bank.hypotheses if isinstance(h, DeterministicHypothesis)]
    assert ha and all(x._filter.traj is y._filter.traj for x, y in zip(ha, hb))      # shared table
    assert all(x._filter.log_w is not y._filter.log_w for x, y in zip(ha, hb))       # own weights
    lib = [h for h in library.bank if isinstance(h, DeterministicHypothesis)]
    assert all(x._filter.traj is y._filter.traj for x, y in zip(ha, lib))


def test_switch_to_counter_based_generator_is_detected_and_identified(library):
    """LFSR, then a freshly keyed weak-seed PRNG: only a segment anchored at the
    switch can lock onto the weak seed's counter-0 output."""
    sensor = SensorModel(1.0, 0.0)
    tr = SwitchTracker(ChannelSet.of_size(M), sensor, library=library)
    _, steps, hits = _drive(tr, [("lfsr", 0), ("weak_seed", 50)], 100, sensor, seed=11)
    assert [d for d, _ in tr.switches if d < 50] == []                  # no alarm before the switch
    det = [(d, s) for d, s in tr.switches if d >= 50]
    assert det and det[0][0] - 50 <= 3 and det[0][1] == 50              # caught fast, dated exactly
    end = steps[-1].ident
    assert end.label == "weak_seed" and end.locked and end.start_slot == 50
    assert all(hits[-20:])                                              # predicting every hop again


def test_noisy_switch_between_known_families(library):
    sensor = SensorModel(0.9, 0.01)
    tr = SwitchTracker(ChannelSet.of_size(M), sensor, library=library)
    _, steps, hits = _drive(tr, [("tent", 0), ("lcg", 60)], 120, sensor, seed=5)
    assert steps[59].ident.label == "chaos"
    assert any(d >= 60 for d, _ in tr.switches)
    assert steps[-1].ident.label == "lcg" and steps[-1].ident.locked
    assert np.mean(hits[-20:]) >= 0.8


def test_no_false_switch_or_algebraic_claim_on_a_csprng(library):
    sensor = SensorModel(0.8, 0.02)
    tr = SwitchTracker(ChannelSet.of_size(M), sensor, library=library)
    _, steps, _ = _drive(tr, [("csprng", 0)], 120, sensor, seed=3)
    assert tr.switches == []
    algebraic = {"lfsr", "filtered_lfsr", "lcg", "lcg_full", "weak_seed"}
    assert not any(s.ident.locked and s.ident.label in algebraic for s in steps)


def test_control_session_runs_a_switch_plan():
    from dashboard.control import ControlSession, run_plan
    s = ControlSession(8, 90, p_d=1.0, p_fa=0.0, seed=2)
    rows = run_plan(s, [("lcg", 0), ("henon", 45)])
    assert [r["algo"] for r in rows] == ["lcg", "henon"]
    assert rows[0]["id_correct"] and rows[1]["id_correct"]
    assert rows[1]["detect_delay"] is not None and rows[1]["detect_delay"] <= 5
    assert s.done and len(s.records) == 90
    json.dumps({"info": s.info(), "records": s.records})                # what the server sends
    rec = s.records[-1]
    assert set(rec) >= {"hop", "algo", "truth", "dets", "call", "hit", "next", "id", "post", "switch_detected"}
    assert len(rec["post"]) == len(s.labels)


def test_control_session_refuses_register_algorithms_off_power_of_two():
    from dashboard.control import ControlSession, algorithms_for
    avail = {a["id"]: a["available"] for a in algorithms_for(12)}
    assert not avail["lfsr"] and not avail["filtered_lfsr"] and avail["lcg"]
    s = ControlSession(12, 30, seed=1)
    with pytest.raises(ValueError):
        s.step("lfsr")
    assert s.step("lcg", 3)[-1]["hop"] == 3


@pytest.mark.parametrize("kw", [{"n_channels": 3}, {"n_channels": 65}, {"hops": 5}, {"p_d": 0.1}])
def test_control_session_validates_settings(kw):
    from dashboard.control import ControlSession
    with pytest.raises(ValueError):
        ControlSession(**kw)


# ============================== the recurrence learner (faster re-learning)
def _recurrence_hits(gen_name, sensor, seed, n=40, M=16):
    from fhew.experts import RecurrenceHypothesis
    from fhew.likelihood import SensorAwareEmission
    ch = ChannelSet.of_size(M)
    rng = np.random.default_rng(seed)
    seq = make_generator(gen_name, M, rng=rng).sequence(n)
    h, em, hits = RecurrenceHypothesis("recurrence", ch), SensorAwareEmission(sensor), []
    for t, o in enumerate(sensor.observe(seq, ch, rng)):
        if t:
            hits.append(int(np.argmax(h.predict_next())) == seq[t])
        h.step(em.log_emission(o))
        h.observe_clean(o.detections[0].channel if len(o.detections) == 1 else None)
    return hits


@pytest.mark.parametrize("M", [12, 16])
def test_recurrence_learner_pins_down_a_lagged_generator_in_a_few_hops(M):
    assert all(_recurrence_hits("lagged_fib", SensorModel(1.0, 0.0), seed=1, M=M)[5:])


def test_recurrence_learner_survives_a_noisy_sensor_and_invents_nothing():
    noisy = [np.mean(_recurrence_hits("lagged_fib", SensorModel(0.8, 0.02), seed=s)[20:]) for s in range(4)]
    assert np.mean(noisy) >= 0.7
    rand = [np.mean(_recurrence_hits("csprng", SensorModel(1.0, 0.0), seed=s)) for s in range(4)]
    assert np.mean(rand) <= 0.2                                         # chance is 1/16


def test_tracker_predicts_a_new_lagged_generator_soon_after_the_switch(library):
    sensor = SensorModel(1.0, 0.0)
    tr = SwitchTracker(ChannelSet.of_size(M), sensor, library=library)
    _, _, hits = _drive(tr, [("weak_seed", 0), ("lagged_fib", 40)], 80, sensor, seed=7)
    assert all(hits[39 + 8:])                                           # every hop from 8 after the switch
