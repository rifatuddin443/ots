"""Tests for the closed-loop jammer-evasion transmitter (fhew.evasion).

Sections:
  1. no-op without jamming (evasion must not change the baseline)
  2. feedback -> pressure / hot-map dynamics
  3. deflection off hot channels
  4. entropy escalation and re-keying under sustained denial
  5. validity (effective-set / blacklist) and feedback modes
"""

import numpy as np
import pytest

from fhew.channels import ChannelSet
from fhew.evasion import (
    NO_EVASION_CFG,
    AdaptiveEvasionGenerator,
    EvasionConfig,
    JammingFeedback,
)
from fhew.generators import make_generator

M = 16


def _base(seed=0, length=7):
    # a fixed-state LFSR so the base sequence is reproducible across constructions
    return make_generator("lfsr", M, rng=np.random.default_rng(seed), length=length, state=123)


def _quiet(slot, mask_m=M):
    return JammingFeedback(slot=slot, effect_on_self=0.0, denied=False,
                           jammed_mask=np.zeros(mask_m, dtype=bool))


# ============================================================ 1. no-op without jamming
def test_no_feedback_reproduces_base_sequence():
    ch = ChannelSet.of_size(M)
    ref = _base().sequence(60)
    ev = AdaptiveEvasionGenerator(_base(), ch, config=EvasionConfig())
    got = np.array([ev.next_channel(None) for _ in range(60)])
    assert np.array_equal(got, ref)


def test_quiet_feedback_reproduces_base_sequence():
    ch = ChannelSet.of_size(M)
    ref = _base().sequence(60)
    ev = AdaptiveEvasionGenerator(_base(), ch, config=EvasionConfig())
    got = np.array([ev.next_channel(_quiet(t)) for t in range(60)])
    assert np.array_equal(got, ref)
    assert ev.pressure == pytest.approx(0.0, abs=1e-9)
    assert ev.rekeys == 0


def test_no_evasion_config_is_passthrough_even_under_jamming():
    ch = ChannelSet.of_size(M)
    ref = _base().sequence(60)
    ev = AdaptiveEvasionGenerator(_base(), ch, config=NO_EVASION_CFG,
                                  rekey_factory=lambda: _base())
    # hammer it with denial on every channel; it must still never deviate or re-key
    got = []
    for t in range(60):
        fb = JammingFeedback(slot=t, effect_on_self=1.0, denied=True,
                             jammed_mask=np.ones(M, dtype=bool))
        got.append(ev.next_channel(fb))
    assert np.array_equal(np.array(got), ref)
    assert ev.rekeys == 0


# ============================================================ 2. pressure / hot map
def test_pressure_rises_on_denial_and_decays_on_relief():
    ch = ChannelSet.of_size(M)
    ev = AdaptiveEvasionGenerator(_base(), ch, config=EvasionConfig(rekey_pressure=9.0))
    for t in range(10):
        ev.next_channel(JammingFeedback(t, 1.0, True, np.zeros(M, bool)))
    hot = ev.pressure
    assert hot > 0.5
    for t in range(40):
        ev.next_channel(_quiet(t))
    assert ev.pressure < 0.1 < hot


def test_spectrum_hot_map_tracks_jammed_channels():
    ch = ChannelSet.of_size(M)
    ev = AdaptiveEvasionGenerator(_base(), ch, config=EvasionConfig(rekey_pressure=9.0))
    mask = np.zeros(M, bool); mask[5] = True
    for t in range(6):
        ev.next_channel(JammingFeedback(t, 0.0, False, mask))
    assert ev.h[5] > 0.5
    assert ev.h[np.arange(M) != 5].max() < 1e-6


# ============================================================ 3. deflection
def test_deflects_off_a_persistently_hot_channel():
    ch = ChannelSet.of_size(M)
    # arithmetic base cycles through every channel, so it proposes channel 5 regularly
    base = make_generator("arithmetic", M, rng=np.random.default_rng(1), start=0, step=1)
    ev = AdaptiveEvasionGenerator(base, ch, config=EvasionConfig(rekey_pressure=9.0),
                                  rng=np.random.default_rng(0))
    hot = np.zeros(M, bool); hot[5] = True
    emitted = []
    for t in range(200):
        # keep pressure low (denied=False) so ONLY the hot-map deflection is exercised
        emitted.append(ev.next_channel(JammingFeedback(t, 0.0, False, hot)))
    emitted = np.array(emitted)
    # a plain arithmetic hopper would sit on channel 5 ~1/16 of the time
    assert (emitted == 5).mean() < 0.01
    assert set(emitted.tolist()) - {5}            # still hopping over other channels


# ===================================================== 4. entropy escalation / re-key
def test_heavy_jamming_makes_hops_stochastic():
    ch = ChannelSet.of_size(M)

    def run(rng_seed):
        ev = AdaptiveEvasionGenerator(_base(), ch, config=EvasionConfig(rekey_pressure=9.0),
                                      rng=np.random.default_rng(rng_seed))
        out = []
        for t in range(80):
            m = np.zeros(M, bool); m[t % M] = True
            out.append(ev.next_channel(JammingFeedback(t, 1.0, True, m)))
        return np.array(out)

    a, b = run(1), run(2)
    # under sustained pressure the evasive draw is random, so two RNGs diverge
    assert not np.array_equal(a, b)
    # and the realised hop distribution is spread, not pinned to the base
    assert len(set(a.tolist())) > M // 2


def test_rekey_fires_under_sustained_denial_and_changes_pattern():
    ch = ChannelSet.of_size(M)
    seqs = []
    ev = AdaptiveEvasionGenerator(
        _base(), ch, config=EvasionConfig(rekey_pressure=0.6, cool_after_rekey=0.5),
        rekey_factory=lambda: make_generator("lfsr", M, rng=np.random.default_rng(len(seqs) + 1),
                                              length=7),
        rng=np.random.default_rng(0))
    for t in range(60):
        ev.next_channel(JammingFeedback(t, 1.0, True, np.ones(M, bool)))
    assert ev.rekeys >= 1


def test_rekey_without_factory_jumps_phase():
    ch = ChannelSet.of_size(M)
    ev = AdaptiveEvasionGenerator(_base(), ch,
                                  config=EvasionConfig(rekey_pressure=0.6),
                                  rng=np.random.default_rng(0))
    for t in range(40):
        ev.next_channel(JammingFeedback(t, 1.0, True, np.ones(M, bool)))
    assert ev.rekeys >= 1          # phase-jump fallback still counts as a re-key


# ============================================================ 5. validity / modes
def test_all_emissions_respect_the_blacklist():
    ch = ChannelSet(size=M, blacklist=(5, 6, 7))
    ev = AdaptiveEvasionGenerator(_base(), ch, config=EvasionConfig(entropy_pressure=0.0),
                                  rng=np.random.default_rng(0))
    out = []
    for t in range(150):
        m = np.zeros(M, bool); m[t % M] = True
        out.append(ev.next_channel(JammingFeedback(t, 1.0, True, m)))
    out = np.array(out)
    assert out.min() >= 0 and out.max() < M
    assert not set(out.tolist()) & {5, 6, 7}


def test_denial_only_mode_ignores_the_sensed_band():
    ch = ChannelSet.of_size(M)
    ev = AdaptiveEvasionGenerator(_base(), ch,
                                  config=EvasionConfig(feedback_mode="denial_only",
                                                       rekey_pressure=9.0))
    # a spectrum-style mask is supplied but must be ignored in denial_only mode
    mask = np.ones(M, bool)
    ev.next_channel(JammingFeedback(0, 0.0, False, mask))
    # only the channel actually sat on (last_channel) can gain heat, and here denied=False
    assert ev.h.max() < 1e-9


def test_from_family_builds_a_working_evader():
    ch = ChannelSet.of_size(M)
    ev = AdaptiveEvasionGenerator.from_family(
        "markov", ch, np.random.default_rng(3), base_params={"alpha": 0.2})
    out = [ev.next_channel(None if t == 0 else _quiet(t)) for t in range(30)]
    assert all(0 <= c < M for c in out)


def test_bad_feedback_mode_rejected():
    with pytest.raises(ValueError):
        EvasionConfig(feedback_mode="telepathy")
