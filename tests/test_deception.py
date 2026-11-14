"""Tests for counter-reconnaissance deception (fhew.deception).

Sections:
  1. decoy injection mechanics (coherence, preservation, counts, chaff)
  2. make_deception_episode preserves the hidden truth
  3. the counter-ID boundary: H_0 vs no-H_0 under an active decoy
"""

import numpy as np
import pytest

from experiments import common as C
from fhew.baselines import PairwiseSPRT
from fhew.channels import ChannelSet
from fhew.deception import (
    DECOY_ID,
    DeceptionConfig,
    DeceptionEmitter,
    make_deception_episode,
)
from fhew.experts import build_hypothesis
from fhew.generators import make_generator
from fhew.likelihood import SensorAwareEmission
from fhew.sensor import SensorModel
from fhew.sim import make_episode, run_bank, run_sprt

M = 16


def _emitter(n_decoys=1, chaff=0.0, p_present=1.0, seed=3, length=7):
    ch = ChannelSet.of_size(M)
    dg = make_generator("lfsr", M, rng=np.random.default_rng(seed), length=length, state=99)
    cfg = DeceptionConfig(n_decoys=n_decoys, chaff_rate=chaff, p_present=p_present)
    return ch, DeceptionEmitter(dg, ch, cfg, decoy_family="lfsr", decoy_params={"length": length},
                                rng=np.random.default_rng(seed))


# ======================================================= 1. injection mechanics
def test_coherent_decoy_follows_the_decoy_generator():
    ch, dec = _emitter()
    sensor = SensorModel(0.9, 0.0)
    ep = make_episode("csprng", ch, sensor, 40, np.random.default_rng(1))
    _, decoys = dec.deceive_stream(ep.observations, np.random.default_rng(0))
    # the decoy channels must equal an independent run of the same decoy LFSR
    ref = make_generator("lfsr", M, rng=np.random.default_rng(3), length=7, state=99).sequence(40)
    assert [d[0] for d in decoys] == ref.tolist()


def test_real_detections_are_preserved_and_decoys_tagged():
    ch, dec = _emitter()
    sensor = SensorModel(0.8, 0.02)
    ep = make_episode("markov", ch, sensor, 25, np.random.default_rng(2))
    new_obs, _ = dec.deceive_stream(ep.observations, np.random.default_rng(0))
    for o_real, o_new in zip(ep.observations, new_obs):
        real = {(d.channel, d.emitter_id) for d in o_real.detections}
        got = {(d.channel, d.emitter_id) for d in o_new.detections}
        assert real <= got                                  # nothing real dropped
    # every added detection carries the decoy id
    added = [d for o_r, o_n in zip(ep.observations, new_obs)
             for d in o_n.detections if d not in o_r.detections]
    assert added and all(d.emitter_id == DECOY_ID for d in added)


def test_n_decoys_controls_coherent_stream_count():
    ch, dec = _emitter(n_decoys=3, chaff=0.0)
    sensor = SensorModel(1.0, 0.0)
    ep = make_episode("csprng", ch, sensor, 20, np.random.default_rng(4))
    _, decoys = dec.deceive_stream(ep.observations, np.random.default_rng(0))
    assert all(len(d) == 3 for d in decoys)                 # three coherent decoys per slot


def test_chaff_adds_extra_random_decoys():
    ch, dec = _emitter(n_decoys=1, chaff=0.3)
    sensor = SensorModel(1.0, 0.0)
    ep = make_episode("csprng", ch, sensor, 60, np.random.default_rng(5))
    _, decoys = dec.deceive_stream(ep.observations, np.random.default_rng(0))
    sizes = np.array([len(d) for d in decoys])
    assert sizes.mean() > 1.0                                # coherent (1) + chaff on average


def test_p_present_blinks_the_decoy():
    ch, dec = _emitter(n_decoys=1, p_present=0.3)
    sensor = SensorModel(1.0, 0.0)
    ep = make_episode("csprng", ch, sensor, 200, np.random.default_rng(6))
    _, decoys = dec.deceive_stream(ep.observations, np.random.default_rng(0))
    present = np.array([len(d) for d in decoys])
    assert 0.15 < present.mean() < 0.45                      # roughly p_present


def test_config_validation():
    with pytest.raises(ValueError):
        DeceptionConfig(n_decoys=0)
    with pytest.raises(ValueError):
        DeceptionConfig(chaff_rate=1.0)
    with pytest.raises(ValueError):
        DeceptionConfig(p_present=0.0)


# =============================================== 2. episode wrapper preserves truth
def test_make_deception_episode_keeps_truth_and_corrupts_obs():
    ch = ChannelSet.of_size(M)
    sensor = SensorModel(0.85, 0.02)
    specs = C.default_specs(M)
    base = C.episode_for("csprng", ch, sensor, 50, 7,
                         target_labels=C.default_target_labels(specs))
    ep = make_deception_episode(base, "lfsr", decoy_params={"length": 7},
                                config=DeceptionConfig(n_decoys=1))
    assert np.array_equal(ep.true_channels, base.true_channels)
    assert ep.true_label == base.true_label
    assert ep.target_label == base.target_label == "unknown"
    assert ep.gen_params["deception"]["decoy_family"] == "lfsr"
    # the observed streams must differ (detections were planted)
    n_base = sum(len(o.detections) for o in base.observations)
    n_dec = sum(len(o.detections) for o in ep.observations)
    assert n_dec > n_base


# =============================================== 3. the counter-ID boundary
def test_persistent_decoy_fools_the_no_h0_test():
    """A pairwise SPRT (no H_0) is forced to name the decoy family."""
    ch = ChannelSet.of_size(M)
    sensor = SensorModel(0.85, 0.02)
    specs = C.default_specs(M)
    tgt = C.default_target_labels(specs)
    emit = SensorAwareEmission(sensor)
    sprt = PairwiseSPRT([build_hypothesis(s, ch, 80) for s in specs], emit, M)
    false_id = []
    for s in C.seeds(8):
        base = C.episode_for("csprng", ch, sensor, 80, s, target_labels=tgt)
        ep = make_deception_episode(base, "lfsr", decoy_params={"length": 7},
                                    config=DeceptionConfig(n_decoys=1, p_present=1.0))
        false_id.append(run_sprt(sprt, ep).false_id)
    assert np.mean(false_id) >= 0.75                         # no H_0 -> fooled


def test_h0_routes_intermittent_deception_to_unknown():
    """H_0 gives the bank a graceful 'unknown' escape an intermittent decoy cannot beat.

    A low-persistence decoy should be routed to unknown far more often than a
    near-always-on one, which is the whole value of the none-of-the-above class under
    active deception (the active-adversary analogue of exp4).
    """
    ch = ChannelSet.of_size(M)
    sensor = SensorModel(0.85, 0.02)
    specs = C.default_specs(M)
    tgt = C.default_target_labels(specs)

    def unknown_rate(p_present):
        bank = C.make_bank(ch, 90, sensor, specs)
        unk = []
        for s in C.seeds(8):
            base = C.episode_for("csprng", ch, sensor, 90, s, target_labels=tgt)
            ep = make_deception_episode(base, "lfsr", decoy_params={"length": 7},
                                        config=DeceptionConfig(n_decoys=1, p_present=p_present))
            unk.append(run_bank(bank, ep).decision_label == "unknown")
        return float(np.mean(unk))

    assert unknown_rate(0.2) > unknown_rate(1.0)
    assert unknown_rate(0.2) >= 0.5                          # intermittent decoy -> mostly unknown
