"""Sensor model and the reduced-emission identity."""

import numpy as np
import pytest

from fhew import ChannelSet, SensorModel
from fhew.sensor import Detection, Observation


def exact_log_emission(obs, p_d, p_fa):
    det = obs.detection_mask()
    M = obs.n_channels
    out = np.empty(M)
    for s in range(M):
        lp = np.log(p_d if det[s] else 1 - p_d)
        for c in range(M):
            if c == s:
                continue
            lp += np.log(p_fa if det[c] else 1 - p_fa)
        out[s] = lp
    return out


@pytest.mark.parametrize("p_d,p_fa", [(0.9, 0.01), (0.7, 0.05), (0.99, 0.1)])
def test_reduced_emission_equals_exact_up_to_constant(p_d, p_fa):
    M = 8
    sm = SensorModel(p_d, p_fa)
    obs = Observation(0, (Detection(0, 3), Detection(0, 5)), M)
    reduced = sm.emission_loglikelihood(obs)
    exact = exact_log_emission(obs, p_d, p_fa)
    diff = exact - reduced
    # difference must be constant across channels (the common log F_t)
    assert np.std(diff) < 1e-9


def test_emission_prefers_detected_channel():
    sm = SensorModel(0.9, 0.01)
    obs = Observation(0, (Detection(0, 2),), 6)
    le = sm.emission_loglikelihood(obs)
    assert np.argmax(le) == 2


def test_observe_respects_detection_rate_statistically():
    rng = np.random.default_rng(0)
    ch = ChannelSet.of_size(16)
    sm = SensorModel(p_d=0.8, p_fa=0.0)
    true = [3] * 4000
    obs = sm.observe(true, ch, rng)
    detected = np.mean([3 in o.detected_channels for o in obs])
    assert abs(detected - 0.8) < 0.03


def test_false_alarm_rate_statistically():
    rng = np.random.default_rng(1)
    ch = ChannelSet.of_size(16)
    sm = SensorModel(p_d=1.0, p_fa=0.05)
    true = [0] * 4000
    obs = sm.observe(true, ch, rng)
    # channel 7 is never the true channel; its detections are pure false alarms
    fa = np.mean([7 in o.detected_channels for o in obs])
    assert abs(fa - 0.05) < 0.02


def test_invalid_params_rejected():
    with pytest.raises(ValueError):
        SensorModel(p_d=0.0, p_fa=0.0)
    with pytest.raises(ValueError):
        SensorModel(p_d=0.5, p_fa=1.0)
