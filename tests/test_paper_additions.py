"""Tests for the baselines and the response set added for the conference paper."""

import numpy as np

from experiments import common as C
from experiments import exp20_conformal_sets as E20
from experiments import exp24_denial as E24
from experiments.exp23_prediction_baselines import PPM, Markov1, _sense_masks
from fhew.channels import ChannelSet
from fhew.conformal import ConformalChannelPredictor
from fhew.predict import ChannelPredictor
from fhew.sensor import Detection, Observation, SensorModel


def _obs(slot, channels, M=16):
    dets = tuple(Detection(slot=slot, channel=c, power=None, bearing=None, emitter_id=0)
                 for c in channels)
    return Observation(slot=slot, detections=dets, n_channels=M)


def test_ppm_learns_an_order_two_rule_a_markov_counter_cannot():
    M = 16
    seq = [1, 3]
    for _ in range(400):
        seq.append((seq[-1] + seq[-2]) % M)          # lagged Fibonacci
    ppm, mk = PPM(M), Markov1(M)
    hits_ppm, hits_mk = [], []
    for t, c in enumerate(seq[:-1]):
        ppm.observe(_obs(t, [c])); mk.observe(_obs(t, [c]))
        if t > 300:
            hits_ppm.append(int(np.argmax(ppm.predict())) == seq[t + 1])
            hits_mk.append(int(np.argmax(mk.predict())) == seq[t + 1])
    assert np.mean(hits_ppm) > 0.95
    assert np.mean(hits_mk) < 0.8


def test_online_baselines_drop_their_context_on_an_ambiguous_slot():
    ppm, mk = PPM(16), Markov1(16)
    for t, c in enumerate([1, 2, 3]):
        ppm.observe(_obs(t, [c])); mk.observe(_obs(t, [c]))
    ppm.observe(_obs(3, [4, 9])); mk.observe(_obs(3, [4, 9]))   # a ghost alongside
    assert ppm.hist == [] and mk.last is None
    assert np.isclose(mk.predict().sum(), 1.0) and np.isclose(ppm.predict().sum(), 1.0)


def test_sense_masks_match_the_sensor_model_rates():
    rng = np.random.default_rng(0)
    true = rng.integers(0, 16, size=(200, 100))
    m = _sense_masks(true, 16, 0.8, 0.02, rng)
    hit = np.take_along_axis(m, true[..., None], axis=-1).mean()
    fa = (m.sum(-1) - np.take_along_axis(m, true[..., None], axis=-1)[..., 0]).mean() / 15
    assert abs(hit - 0.8) < 0.01 and abs(fa - 0.02) < 0.002


def test_denial_is_monotone_in_the_tone_budget_and_cheap_on_an_lfsr():
    ch = ChannelSet.of_size(16)
    sensor = SensorModel(p_d=0.95, p_fa=0.01)
    cpred = ConformalChannelPredictor(
        ChannelPredictor(ch, sensor, specs=E20._pred_specs(16)), alpha=0.1, gamma=0.03)
    ranks, sizes = E24.run_episode(cpred, E20._episode("lfsr", ch, sensor, 100, C.seeds(1)[0]))
    assert ranks.shape == sizes.shape and sizes.min() >= 1
    denied = [float((ranks < np.minimum(sizes, b)).mean()) for b in range(1, 17)]
    assert all(b >= a for a, b in zip(denied, denied[1:]))
    assert denied[0] > 0.9                    # one tone already denies an LFSR
    assert denied[-1] == float((ranks < sizes).mean())
