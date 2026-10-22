"""The hypothesis bank, forward filter, and stopping rules."""

import numpy as np
import pytest

from fhew import ChannelSet, SensorModel
from fhew.config import build_bank, default_bank_specs, in_bank_labels
from fhew.likelihood import ChannelHMMFilter, DeterministicTrajectoryFilter, SensorAwareEmission
from fhew.sim.engine import run_bank
from fhew.sim.episode import make_episode
from fhew.stopping import EProcess, MSPRT

M, T = 16, 100


def _bank(sensor, window=T, stopping=None):
    ch = ChannelSet.of_size(M)
    specs = default_bank_specs(M, lcg_seed_max=512)
    return build_bank(ch, window, sensor, specs, stopping=stopping or {"kind": "msprt", "delta": 0.05}), ch, specs


def test_forward_filter_prefers_true_trajectory():
    # Two candidate trajectories; the emission should up-weight the true one.
    M2 = 4
    traj = np.array([[0, 1, 2, 3, 0], [3, 2, 1, 0, 3]])
    filt = DeterministicTrajectoryFilter(traj, M2)
    sensor = SensorModel(0.9, 0.0)
    from fhew.sensor import Detection, Observation
    true = [0, 1, 2, 3, 0]
    for t, c in enumerate(true):
        obs = Observation(t, (Detection(t, c),), M2)
        filt.step(sensor.emission_loglikelihood(obs))
    assert np.argmax(np.exp(filt.log_w)) == 0


def test_hmm_filter_posterior_is_distribution():
    T_mat = np.full((M, M), 1.0 / M)
    filt = ChannelHMMFilter(T_mat)
    from fhew.sensor import Detection, Observation
    sensor = SensorModel(0.8, 0.02)
    obs = Observation(0, (Detection(0, 5),), M)
    res = filt.step(sensor.emission_loglikelihood(obs))
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
    tgt = in_bank_labels(specs)
    rng = np.random.default_rng(2)
    ep = make_episode(gen, ch, sensor, T, rng, gen_params=kw).set_target(tgt)
    r = run_bank(bank, ep)
    assert r.correct and r.declared
    assert r.tau_stop < 20


def test_bank_routes_csprng_to_unknown_no_hard_false_id():
    sensor = SensorModel(0.85, 0.02)
    bank, ch, specs = _bank(sensor)
    tgt = in_bank_labels(specs)
    hard_ids = 0
    for s in range(12):
        rng = np.random.default_rng(100 + s)
        ep = make_episode("csprng", ch, sensor, T, rng).set_target(tgt)
        r = run_bank(bank, ep)
        # must never claim a deterministic/algebraic break on crypto
        if r.decision_label in {"lfsr", "lcg", "weak_seed"}:
            hard_ids += 1
    assert hard_ids == 0


def test_msprt_threshold_logic():
    rule = MSPRT(delta=0.05)
    log_prior = np.log(np.full(3, 1 / 3))
    # one hypothesis dominates hugely
    cum = np.array([10.0, 0.0, 0.0])
    d = rule.decide(log_prior, cum)
    assert d.stop and d.k_star == 0


def test_eprocess_threshold_logic():
    rule = EProcess(beta=0.05)
    log_prior = np.log(np.full(3, 1 / 3))
    cum = np.array([30.0, 0.0, 0.0])
    d = rule.decide(log_prior, cum)
    assert d.stop and d.k_star == 0
    # no evidence -> no stop
    d2 = rule.decide(log_prior, np.zeros(3))
    assert not d2.stop
