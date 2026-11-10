"""Tests for the conformal next-channel prediction set (exp19) and the
identification converse (exp20).

Sections:
  1. aps_set: monotone size in the coverage level, bounds, degenerate input
  2. ConformalChannelPredictor: ACI calibration moves the level the right way
  3. coverage ~ target on a stream; the set adapts (small on LFSR, large on CSPRNG)
  4. D1 coupling: funding the conformal set denies the next hop at the coverage rate
  5. converse: the measured E[tau] sits at or above the sequential-test bound
"""

import numpy as np
import pytest

from fhew import ChannelSet, SensorModel
from fhew.allocator import EffectModel
from fhew.channels import ChannelSet as CS
from fhew.config import prediction_expert_specs
from fhew.conformal import (
    ConformalChannelPredictor,
    aps_set,
    conformal_over_episode,
    jam_set_from_conformal,
)
from fhew.predict import ChannelPredictor
from experiments import common as C


def _pred(M=16):
    ch = ChannelSet.of_size(M)
    sensor = SensorModel(p_d=0.95, p_fa=0.01)
    specs = prediction_expert_specs(M, lcg_seed_max=256, weak_keyspace=256)
    return ch, sensor, ChannelPredictor(ch, sensor, specs=specs)


# --------------------------------------------------------------- 1. aps_set
def test_aps_set_monotone_and_bounds():
    p = np.array([0.5, 0.25, 0.15, 0.1])
    sizes = [len(aps_set(p, lv)) for lv in (0.0, 0.4, 0.6, 0.8, 1.0)]
    assert sizes == sorted(sizes)              # non-decreasing in level
    assert sizes[0] == 1 and sizes[-1] == 4    # level 0 -> top-1; level 1 -> all
    assert aps_set(p, 0.6) == [0, 1]           # 0.5 + 0.25 >= 0.6
    assert aps_set(p, 0.9, min_size=3)[:3] == [0, 1, 2]


def test_aps_set_handles_degenerate():
    z = np.zeros(8)
    assert len(aps_set(z, 0.9)) >= 1           # falls back to uniform, no crash


# -------------------------------------------- 2. ACI calibration direction
def test_calibrate_moves_level():
    _, _, pred = _pred()
    cp = ConformalChannelPredictor(pred, alpha=0.1, gamma=0.1)
    lv0 = cp.level()
    cp.calibrate(covered=False)                # a miss -> grow the set -> higher level
    assert cp.level() > lv0
    hi = cp.level()
    for _ in range(20):
        cp.calibrate(covered=True)             # comfortable hits -> shrink
    assert cp.level() < hi


# ------------------------------------------ 3. coverage ~ target; adaptivity
@pytest.mark.parametrize("gen", ["lfsr", "markov", "csprng"])
def test_coverage_near_target(gen):
    ch, sensor, pred = _pred()
    cp = ConformalChannelPredictor(pred, alpha=0.1, gamma=0.05)
    covs = []
    for s in C.seeds(6):
        ep = C.episode_for(gen, ch, sensor, 160, s, lcg_keyspace=256)
        covs.append(conformal_over_episode(cp, ep.true_channels, ep.observations).coverage)
    # ACI holds long-run coverage near the 0.90 target on any source (loose band)
    assert 0.80 <= float(np.mean(covs)) <= 1.0


def test_set_size_adapts_to_predictability():
    ch, sensor, pred = _pred()
    cp = ConformalChannelPredictor(pred, alpha=0.1, gamma=0.05)

    def mean_size(gen):
        szs = []
        for s in C.seeds(6):
            ep = C.episode_for(gen, ch, sensor, 160, s, lcg_keyspace=256)
            szs.append(conformal_over_episode(cp, ep.true_channels, ep.observations).mean_set_size)
        return float(np.mean(szs))

    # a pinned-down LFSR needs far fewer tones than an unpredictable CSPRNG
    assert mean_size("lfsr") < mean_size("csprng")
    assert mean_size("csprng") > 4.0


# ----------------------------------------------- 4. D1: set -> jam allocation
def test_jam_set_funds_the_conformal_set():
    ch = CS.of_size(16)
    effect = EffectModel(ch, tau=1.0)
    budget = float(16 * effect.costs().mean())
    power = jam_set_from_conformal([2, 5, 9], effect, budget)
    assert effect.effect_on(power, 2) >= 0.5          # a funded channel is denied
    assert effect.effect_on(power, 5) >= 0.5
    assert power[0] == 0.0                              # an unfunded one is not


def test_jam_hit_tracks_coverage():
    ch, sensor, pred = _pred()
    cp = ConformalChannelPredictor(pred, alpha=0.1, gamma=0.05)
    effect = EffectModel(ch, tau=1.0)
    budget = float(16 * effect.costs().mean())         # funds the whole set
    ep = C.episode_for("lcg", ch, sensor, 160, C.seeds(1)[0], lcg_keyspace=256)
    # coverage of the set and the realised jam-hit rate must agree when the budget
    # is enough to fund the set (the D1 guarantee)
    cp.reset()
    true = ep.true_channels
    cov, hit = [], []
    for t, o in enumerate(ep.observations):
        cp.observe(o)
        if t >= len(ep.observations) - 1:
            break
        S = cp.prediction_set()
        from fhew.conformal import jam_set_from_conformal as _j
        power = _j(S, effect, budget)
        nxt = int(true[t + 1])
        if t >= 5:
            cov.append(int(nxt in S))
            hit.append(int(effect.effect_on(power, nxt) >= 0.5))
        if len(ep.observations[t + 1].detections) == 1:
            cp.calibrate(nxt in S)
    assert np.mean(hit) == pytest.approx(np.mean(cov), abs=0.02)


# ------------------------------------------------------- 5. converse (exp20)
def test_converse_bound_is_a_lower_bound():
    from experiments import exp21_identification_converse as E
    r = E.compute(n_seeds=10, drift_seeds=8, window=160,
                  delta_grid=(0.1, 0.05, 0.01), p_d_grid=(1.0, 0.8, 0.6))
    effs = [row["efficiency"] for row in r["panelA_delta"]["rows"] + r["panelB_pd"]["rows"]
            if row["efficiency"] == row["efficiency"]]
    assert effs
    # measured hops never materially below the information floor (allow slack for
    # the drift estimator's noise)
    assert min(effs) >= 0.75
    assert r["efficiency"]["mean"] >= 1.0
