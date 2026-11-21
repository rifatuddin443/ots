"""Tests for the RL layer: env, agents, learned hopper / jammer, and PSRO."""

from __future__ import annotations

import numpy as np
import pytest

from fhew.allocator import EffectModel
from fhew.channels import ChannelSet
from fhew.generators import make_generator, registered_names
from fhew.sensor import SensorModel
from fhew.rl.agents import (LinearBaseline, LinearSoftmaxPolicy, PPOConfig,
                            cvar_weights, discounted_returns, ppo_update)
from fhew.rl.env import (EngagementEnv, GeneratorHopper, HopperObs, OTSJammerPolicy,
                         ScriptedJammer, run_engagement)
from fhew.rl.hopper import OnlineBanditHopper, RLHopperAgent, train_hopper
from fhew.rl.jammer import RLJammerAgent, train_jammer, train_jammer_robust
from fhew.rl.psro import (EmpiricalGame, Strategy, exploitability, meta_value, run_psro,
                         solve_zero_sum)


@pytest.fixture
def setup():
    ch = ChannelSet.of_size(16)
    sensor = SensorModel(0.95, 0.01)
    effect = EffectModel(ch, tau=1.0)
    env = EngagementEnv(ch, sensor, effect, window=60)
    return ch, sensor, effect, env


# ------------------------------------------------------------------- environment
def test_run_engagement_shapes_and_determinism(setup):
    ch, sensor, effect, env = setup
    # a deterministic (reset-restorable) hopper so two runs at one seed must match
    hop = GeneratorHopper(make_generator("lfsr", 16, length=7, rng=np.random.default_rng(1)))
    jam = ScriptedJammer(ch, effect, 3.0, mode="spot")
    r1 = run_engagement(hop, jam, env, np.random.default_rng(7))
    r2 = run_engagement(hop, jam, env, np.random.default_rng(7))
    assert r1.true_channel.shape == (60,)
    assert r1.effect_on_true.shape == (60,)
    assert np.array_equal(r1.true_channel, r2.true_channel)          # reproducible
    assert np.allclose(r1.effect_on_true, r2.effect_on_true)
    assert 0.0 <= r1.jammer_payoff() <= 1.0
    assert np.isclose(r1.jammer_payoff() + r1.hopper_payoff(), 1.0)   # zero-sum
    assert r1.true_channel.min() >= 0 and r1.true_channel.max() < 16  # in-range


def test_engagement_respects_channel_mask(setup):
    ch = ChannelSet(size=16, blacklist=(0, 1, 2, 3))
    sensor = SensorModel(0.98, 0.0)
    effect = EffectModel(ch, tau=1.0)
    env = EngagementEnv(ch, sensor, effect, window=50)
    hop = RLHopperAgent(ch, rng=np.random.default_rng(0))      # masked to effective set
    jam = ScriptedJammer(ch, effect, 3.0, mode="spot")
    rec = run_engagement(hop, jam, env, np.random.default_rng(0))
    assert set(rec.true_channel.tolist()).isdisjoint({0, 1, 2, 3})


def test_jammer_respects_budget_and_tone_cap(setup):
    ch, sensor, effect, env = setup
    agent = RLJammerAgent(ch, effect, budget=3.0, max_tones=3, rng=np.random.default_rng(0))
    hop = GeneratorHopper(make_generator("markov", 16, rng=np.random.default_rng(1), alpha=0.1))
    # drive a few slots and inspect the power vectors directly
    agent.reset(); hop.reset()
    obs = sensor.observe([5], ch, np.random.default_rng(0))[0]
    for _ in range(5):
        power = agent.act(obs)
        assert power.sum() <= 3.0 + 1e-9
        assert int((power > 1e-12).sum()) <= 3


# ------------------------------------------------------------------------ agents
def test_discounted_returns():
    r = discounted_returns([1.0, 0.0, 2.0], gamma=0.5)
    assert np.allclose(r, [1 + 0.5 * (0 + 0.5 * 2), 0 + 0.5 * 2, 2.0])


def test_cvar_weights_picks_worst_tail():
    returns = np.array([0.9, 0.1, 0.5, 0.2, 0.8])
    w = cvar_weights(returns, alpha=0.4)           # worst 40% -> the two lowest (0.1, 0.2)
    assert w[1] > 0 and w[3] > 0
    assert w[0] == 0 and w[4] == 0
    assert np.isclose(w.mean(), 1.0)               # mean-1 normalised
    assert np.allclose(cvar_weights(returns, 0.0), 1.0)   # disabled


def test_linear_softmax_policy_is_a_distribution():
    pol = LinearSoftmaxPolicy(4, 3, rng=np.random.default_rng(0))
    phi = np.array([0.5, -1.0, 2.0])
    p = pol.probs(phi)
    assert np.isclose(p.sum(), 1.0) and np.all(p >= 0)
    mask = np.array([True, False, True, False])
    pm = pol.probs(phi, mask=mask)
    assert pm[1] < 1e-10 and pm[3] < 1e-10 and np.isclose(pm.sum(), 1.0)


def test_ppo_update_moves_probability_toward_high_advantage_action():
    # Bias-only features, 2 actions; action 1 always earns more than action 0.
    rng = np.random.default_rng(0)
    pol = LinearSoftmaxPolicy(2, 1, rng=rng)
    base = LinearBaseline(1)
    phi = np.array([1.0])
    before = pol.probs(phi)[1]
    for _ in range(8):
        episodes = []
        for _ in range(64):
            a, _ = pol.sample(phi)
            reward = 1.0 if a == 1 else 0.0
            episodes.append({"phi": [phi], "action": [a], "reward": [reward]})
        ppo_update(pol, base, episodes, PPOConfig(lr=0.2, epochs=4, entropy_coef=0.0))
    after = pol.probs(phi)[1]
    assert after > before + 0.2


# --------------------------------------------------------------------- hopper RL
def test_rl_hopper_registered_and_valid_sequence():
    assert "rl_hopper" in registered_names()
    g = make_generator("rl_hopper", 16, rng=np.random.default_rng(1))
    seq = g.sequence(60)
    assert seq.min() >= 0 and seq.max() < 16
    assert len(set(seq.tolist())) >= 4                 # actually hops around


def test_online_bandit_hopper_escapes_a_static_jammer(setup):
    ch, sensor, effect, env = setup
    # a static jammer that always jams channels {0,1,2}; the bandit should learn to avoid them
    jam = ScriptedJammer(ch, effect, 3.0, mode="static", static_channels=(0, 1, 2))
    hop = OnlineBanditHopper(ch, rng=np.random.default_rng(0))
    rec = run_engagement(hop, jam, env, np.random.default_rng(0))
    early = rec.denied[:20].mean()
    late = rec.denied[-20:].mean()
    assert late <= early + 1e-9                        # denial does not grow
    assert late < 0.25                                 # and ends low


def test_rl_hopper_training_lowers_denial_vs_static(setup):
    ch, sensor, effect, env = setup

    # a static jammer on a fixed set: the hopper can learn to avoid those channels
    def opp():
        return ScriptedJammer(ch, effect, 3.0, mode="static", static_channels=(0, 1, 2, 3))
    untrained = RLHopperAgent(ch, rng=np.random.default_rng(0))
    d0 = np.mean([run_engagement(untrained, opp(), env, np.random.default_rng(100 + s)).denied_frac()
                  for s in range(6)])
    agent = train_hopper(opp, ch, env, n_iters=16, episodes_per_iter=10,
                         cfg=PPOConfig(), rng=np.random.default_rng(1)).freeze()
    d1 = np.mean([run_engagement(agent, opp(), env, np.random.default_rng(100 + s)).denied_frac()
                  for s in range(6)])
    assert d1 < d0                                      # learns to dodge the jammed set
    assert d1 < 0.12


# --------------------------------------------------------------------- jammer RL
def test_rl_jammer_learns_peaky_markov(setup):
    ch, sensor, effect, env = setup
    base = make_generator("markov", 16, rng=np.random.default_rng(42), alpha=0.05)

    def opp():
        return GeneratorHopper(base)
    untrained = RLJammerAgent(ch, effect, 3.0, rng=np.random.default_rng(0))
    d0 = np.mean([run_engagement(opp(), untrained, env, np.random.default_rng(100 + s)).denied_frac()
                  for s in range(6)])
    trained = train_jammer(opp, ch, effect, env, n_iters=30, episodes_per_iter=12,
                           cfg=PPOConfig(), rng=np.random.default_rng(1)).freeze()
    d1 = np.mean([run_engagement(opp(), trained, env, np.random.default_rng(100 + s)).denied_frac()
                  for s in range(6)])
    assert d1 > d0 + 0.05                               # clearly learns the structure


def test_cvar_ppo_path_runs(setup):
    ch, sensor, effect, env = setup
    base = make_generator("markov", 16, rng=np.random.default_rng(1), alpha=0.1)

    def opp():
        return GeneratorHopper(base)
    # the cvar_alpha>0 code path (tail-filtering) must run and yield a valid policy
    agent = train_jammer(opp, ch, effect, env, n_iters=6, episodes_per_iter=12,
                         cfg=PPOConfig(cvar_alpha=0.3), rng=np.random.default_rng(0)).freeze()
    rec = run_engagement(opp(), agent, env, np.random.default_rng(0))
    assert 0.0 <= rec.denied_frac() <= 1.0


def test_train_jammer_robust_runs(setup):
    ch, sensor, effect, env = setup
    opps = {f"a{i}": GeneratorHopper(make_generator("markov", 16, alpha=a,
            rng=np.random.default_rng(10 + i))) for i, a in enumerate((0.05, 0.1, 0.2))}
    agent = train_jammer_robust(opps, ch, effect, env, n_iters=5, episodes_per_group=6,
                                worst_k=1, robust=True, rng=np.random.default_rng(0)).freeze()
    for g in opps.values():
        d = run_engagement(GeneratorHopper(g.gen), agent, env, np.random.default_rng(0)).denied_frac()
        assert 0.0 <= d <= 1.0


# ----------------------------------------------------------------------- PSRO
def test_solve_zero_sum_matching_pennies():
    U = np.array([[1.0, 0.0], [0.0, 1.0]])             # jammer wants to match the row
    sh, sj = solve_zero_sum(U, iters=5000)
    assert np.allclose(sh, [0.5, 0.5], atol=0.05)
    assert np.allclose(sj, [0.5, 0.5], atol=0.05)


def test_solve_zero_sum_dominant_column():
    U = np.array([[0.9, 0.1], [0.8, 0.2]])             # column 0 dominates for the jammer
    sh, sj = solve_zero_sum(U, iters=3000)
    assert sj[0] > 0.9                                 # jammer plays the dominant column
    assert sh[1] > 0.9                                 # hopper plays the row that denies least


def test_empirical_game_matrix_grows(setup):
    ch, sensor, effect, env = setup
    game = EmpiricalGame(env, n_seeds=2)
    game.jammers.append(Strategy("spot", ScriptedJammer(ch, effect, 3.0, mode="spot"), "jammer"))
    game.U = np.zeros((0, 1))
    game.add_hopper(Strategy("lfsr", GeneratorHopper(
        make_generator("lfsr", 16, length=7, rng=np.random.default_rng(0))), "hopper"))
    game.add_hopper(Strategy("markov", GeneratorHopper(
        make_generator("markov", 16, rng=np.random.default_rng(0), alpha=0.2)), "hopper"))
    assert game.U.shape == (2, 1)
    game.add_jammer(Strategy("barrage", ScriptedJammer(ch, effect, 3.0, mode="barrage"), "jammer"))
    assert game.U.shape == (2, 2)
    assert np.all((game.U >= 0) & (game.U <= 1))


def test_run_psro_produces_finite_certificate(setup):
    ch, sensor, effect, _ = setup
    br = {"n_iters": 4, "episodes_per_iter": 6, "cfg": PPOConfig()}
    res = run_psro(ch, sensor, effect, budget=3.0, max_tones=4, window=50,
                   n_iters=2, n_seeds=3, br=br, include_unpredictable=False, seed=0)
    assert len(res.exploitability) == 2
    assert all(np.isfinite(x) and x >= 0 for x in res.exploitability)
    assert all(0.0 <= v <= 1.0 for v in res.meta_value)
    # populations grew by one per side per iteration
    assert len(res.hopper_labels) == res.meta["n_init_hoppers"] + 2
    assert len(res.jammer_labels) == res.meta["n_init_jammers"] + 2
    # exploitability should fall from the (exploitable) start toward the equilibrium
    assert res.exploitability[-1] <= res.exploitability[0] + 0.05
