"""The end-to-end learned jammer -- the "PPO / CVaR-PPO jammer" baseline (Section 12).

The OTS pipeline jammer (:class:`fhew.duel.OTSJammer`) is modular: identify, predict
a next-channel distribution, then allocate power with a calibrated knob. The thesis
asks what a *monolithic learned* jammer -- one policy mapping the sensor stream
straight to a power vector -- buys or loses against that pipeline, as the baseline
that shows "what the guarantee is worth". :class:`RLJammerAgent` is that baseline: a
policy-gradient (:func:`fhew.rl.agents.ppo_update`) agent with no identification, no
explicit predictor and no conformal calibration, trained only to maximise realised
denial.

It reads only the sensor observation stream (never the true channel), and it acts
through the shared :class:`~fhew.allocator.EffectModel`, so its tones cross the same
``tau`` the pipeline and the conformal loss use -- a fair comparison on one physics.
The ``cvar_alpha`` knob turns the average-denial objective into a worst-tail
(CVaR-PPO) objective: the robust-jammer variant that optimises the engagements where
denial was *lowest* rather than the mean, which matters against an adversary that is
itself trying to drive denial down.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from fhew.allocator import EffectModel, predictability, tones_from_confidence
from fhew.channels import ChannelSet
from fhew.rl.agents import LinearBaseline, LinearSoftmaxPolicy, PPOConfig, ppo_update
from fhew.rl.env import JammerPolicy
from fhew.sensor import Observation


class _JammerFeatures:
    """Sensor-history feature view for the jammer: decaying occupancy + last sighting.

    ``phi(s) = [ occ (M) | onehot(last_detected) (M) | 1 ]``. ``occ`` is an EWMA of
    where detections have fallen, so a linear policy can learn the emitter's channel
    bias and (through the last-sighting block) a first-order transition structure --
    an implicit predictor, learned end-to-end from the denial reward alone.
    """

    def __init__(self, M: int, decay: float = 0.7):
        self.M = int(M)
        self.decay = float(decay)
        self.reset()

    @property
    def dim(self) -> int:
        return 2 * self.M + 1

    def reset(self) -> None:
        self.occ = np.zeros(self.M, dtype=np.float64)
        self.last_detected = None

    def ingest(self, obs: Observation) -> None:
        self.occ *= self.decay
        for d in obs.detections:
            self.occ[d.channel] += 1.0
            self.last_detected = d.channel

    def phi(self) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float64)
        s = self.occ.sum()
        v[:self.M] = self.occ / s if s > 0 else 0.0
        if self.last_detected is not None:
            v[self.M + self.last_detected] = 1.0
        v[-1] = 1.0
        return v


class RLJammerAgent(JammerPolicy):
    """End-to-end policy-gradient jammer (a :class:`JammerPolicy`).

    The stochastic action is the *primary* tone channel, sampled from the softmax
    policy; the budget then funds that channel and the next highest-probability
    channels up to a confidence-scaled tone count, through the shared
    :class:`~fhew.allocator.EffectModel`. Gradient flows through the sampled primary,
    so the policy learns to place probability mass on the channel the emitter will
    actually use next -- denial without an explicit identify/predict stage.
    """

    def __init__(self, channels: ChannelSet, effect: EffectModel, budget: float, *,
                 rng: np.random.Generator | None = None, max_tones: int = 4,
                 decay: float = 0.7, label: str = "rl_jammer", training: bool = False):
        self.channels = channels
        self.M = channels.size
        self.effect = effect
        self.budget = float(budget)
        self.max_tones = int(max_tones)
        self.rng = rng or np.random.default_rng()
        self.feat = _JammerFeatures(self.M, decay=decay)
        self.policy = LinearSoftmaxPolicy(self.M, self.feat.dim, rng=self.rng)
        self.baseline = LinearBaseline(self.feat.dim)
        self.label = label
        self.training = bool(training)
        self._mask = _mask_of(channels)
        self.reset()

    def reset(self) -> None:
        self.feat.reset()
        self._episode = {"phi": [], "action": [], "reward": []} if self.training else None
        self._pending = 0
        self.last_n_tones = 0

    def act(self, obs: Observation) -> np.ndarray:
        self.feat.ingest(obs)
        phi = self.feat.phi()
        p = self.policy.probs(phi, mask=self._mask)
        a = int(self.rng.choice(self.M, p=p))
        if self.training and self._episode is not None:
            self._episode["phi"].append(phi)
            self._episode["action"].append(a)
            self._pending += 1
        power = self._allocate(p, a)
        return power

    def _allocate(self, p: np.ndarray, primary: int) -> np.ndarray:
        """Fund the primary tone + the next most-probable channels within budget."""
        kappa = predictability(p, self.channels)
        n = tones_from_confidence(kappa, self.max_tones)
        order = [primary] + [int(c) for c in np.argsort(p)[::-1] if int(c) != primary]
        costs = self.effect.costs()
        power = np.zeros(self.M, dtype=np.float64)
        remaining = self.budget
        lit = 0
        for c in order:
            if lit >= n or remaining <= 1e-12 or not self._mask[c]:
                break
            pay = min(costs[c], remaining)
            if pay <= 0:
                break
            power[c] = pay
            remaining -= pay
            lit += 1
        self.last_n_tones = lit
        return power

    def observe_reward(self, reward: float) -> None:
        if self.training and self._episode is not None and self._pending > 0:
            self._episode["reward"].append(float(reward))
            self._pending -= 1

    def drop_pending(self) -> None:
        self._pending = 0

    # ---------------------------------------------------------------- training
    def end_episode(self) -> None:
        if not self.training or self._episode is None:
            return
        n = min(len(self._episode["action"]), len(self._episode["reward"]))
        if n > 0:
            self._batch = getattr(self, "_batch", [])
            self._batch.append({
                "phi": np.asarray(self._episode["phi"][:n], dtype=np.float64),
                "action": np.asarray(self._episode["action"][:n], dtype=np.int64),
                "reward": np.asarray(self._episode["reward"][:n], dtype=np.float64)})
        self._episode = {"phi": [], "action": [], "reward": []}
        self._pending = 0

    def learn(self, cfg: PPOConfig) -> dict:
        batch = getattr(self, "_batch", [])
        info = ppo_update(self.policy, self.baseline, batch, cfg)
        self._batch = []
        return info

    def freeze(self) -> "RLJammerAgent":
        clone = RLJammerAgent(self.channels, self.effect, self.budget, rng=self.rng,
                              max_tones=self.max_tones, label=self.label, training=False)
        clone.policy.load_weights(self.policy.copy_weights())
        clone.feat.decay = self.feat.decay
        return clone


def _mask_of(channels: ChannelSet) -> np.ndarray:
    m = np.zeros(channels.size, dtype=bool)
    m[channels.effective_indices()] = True
    return m


def train_jammer_robust(opponents: dict, channels: ChannelSet, effect: EffectModel, env, *,
                        budget: float = 3.0, max_tones: int = 4, n_iters: int = 40,
                        episodes_per_group: int = 12, worst_k: int = 1, robust: bool = True,
                        cfg: PPOConfig | None = None, rng: np.random.Generator | None = None,
                        seed: int = 0) -> RLJammerAgent:
    """Train one jammer against a *labelled set* of adversaries, optionally robustly.

    ``opponents`` maps a label to a reusable :class:`~fhew.rl.env.HopperPolicy` (reset
    each rollout). Each iteration plays ``episodes_per_group`` engagements against every
    adversary; with ``robust=True`` the PPO update then uses only the ``worst_k``
    adversary groups -- the ones currently denied *least* -- which is a
    distributionally-robust (CVaR-over-adversaries) objective: it maximises the
    worst-case adversary's denial rather than the average. With ``robust=False`` it is
    ordinary PPO over the pooled batch (the average-denial objective).

    Unlike per-episode CVaR, grouping by adversary makes the worst case *structural*
    (which emitter), not a matter of per-episode sensor luck, so the robust lift is
    stable across seeds.
    """
    cfg = cfg or PPOConfig(cvar_alpha=0.0)
    rng = rng or np.random.default_rng(seed)
    agent = RLJammerAgent(channels, effect, budget, rng=rng, max_tones=max_tones,
                          training=True)
    labels = list(opponents)
    from fhew.rl.env import run_engagement
    for _ in range(n_iters):
        group_eps = {}
        for lab in labels:
            agent._batch = []
            for _ in range(episodes_per_group):
                run_engagement(opponents[lab], agent, env, rng)
                agent.end_episode()
            group_eps[lab] = agent._batch
        if robust and len(labels) > worst_k:
            gmean = {lab: float(np.mean([ep["reward"].sum() for ep in eps]))
                     for lab, eps in group_eps.items()}
            keep = sorted(labels, key=lambda l: gmean[l])[:worst_k]
        else:
            keep = labels
        agent._batch = [ep for lab in keep for ep in group_eps[lab]]
        agent.learn(cfg)
    return agent


def train_jammer(opponent_factory: Callable[[], object], channels: ChannelSet,
                 effect: EffectModel, env, *, budget: float = 3.0, max_tones: int = 4,
                 n_iters: int = 30, episodes_per_iter: int = 16,
                 cfg: PPOConfig | None = None, rng: np.random.Generator | None = None,
                 seed: int = 0) -> RLJammerAgent:
    """Train an :class:`RLJammerAgent` against a (resampled) hopper opponent.

    ``cfg.cvar_alpha > 0`` trains the CVaR-PPO robust variant (optimise the
    worst-``alpha`` tail of per-engagement denial). Returns the trained agent.
    """
    cfg = cfg or PPOConfig()
    rng = rng or np.random.default_rng(seed)
    agent = RLJammerAgent(channels, effect, budget, rng=rng, max_tones=max_tones,
                          training=True)
    from fhew.rl.env import run_engagement
    for _ in range(n_iters):
        for _ in range(episodes_per_iter):
            opp = opponent_factory()
            run_engagement(opp, agent, env, rng)
            agent.end_episode()
        agent.learn(cfg)
    return agent
