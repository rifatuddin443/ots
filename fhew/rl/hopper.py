"""The learned anti-jam frequency hopper -- the "DQN / PPO adversary" (Section 11).

A real frequency-hopping radio has a receiver, so it *knows* when its link is being
denied and (with a spectrum-sensing front end) where the jamming sits. The scripted
:class:`~fhew.evasion.AdaptiveEvasionGenerator` hand-codes a reaction to that; here
the reaction is **learned** from denial feedback, which is the honest form of the
thesis' "learned (DQN, PPO, reactive) adversary". Two learners share one feature
view and one action space (the next channel):

* :class:`RLHopperAgent` -- an offline-trainable policy-gradient hopper
  (:func:`fhew.rl.agents.ppo_update`). It is the best-response oracle the empirical
  game trains against a jammer meta-strategy (:mod:`fhew.rl.psro`).
* :class:`OnlineBanditHopper` -- an EXP3-style online learner that needs *no*
  training phase: it adapts within a single engagement from denial feedback. This
  is the reactive variant, and it backs the registry generator so a learned hopper
  is a first-class truth-side emitter you can draw like any other family.

Both read only what a radio legally has (``HopperObs``: own-hop denial and, in
spectrum mode, the sensed jammed band). Neither reads the jammer's power, plan or
belief, so a learned-hopper result respects information parity exactly as the
prediction results do.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from fhew.channels import ChannelSet
from fhew.generators import Generator, register
from fhew.rl.agents import LinearBaseline, LinearSoftmaxPolicy, PPOConfig, ppo_update
from fhew.rl.env import HopperObs, HopperPolicy


class _HopperFeatures:
    """Shared feature view for a hopper: a decaying jammed-channel map + last hop.

    ``phi(s) = [ h (M) | onehot(last_channel) (M) | 1 ]`` with ``d = 2M + 1``. The
    hot map ``h`` is an EWMA of where jamming energy landed (spectrum mode) or of
    the radio's own denials (denial-only mode); it is the context that lets a
    *linear* policy encode a reactive rule ("steer away from where I was just
    jammed") while the learned weights set how hard to react.
    """

    def __init__(self, M: int, memory: float = 0.6):
        self.M = int(M)
        self.memory = float(memory)
        self.reset()

    @property
    def dim(self) -> int:
        return 2 * self.M + 1

    def reset(self) -> None:
        self.h = np.zeros(self.M, dtype=np.float64)
        self.last_channel: Optional[int] = None

    def ingest(self, obs: HopperObs | None) -> None:
        if obs is None:
            return
        if obs.jammed_mask is not None:
            jm = np.asarray(obs.jammed_mask, dtype=np.float64)
            self.h = (1.0 - self.memory) * self.h + self.memory * jm
        else:  # denial-only: only the channel the radio sat on is informative
            self.h *= (1.0 - self.memory)
            if obs.last_channel is not None:
                self.h[obs.last_channel] = (
                    (1.0 - self.memory) * self.h[obs.last_channel]
                    + self.memory * float(obs.denied))
        self.last_channel = obs.last_channel

    def phi(self) -> np.ndarray:
        v = np.empty(self.dim, dtype=np.float64)
        v[:self.M] = self.h
        v[self.M:2 * self.M] = 0.0
        if self.last_channel is not None:
            v[self.M + self.last_channel] = 1.0
        v[-1] = 1.0
        return v


class RLHopperAgent(HopperPolicy):
    """Offline-trainable policy-gradient anti-jam hopper (a :class:`HopperPolicy`).

    Drive it through :func:`fhew.rl.env.run_engagement`; with ``training=True`` it
    records a trajectory each episode (call :meth:`end_episode` after each rollout,
    then :meth:`learn` over the batch). With ``training=False`` it is a frozen
    policy for evaluation / use as a fixed strategy in the empirical game.
    """

    def __init__(self, channels: ChannelSet, *, rng: np.random.Generator | None = None,
                 memory: float = 0.6, eff_mask: np.ndarray | None = None,
                 label: str = "rl_hopper", training: bool = False):
        self.channels = channels
        self.M = channels.size
        self.rng = rng or np.random.default_rng()
        self.feat = _HopperFeatures(self.M, memory=memory)
        self.policy = LinearSoftmaxPolicy(self.M, self.feat.dim, rng=self.rng)
        self.baseline = LinearBaseline(self.feat.dim)
        self.label = label
        self.training = bool(training)
        # restrict hopping to the effective (non-blacklisted) channel set
        self._mask = (np.asarray(eff_mask, dtype=bool) if eff_mask is not None
                      else _mask_of(channels))
        self._episode: dict | None = None
        self._batch: list[dict] = []
        self.reset()

    # ------------------------------------------------------------------ policy
    def reset(self) -> None:
        self.feat.reset()
        self._episode = {"phi": [], "action": [], "reward": []} if self.training else None
        self._pending = 0

    def act(self, obs: HopperObs | None) -> int:
        self.feat.ingest(obs)
        phi = self.feat.phi()
        a, _logp = self.policy.sample(phi, mask=self._mask)
        if self.training and self._episode is not None:
            self._episode["phi"].append(phi)
            self._episode["action"].append(a)
            self._pending += 1
        self.feat.last_channel = a          # the hop we are about to make
        return a

    def observe_reward(self, reward: float) -> None:
        if self.training and self._episode is not None and self._pending > 0:
            self._episode["reward"].append(float(reward))
            self._pending -= 1

    # ---------------------------------------------------------------- training
    def end_episode(self) -> None:
        """Flush the just-finished episode into the training batch."""
        if not self.training or self._episode is None:
            return
        n = min(len(self._episode["action"]), len(self._episode["reward"]))
        if n > 0:
            self._batch.append({
                "phi": np.asarray(self._episode["phi"][:n], dtype=np.float64),
                "action": np.asarray(self._episode["action"][:n], dtype=np.int64),
                "reward": np.asarray(self._episode["reward"][:n], dtype=np.float64)})
        self._episode = {"phi": [], "action": [], "reward": []}
        self._pending = 0

    def learn(self, cfg: PPOConfig) -> dict:
        """PPO-clip update over the collected batch; clears it."""
        info = ppo_update(self.policy, self.baseline, self._batch, cfg)
        self._batch = []
        return info

    def freeze(self) -> "RLHopperAgent":
        """Return an evaluation copy (same weights, no training buffers)."""
        clone = RLHopperAgent(self.channels, rng=self.rng, label=self.label,
                              eff_mask=self._mask, training=False)
        clone.policy.load_weights(self.policy.copy_weights())
        clone.feat.memory = self.feat.memory
        return clone


class OnlineBanditHopper(HopperPolicy):
    """EXP3-style online anti-jam hopper -- learns within one engagement, no training.

    Keeps adversarial-bandit weights ``w`` over channels (updated from the realised
    denial on the channel it chose) and a decaying hazard map ``h`` from spectrum
    feedback; it hops from ``p ∝ w · exp(-beta·h)`` over the effective channels. The
    bandit term learns which channels are *globally* safe against this jammer; the
    hazard term gives the fast *reactive* deflection off freshly-jammed channels.
    Because it needs no offline phase it is the registry's learned hopper.
    """

    def __init__(self, channels: ChannelSet, *, rng: np.random.Generator | None = None,
                 gamma: float = 0.08, beta: float = 3.0, memory: float = 0.6,
                 eff_mask: np.ndarray | None = None, label: str = "rl_bandit_hopper"):
        self.channels = channels
        self.M = channels.size
        self.rng = rng or np.random.default_rng()
        self.gamma = float(gamma)        # EXP3 exploration / learning rate
        self.beta = float(beta)          # reactive hazard weight
        self.memory = float(memory)
        self._mask = (np.asarray(eff_mask, dtype=bool) if eff_mask is not None
                      else _mask_of(channels))
        self.label = label
        self.reset()

    def reset(self) -> None:
        self.logw = np.zeros(self.M, dtype=np.float64)
        self.h = np.zeros(self.M, dtype=np.float64)
        self._last_a = None
        self._last_p = None

    def _dist(self) -> np.ndarray:
        z = self.logw - self.beta * self.h
        z = np.where(self._mask, z, -1e30)
        z -= z.max()
        w = np.exp(z)
        p = w / w.sum()
        # EXP3 exploration floor over the effective set
        k = int(self._mask.sum())
        p = (1.0 - self.gamma) * p + self.gamma * self._mask / k
        return p / p.sum()

    def act(self, obs: HopperObs | None) -> int:
        if obs is not None and obs.jammed_mask is not None:
            jm = np.asarray(obs.jammed_mask, dtype=np.float64)
            self.h = (1.0 - self.memory) * self.h + self.memory * jm
        elif obs is not None:
            self.h *= (1.0 - self.memory)
            if obs.last_channel is not None:
                self.h[obs.last_channel] = (1.0 - self.memory) * self.h[obs.last_channel] \
                    + self.memory * float(obs.denied)
        p = self._dist()
        a = int(self.rng.choice(self.M, p=p))
        self._last_a, self._last_p = a, p
        return a

    def observe_reward(self, reward: float) -> None:
        # EXP3 on the escape reward in [0,1] for the channel actually chosen.
        if self._last_a is None or self._last_p is None:
            return
        a = self._last_a
        pa = max(self._last_p[a], 1e-6)
        est = reward / pa                         # importance-weighted gain
        self.logw[a] += self.gamma * est / self.M
        self.logw -= self.logw.max()


@register("rl_hopper")
class LearnedHopperGenerator(Generator):
    """A learned anti-jam hopper as a truth-side registry generator.

    Backed by :class:`OnlineBanditHopper`, so it needs no training and adapts
    online -- a genuinely *learned* reactive emitter you can draw with
    ``make_generator("rl_hopper", M)`` and feed through the duel / engagement env.
    Used open-loop (no feedback, e.g. in a fixed-tape experiment) it degrades to its
    broad exploration distribution, which is still a valid hop sequence.

    ``next_channel`` accepts an optional :class:`~fhew.evasion.JammingFeedback` so
    the reactive path in :class:`fhew.rl.env.GeneratorHopper` drives it; called with
    no argument (the plain :class:`Generator` contract) it hops obliviously.
    """

    def __init__(self, n_channels: int, *, seed: int = 0, gamma: float = 0.08,
                 beta: float = 3.0):
        super().__init__(n_channels)
        self._seed = int(seed)
        self._gamma = float(gamma)
        self._beta = float(beta)
        self.reset()

    @classmethod
    def create(cls, n_channels: int, rng: np.random.Generator | None = None,
               seed: int | None = None, gamma: float = 0.08, beta: float = 3.0):
        rng = rng or np.random.default_rng()
        if seed is None:
            seed = int(rng.integers(0, 2 ** 31 - 1))
        return cls(n_channels=n_channels, seed=seed, gamma=gamma, beta=beta)

    def reset(self) -> None:
        ch = ChannelSet.of_size(self.n_channels)
        self._bandit = OnlineBanditHopper(ch, rng=np.random.default_rng(self._seed),
                                          gamma=self._gamma, beta=self._beta)
        self._bandit.reset()

    def next_channel(self, feedback=None) -> int:
        obs = None
        if feedback is not None:
            obs = HopperObs(slot=getattr(feedback, "slot", 0),
                            denied=bool(getattr(feedback, "denied", False)),
                            effect_on_self=float(getattr(feedback, "effect_on_self", 0.0)),
                            last_channel=self._bandit._last_a,
                            jammed_mask=getattr(feedback, "jammed_mask", None))
            self._bandit.observe_reward(1.0 - obs.effect_on_self)
        return int(self._bandit.act(obs))

    def describe(self) -> dict:
        return {"name": self.name, "n_channels": self.n_channels, "learned": True}


def _mask_of(channels: ChannelSet) -> np.ndarray:
    m = np.zeros(channels.size, dtype=bool)
    m[channels.effective_indices()] = True
    return m


def train_hopper(opponent_factory: Callable[[], object], channels: ChannelSet,
                 env, *, n_iters: int = 30, episodes_per_iter: int = 16,
                 cfg: PPOConfig | None = None, rng: np.random.Generator | None = None,
                 seed: int = 0) -> RLHopperAgent:
    """Train an :class:`RLHopperAgent` against a (possibly stochastic) opponent.

    ``opponent_factory`` returns a fresh :class:`~fhew.rl.env.JammerPolicy` each
    call (so a meta-strategy can resample which jammer is faced per episode). The
    agent plays ``episodes_per_iter`` engagements per PPO phase for ``n_iters``
    phases. Returns the trained (still-training-capable) agent.
    """
    cfg = cfg or PPOConfig()
    rng = rng or np.random.default_rng(seed)
    agent = RLHopperAgent(channels, rng=rng, eff_mask=_mask_of(channels), training=True)
    for _ in range(n_iters):
        for _ in range(episodes_per_iter):
            opp = opponent_factory()
            from fhew.rl.env import run_engagement
            run_engagement(agent, opp, env, rng)
            agent.end_episode()
        agent.learn(cfg)
    return agent
