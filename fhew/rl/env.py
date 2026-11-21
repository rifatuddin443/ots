"""The closed-loop engagement environment and the two-sided policy interfaces.

This is the shared substrate the RL layer plays on. It is deliberately thinner
than :mod:`fhew.duel` (which hard-wires the OTS predictor against the scripted
:class:`~fhew.evasion.AdaptiveEvasionGenerator`): here *both* sides are arbitrary
policies, so the same loop scores a scripted hopper against the OTS pipeline, a
learned hopper against a learned jammer, or any mixture -- which is exactly what
the empirical game (:mod:`fhew.rl.psro`) needs from its payoff oracle.

One episode is resolved **from history alone**, so neither side sees the other's
current move (the information-parity rule, enforced structurally):

    slot t:
      1. the jam power committed for this slot (planned last slot) lands on the
         emitter's current channel  ->  the realised denial effect;
      2. the sensor observes the emitter's current channel  ->  the jammer ingests
         that (and only that);
      3. the jammer plans next slot's power from the observation history;
      4. the emitter receives *this* slot's jamming feedback (was my hop denied,
         which channels were jammed) and -- without any sight of the power just
         planned -- chooses next slot's channel.

Because the jammer's slot-t action (the power for slot t+1) is only *scored* at
slot t+1, and likewise for the hopper, the environment carries a one-slot reward
lag and books each side's reward to the action that earned it. Learning policies
receive their reward through :meth:`observe_reward`; scripted policies ignore it.

Payoff convention (zero-sum in denial): the jammer's per-slot reward is the
realised effect on the true channel in ``[0, 1]`` (fraction denied); the hopper's
is ``1 - effect`` (escape). Episode payoff is the mean over scored slots.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from fhew.allocator import EffectModel
from fhew.channels import ChannelSet
from fhew.sensor import Observation, SensorModel


# --------------------------------------------------------------------------- views
@dataclass(frozen=True)
class HopperObs:
    """Everything a transmitter radio legally learns about one slot's jamming.

    Mirrors :class:`fhew.evasion.JammingFeedback`: the denial on *my* hop is always
    available (a missing ACK / failed CRC); ``jammed_mask`` -- which channels
    carried jamming energy -- is sensed by the radio's own receiver and is present
    only in ``"spectrum"`` feedback mode. Never carries the jammer's plan, belief,
    or the power about to be placed next slot.
    """

    slot: int
    denied: bool
    effect_on_self: float
    last_channel: Optional[int]
    jammed_mask: Optional[np.ndarray] = None


# --------------------------------------------------------------------- interfaces
class HopperPolicy(abc.ABC):
    """A transmitter: choose the next channel from jamming feedback alone."""

    label: str = "hopper"

    @abc.abstractmethod
    def reset(self) -> None:
        ...

    @abc.abstractmethod
    def act(self, obs: HopperObs | None) -> int:
        """Return the channel to transmit on now (``obs`` is the previous slot's
        feedback, or ``None`` before any jamming has been observed)."""

    def observe_reward(self, reward: float) -> None:
        """Book the reward earned by the most recent :meth:`act` (learners only)."""
        return None


class JammerPolicy(abc.ABC):
    """A jammer: plan next slot's power vector from sensor observations alone."""

    label: str = "jammer"

    @abc.abstractmethod
    def reset(self) -> None:
        ...

    @abc.abstractmethod
    def act(self, obs: Observation) -> np.ndarray:
        """Ingest this slot's sensor observation and return next slot's power
        vector over channels (never the true channel, never the hopper's plan)."""

    def observe_reward(self, reward: float) -> None:
        """Book the reward earned by the most recent :meth:`act` (learners only)."""
        return None


# ------------------------------------------------------------------------- the env
@dataclass
class EngagementEnv:
    """The engagement parameters shared by both policies and the rollout loop.

    ``effect`` (with its g and tau) is read by the jammer *and* by the scorer, so
    the realised denial, any learned jammer's reward, and the pipeline's conformal
    loss all describe the *same* physical effect -- the "one g and tau" rule.
    """

    channels: ChannelSet
    sensor: SensorModel
    effect: EffectModel
    window: int = 120
    deny_threshold: float = 0.5
    feedback_mode: str = "spectrum"      # "spectrum" | "denial_only"
    warmup: int = 5                      # slots excluded from the scored payoff

    def __post_init__(self) -> None:
        if self.feedback_mode not in ("spectrum", "denial_only"):
            raise ValueError("feedback_mode must be 'spectrum' or 'denial_only'")

    @property
    def M(self) -> int:
        return self.channels.size


@dataclass
class EpisodeRecord:
    """Per-slot trace of one engagement (truth fields are for scoring only)."""

    true_channel: np.ndarray             # c_t actually used
    effect_on_true: np.ndarray           # realised jam effect on c_t, in [0, 1]
    denied: np.ndarray                   # effect >= deny_threshold
    n_tones: np.ndarray                  # transmit frequencies lit each slot
    meta: dict = field(default_factory=dict)

    def jammer_payoff(self, warmup: int = 5) -> float:
        """Mean denial effect over scored slots (the jammer's episode reward)."""
        e = self.effect_on_true[warmup:]
        return float(e.mean()) if e.size else float("nan")

    def hopper_payoff(self, warmup: int = 5) -> float:
        return 1.0 - self.jammer_payoff(warmup)

    def denied_frac(self, warmup: int = 5) -> float:
        d = self.denied[warmup:]
        return float(d.mean()) if d.size else float("nan")

    def mean_tones(self, warmup: int = 5) -> float:
        t = self.n_tones[warmup:]
        return float(t.mean()) if t.size else float("nan")


def run_engagement(hopper: HopperPolicy, jammer: JammerPolicy, env: EngagementEnv,
                   rng: np.random.Generator, *, reset: bool = True) -> EpisodeRecord:
    """Play one episode, resolving both sides from history, and return the trace.

    Both policies receive their per-slot reward (with the correct one-slot lag) via
    :meth:`observe_reward`, so a learning agent passed here collects a full training
    trajectory as a side effect; scripted policies simply ignore it.
    """
    if reset:
        hopper.reset()
        jammer.reset()
    M = env.M
    W = env.window
    true_c = np.empty(W, dtype=np.int64)
    eff = np.empty(W, dtype=np.float64)
    den = np.empty(W, dtype=bool)
    ntones = np.empty(W, dtype=np.int64)

    power_cur = np.zeros(M, dtype=np.float64)     # committed for slot 0 (nothing yet)
    c_cur = int(hopper.act(None))                 # c_0: no feedback yet
    # The jammer's slot-t action is scored at slot t+1; hold it until then.
    jammer_pending = False
    hopper_pending = True                         # the c_0 action awaits its reward

    for t in range(W):
        # 1. realised jamming on the current slot (power committed last slot)
        effect = float(env.effect.effect_on(power_cur, c_cur))
        denied = bool(effect >= env.deny_threshold)

        # Book the lagged rewards for the actions that led into THIS slot.
        if jammer_pending:
            jammer.observe_reward(effect)         # jammer wanted effect high
        if hopper_pending:
            hopper.observe_reward(1.0 - effect)   # hopper wanted effect low

        # 2. sense the true channel (the jammer's only legal input)
        obs = env.sensor.observe([c_cur], env.channels, rng, start_slot=t)[0]

        # 3. the jammer plans next slot's power from the observation history
        power_next = np.asarray(jammer.act(obs), dtype=np.float64)
        jammer_pending = True

        # 4. the emitter gets THIS slot's feedback and picks next slot's channel
        jammed_mask = (power_cur > 1e-12) if env.feedback_mode == "spectrum" else None
        fb = HopperObs(slot=t, denied=denied, effect_on_self=effect,
                       last_channel=c_cur, jammed_mask=jammed_mask)
        if t < W - 1:
            c_next = int(hopper.act(fb))
            hopper_pending = True
        else:
            c_next = c_cur
            hopper_pending = False

        true_c[t] = c_cur
        eff[t] = effect
        den[t] = denied
        ntones[t] = int((power_cur > 1e-12).sum())

        power_cur = power_next
        c_cur = c_next

    # The final committed jammer action is never scored (no following slot); drop it.
    if jammer_pending:
        jammer.drop_pending() if hasattr(jammer, "drop_pending") else None

    return EpisodeRecord(true_channel=true_c, effect_on_true=eff, denied=den,
                         n_tones=ntones,
                         meta={"window": W, "feedback_mode": env.feedback_mode})


# ------------------------------------------------------------------------ adapters
class GeneratorHopper(HopperPolicy):
    """Expose a scripted hopper as a :class:`HopperPolicy`.

    Wraps either a plain :class:`~fhew.generators.Generator` (open loop -- ignores
    feedback) or an :class:`~fhew.evasion.AdaptiveEvasionGenerator` (reactive --
    folds the feedback into its evasion state). Both are fixed strategies in the
    empirical game's initial hopper population.
    """

    def __init__(self, gen, *, label: str | None = None):
        self.gen = gen
        self._reactive = hasattr(gen, "next_channel") and _takes_feedback(gen)
        self.label = label or getattr(gen, "name", getattr(gen, "label", "scripted_hopper"))

    def reset(self) -> None:
        self.gen.reset()

    def act(self, obs: HopperObs | None) -> int:
        if self._reactive:
            fb = None
            if obs is not None:
                from fhew.evasion import JammingFeedback
                fb = JammingFeedback(slot=obs.slot, effect_on_self=obs.effect_on_self,
                                     denied=obs.denied, jammed_mask=obs.jammed_mask)
            return int(self.gen.next_channel(fb))
        return int(self.gen.next_channel())


def _takes_feedback(gen) -> bool:
    """True if ``gen.next_channel`` accepts a feedback argument (evasion wrapper)."""
    import inspect

    try:
        sig = inspect.signature(gen.next_channel)
    except (TypeError, ValueError):  # pragma: no cover - builtins
        return False
    return len(sig.parameters) >= 1


class OTSJammerPolicy(JammerPolicy):
    """Expose the full OTS predict-then-allocate pipeline as a :class:`JammerPolicy`.

    Wraps :class:`fhew.duel.OTSJammer` so the hand-built jammer is a fixed strategy
    the empirical game can pit against learned hoppers -- and the yardstick the
    learned jammer (:mod:`fhew.rl.jammer`) is measured against.
    """

    def __init__(self, ots, *, label: str | None = None):
        self.ots = ots
        self.label = label or f"ots_{ots.plan}"
        self._last_obs_channel = None

    def reset(self) -> None:
        self.ots.reset()
        self._last_obs_channel = None

    def act(self, obs: Observation) -> np.ndarray:
        self.ots.observe(obs)
        p = self.ots.predict_next()
        power, _n, _kappa, _set = self.ots.plan_power(
            p, observed_channel=self.ots._clean(obs))
        return power


class ScriptedJammer(JammerPolicy):
    """A fixed, non-learning jammer baseline (spot / barrage / reactive-follow).

    These seed the jammer population and are the dumb baselines the learned jammer
    and the OTS pipeline must beat. ``mode``:

    * ``"spot"``     -- the whole budget on the single most-recently-detected
      channel (a memoryless follower of the last sighting);
    * ``"barrage"``  -- the budget spread equally over ``n_tones`` channels chosen
      by a decaying occupancy estimate;
    * ``"static"``   -- jam a fixed set of channels regardless of observation.
    """

    def __init__(self, channels: ChannelSet, effect: EffectModel, budget: float, *,
                 mode: str = "spot", n_tones: int = 3, decay: float = 0.8,
                 static_channels: tuple[int, ...] | None = None,
                 label: str | None = None):
        if mode not in ("spot", "barrage", "static"):
            raise ValueError("mode must be 'spot', 'barrage' or 'static'")
        self.channels = channels
        self.M = channels.size
        self.effect = effect
        self.budget = float(budget)
        self.mode = mode
        self.n_tones = int(n_tones)
        self.decay = float(decay)
        self.static_channels = static_channels
        self.label = label or f"scripted_{mode}"
        self.reset()

    def reset(self) -> None:
        self.occ = np.zeros(self.M, dtype=np.float64)
        self._last = None

    def act(self, obs: Observation) -> np.ndarray:
        self.occ *= self.decay
        for d in obs.detections:
            self.occ[d.channel] += 1.0
            self._last = d.channel
        power = np.zeros(self.M, dtype=np.float64)
        costs = self.effect.costs()
        if self.mode == "static":
            chans = self.static_channels or tuple(range(min(self.n_tones, self.M)))
        elif self.mode == "spot":
            chans = (int(np.argmax(self.occ)) if self.occ.any()
                     else (self._last if self._last is not None else 0),)
        else:  # barrage over the top-n occupied channels
            chans = tuple(int(c) for c in np.argsort(self.occ)[::-1][:self.n_tones])
        remaining = self.budget
        for c in chans:
            pay = min(costs[c], remaining)
            if pay <= 0:
                break
            power[c] = pay
            remaining -= pay
        return power
