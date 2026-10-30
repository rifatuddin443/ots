"""Track a hopper that switches generator mid-engagement: identify it and predict it.

The identification bank (:class:`~fhew.experts.HypothesisBank`) assumes one
generator for the whole engagement, and its deterministic experts are anchored
in time: candidate ``j`` predicts channel ``traj[j, t]`` at slot ``t`` counted
from the start of the engagement. That is exactly right for one generator, but
it cannot follow an emitter that *changes its generator mid-engagement*: a
freshly keyed weak-seed PRNG or truncated LCG started at slot 137 emits its own
counter-0 output there, which no candidate anchored at slot 0 produces. Resetting
the weights or flushing the context (``forget``) does not help either, because
the alarm arrives a few slots after the switch and the anchor would still be off.

:class:`SwitchTracker` solves this with Bayesian online change-point detection
(Adams & MacKay, 2007) whose run-length hypotheses each carry their own
*anchored segment model*: a fresh copy of the identification bank plus the
zero-shot universal predictors, started at the slot the segment assumes the
current generator began. Every slot a new segment is born (the "it just
switched" hypothesis, prior mass ``hazard``) and every segment is scored by its
sensor-aware predictive likelihood of the observation. After a real switch the
old segment's confident predictions fail while the segment born at the switch
slot locks onto the new generator -- including counter-based ones -- so the
run-length posterior moves to it within a few hops. That posterior gives

    * the **switch detector**: a switch is declared once most of the posterior
      mass sits on segments younger than the current regime, and the most
      probable of them dates it;
    * the **identifier**: the family posterior marginalised over segments,
      ``P(family) = sum_seg P(seg) P(family | seg)``. Marginalising (rather
      than reading one segment's MSPRT stop) charges every newborn segment its
      switch prior, so a segment that fits a few noisy hops by luck cannot
      produce a false lock;
    * the **next-channel predictor**: the run-length-weighted mixture of every
      segment's forecast.

Within one segment the source is stationary, so the segment combines its bank
with the universal experts by a (near-Bayesian) Fixed-Share rule, as in
:class:`~fhew.experts.FixedShareMixture`; a generator the bank has no expert for
is still predicted by the universal context-mixing learner.

Cost is bounded by pruning: the newest ``protect`` segments are always kept (so
the segment born at a switch survives until it has had time to lock -- longer
for a worse sensor) plus the ``max_segments`` most probable older ones.
Segments share the read-only trajectory tables of a pristine
:class:`ExpertLibrary`, so a birth is a cheap copy, not a rebuild. Like every
expert here, the tracker consumes only :class:`~fhew.sensor.Observation`
objects (information parity).
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp

from fhew.channels import ChannelSet
from fhew.config import identification_bank_specs
from fhew.experts import (
    DeterministicHypothesis,
    H0Hypothesis,
    Hypothesis,
    HypothesisBank,
    HypothesisSpec,
    MixtureHMMHypothesis,
    build_hypothesis,
)
from fhew.likelihood import SensorAwareEmission
from fhew.sensor import Observation, SensorModel
from fhew.stopping import make_stopping_rule

#: expert families whose trajectories need a power-of-two channel count
POWER_OF_TWO_FAMILIES = frozenset({"lfsr", "filtered_lfsr"})


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def tracker_bank_specs(n_channels: int) -> list[HypothesisSpec]:
    """The identification bank (one expert per family) for ``n_channels``.

    The register-based experts expand ``log2 M`` bits per hop, so they exist
    only for a power-of-two channel count; every other family works for any M.
    """
    specs = identification_bank_specs(n_channels)
    if not is_power_of_two(n_channels):
        specs = [s for s in specs if s.family not in POWER_OF_TWO_FAMILIES]
    return specs


def universal_specs(max_order: int = 4) -> list[HypothesisSpec]:
    """The zero-shot experts that predict generators the bank has no expert for.

    Context mixing (any in-context regularity), the phase-space predictor
    (unseen low-dimensional dynamics) and the modular linear-recurrence learner
    (affine order-<=2 rules, which it pins down in a few hops where context
    mixing must first see every context). None of them names a generator, so
    identification is unchanged; they make the next-hop prediction re-converge
    faster after a switch.
    """
    return [
        HypothesisSpec("context_mixing", "markov", "context_mixing",
                       options={"max_order": max_order}),
        HypothesisSpec("phase_space", "logistic", "phase_space"),
        HypothesisSpec("recurrence", "lagged_fib", "recurrence"),
    ]


def default_protect(sensor: SensorModel, n_channels: int) -> int:
    """How many newborn segments to keep regardless of their weight.

    A lock needs a handful of *clean* hops (one detection, on the true channel),
    which arrive at rate ``P_d (1 - P_fa)^(M-1)``; the segment born at a switch
    must survive that long, so a worse sensor protects more newborns.
    """
    clean = sensor.p_d * (1.0 - sensor.p_fa) ** (n_channels - 1)
    return int(np.clip(math.ceil(9.0 / max(clean, 1e-3)), 10, 32))


@dataclass
class ExpertLibrary:
    """Pristine experts for one channel count, shared by every tracker built on it.

    Building the deterministic experts' trajectory tables is the expensive part
    (seconds); the tracker only ever deep-copies these objects, sharing their
    read-only tables, so one library serves any number of trackers whose
    engagement fits in ``horizon`` slots.
    """

    channels: ChannelSet
    horizon: int
    specs: list[HypothesisSpec]
    bank: list[Hypothesis]
    universal: list[Hypothesis]

    @classmethod
    def build(cls, channels: ChannelSet, horizon: int, *,
              specs: list[HypothesisSpec] | None = None,
              universal_order: int = 4) -> "ExpertLibrary":
        specs = specs if specs is not None else tracker_bank_specs(channels.size)
        bank = [build_hypothesis(s, channels, horizon) for s in specs]
        universal = [build_hypothesis(s, channels, horizon) for s in universal_specs(universal_order)]
        return cls(channels, int(horizon), list(specs), bank, universal)


class _Segment:
    """One run-length hypothesis: "the current generator started at ``birth``".

    Holds an anchored identification bank and the universal experts, combined by
    Fixed-Share on their sensor-aware predictive log-likelihoods.
    """

    def __init__(self, bank: HypothesisBank, universal: list, *, eta: float, alpha: float):
        self.bank = bank
        self.universal = universal
        self.labels = ["bank"] + [h.label for h in universal]
        n = len(self.labels)
        self.eta, self.alpha = float(eta), float(alpha)
        self.log_w = np.full(n, -np.log(n))
        self._preds = np.tile(bank.channels.uniform(), (n, 1))
        self.birth = 0
        self.log_mass = 0.0

    def update(self, obs: Observation, log_e: np.ndarray, clean: int | None) -> float:
        """Consume one slot; return the segment's predictive log-likelihood of it."""
        lls = np.empty(len(self.labels))
        lls[0] = self.bank.update(obs)
        self._preds[0] = self.bank.predict()
        for k, h in enumerate(self.universal, start=1):
            lls[k] = h.step(log_e).loglik
            h.observe_clean(clean)
            self._preds[k] = h.predict_next()
        loglik = float(logsumexp(self.log_w + lls))
        a = self.log_w + self.eta * lls
        w = np.exp(a - logsumexp(a))
        w = (1.0 - self.alpha) * w + self.alpha / w.size
        self.log_w = np.log(w)
        return loglik

    def predict(self) -> np.ndarray:
        p = np.exp(self.log_w) @ self._preds
        s = p.sum()
        return p / s if s > 0 else self.bank.channels.uniform()

    def weights(self) -> dict[str, float]:
        return {lab: float(w) for lab, w in zip(self.labels, np.exp(self.log_w))}

    @property
    def age(self) -> int:
        return self.bank.t


@dataclass
class Identification:
    """What the tracker currently believes the emitter is running."""

    label: str                      # leading bank label ('H0' = none of the known families)
    p: float                        # its posterior, marginalised over segments
    locked: bool                    # reached the lock level (and has not dropped below unlock)
    lock_slot: int | None           # 0-based slot this regime's lock on `label` was reached
    start_slot: int                 # estimated 0-based slot the current generator started
    posterior: np.ndarray           # marginal family posterior aligned with ``labels``
    labels: list[str]
    carrier: str                    # expert carrying the leading segment's forecast
    carrier_weights: dict = field(default_factory=dict)


@dataclass
class TrackerStep:
    """One slot's tracker output (``slot`` is 0-based)."""

    slot: int
    loglik: float                   # predictive log-likelihood of this slot's observation
    forecast: np.ndarray            # P(next channel)
    ident: Identification
    switch_detected: bool           # a switch was declared at this slot
    p_switch: float                 # posterior mass on "switched since the current start"
    n_segments: int


class SwitchTracker:
    """Identify and predict a frequency hopper that may switch generator at any time.

    Parameters
    ----------
    channels, sensor:
        The channel set and the sensor model (``P_d`` / ``P_fa``).
    horizon:
        Longest run a segment may need (the engagement length): the deterministic
        experts precompute trajectories this long. Taken from ``library`` if given.
    library:
        Prebuilt :class:`ExpertLibrary` to share (else one is built).
    hazard:
        Prior probability per slot that the generator switches (``1 / expected
        regime length``). Smaller is steadier but needs a little more evidence.
    max_segments, protect:
        Keep the ``protect`` newest segments plus the ``max_segments`` most
        probable older ones. ``protect=None`` scales it with the sensor
        (:func:`default_protect`).
    eta, alpha:
        Fixed-Share learning rate and share floor inside each segment. Within a
        segment the source is stationary, so the floor is tiny: a larger one
        would let a dead segment limp on with its universal learner and hide the
        switch.
    lock_level, unlock_level, confirm:
        The identification locks once the marginal family posterior has stayed
        at or above ``lock_level`` for ``confirm`` consecutive slots, and an
        existing lock is released below ``unlock_level`` (hysteresis, so a burst
        of sensor noise does not make it flicker). Confirmation matters most
        just after a switch to an unpredictable source: one enumerated candidate
        can match the first few hops exactly by luck, and the next hop refutes
        it with probability ``1 - 1/M``.
    switch_level:
        Posterior mass on "a switch happened after the current start" at which a
        switch is declared.
    min_takeover:
        Only segments born at least this many slots after the current start count
        as evidence of a new switch (younger ones of the same regime refine it).
    merge:
        A takeover within this many slots of the last declared switch re-dates
        that switch instead of declaring a new one: just after a switch a
        slightly younger segment, whose online learners started cleaner, often
        overtakes the first one to win.
    """

    def __init__(self, channels: ChannelSet, sensor: SensorModel, *, horizon: int | None = None,
                 library: ExpertLibrary | None = None, specs: list[HypothesisSpec] | None = None,
                 hazard: float = 1 / 150, max_segments: int = 6, protect: int | None = None,
                 eta: float = 1.0, alpha: float = 1e-4, lock_level: float = 0.95,
                 unlock_level: float = 0.8, confirm: int = 2, switch_level: float = 0.9, min_takeover: int = 4, merge: int = 20,
                 universal_order: int = 4):
        if not 0.0 < hazard < 1.0:
            raise ValueError("hazard must be in (0, 1)")
        if library is None:
            if horizon is None:
                raise ValueError("pass a horizon or a prebuilt library")
            library = ExpertLibrary.build(channels, horizon, specs=specs,
                                          universal_order=universal_order)
        elif library.channels.size != channels.size:
            raise ValueError("library was built for a different channel count")
        self.channels = channels
        self.M = channels.size
        self.sensor = sensor
        self.library = library
        self.horizon = library.horizon
        self.log_h = float(np.log(hazard))
        self.log_1mh = float(np.log1p(-hazard))
        self.max_segments = int(max_segments)
        self.protect = int(protect) if protect is not None else default_protect(sensor, self.M)
        self.lock_level = float(lock_level)
        self.unlock_level = min(float(unlock_level), self.lock_level)
        self.confirm = max(1, int(confirm))
        self.switch_level = float(switch_level)
        self.min_takeover = int(min_takeover)
        self.merge = int(merge)
        self.emission = SensorAwareEmission(sensor)
        # Same assembly as fhew.config.build_bank, around the library's pristine experts.
        bank = HypothesisBank(channels, H0Hypothesis(channels, mode="uniform", floor="uniform"),
                              library.bank, emission=self.emission,
                              stopping=make_stopping_rule(kind="msprt", delta=1.0 - lock_level),
                              h0_floor=0.15)
        self._template = _Segment(bank, library.universal, eta=eta, alpha=alpha)
        self._memo = _readonly_memo(library.bank + library.universal)
        self.labels = list(bank.labels)
        self.reset()

    # ------------------------------------------------------------------ driving
    def reset(self) -> None:
        self.segments: list[_Segment] = []
        self.t = 0
        self._start = 0
        self._lock: tuple[str, int] | None = None      # (label, slot) of this regime's lock
        self._streak: tuple[str, int] = ("", 0)        # label and slots it has cleared lock_level
        self.switches: list[tuple[int, int]] = []     # (slot declared, estimated start slot)
        self._forecast = self.channels.uniform()

    def _spawn(self) -> _Segment:
        seg = copy.deepcopy(self._template, dict(self._memo))
        seg.birth = self.t
        return seg

    def observe(self, obs: Observation) -> TrackerStep:
        """Consume one slot's observation and return the updated belief."""
        log_e = self.emission.log_emission(obs)
        clean = obs.detections[0].channel if len(obs.detections) == 1 else None
        # change-point prior: a newborn segment takes `hazard` of the mass, the rest grow
        new = self._spawn()
        if self.segments:
            total = float(logsumexp([s.log_mass for s in self.segments]))
            for s in self.segments:
                s.log_mass += self.log_1mh
            new.log_mass = total + self.log_h
        self.segments.append(new)
        prior = np.array([s.log_mass for s in self.segments])
        post = prior + np.array([s.update(obs, log_e, clean) for s in self.segments])
        loglik = float(logsumexp(post) - logsumexp(prior))
        post -= logsumexp(post)
        for s, m in zip(self.segments, post):
            s.log_mass = float(m)
        self._prune()
        self.t += 1

        w = self._weights()
        self._forecast = w @ np.stack([s.predict() for s in self.segments])
        self._forecast /= self._forecast.sum()
        switched, p_switch = self._update_start(w)
        return TrackerStep(slot=self.t - 1, loglik=loglik, forecast=self._forecast.copy(),
                           ident=self._identify(w), switch_detected=switched,
                           p_switch=p_switch, n_segments=len(self.segments))

    def _weights(self) -> np.ndarray:
        m = np.array([s.log_mass for s in self.segments])
        w = np.exp(m - m.max())
        return w / w.sum()

    def _prune(self) -> None:
        if len(self.segments) <= self.protect + self.max_segments:
            return
        young = self.segments[-self.protect:]
        old = sorted(self.segments[:-self.protect], key=lambda s: s.log_mass,
                     reverse=True)[: self.max_segments]
        self.segments = sorted(old, key=lambda s: s.birth) + young

    def _update_start(self, w: np.ndarray) -> tuple[bool, float]:
        """Declare a switch, or refine the current regime's start estimate."""
        later = [k for k, s in enumerate(self.segments)
                 if s.birth >= self._start + self.min_takeover]
        p_switch = float(w[later].sum()) if later else 0.0
        if p_switch >= self.switch_level:
            k = max(later, key=lambda i: w[i])
            self._start = self.segments[k].birth
            self._lock, self._streak = None, ("", 0)
            if self.switches and self.t - 1 - self.switches[-1][0] <= self.merge:
                self.switches[-1] = (self.switches[-1][0], self._start)   # re-date it
                return False, p_switch
            self.switches.append((self.t - 1, self._start))
            return True, p_switch
        k = int(np.argmax(w))
        if w[k] >= 0.5 and abs(self.segments[k].birth - self._start) < self.min_takeover:
            self._start = self.segments[k].birth
        return False, p_switch

    # --------------------------------------------------------------- outputs
    def family_posterior(self) -> np.ndarray:
        """``P(family)`` marginalised over the run-length posterior (aligned with ``labels``)."""
        w = self._weights()
        return w @ np.stack([s.bank.posterior() for s in self.segments])

    def _identify(self, w: np.ndarray) -> Identification:
        post = w @ np.stack([s.bank.posterior() for s in self.segments])
        k = int(np.argmax(post))
        label, p = self.labels[k], float(post[k])
        held = self._lock is not None and self._lock[0] == label and p >= self.unlock_level
        if p >= self.lock_level:
            n = self._streak[1] + 1 if self._streak[0] == label else 1
            self._streak = (label, n)
        else:
            self._streak = ("", 0)
        locked = held or self._streak[1] >= self.confirm
        if not locked:
            self._lock = None
        elif not held:
            self._lock = (label, self.t - 1)
        lead = self.segments[int(np.argmax(w))]
        cw = lead.weights()
        return Identification(
            label=label, p=p, locked=locked, lock_slot=self._lock[1] if self._lock else None,
            start_slot=self._start, posterior=post, labels=list(self.labels),
            carrier=max(cw, key=cw.get), carrier_weights=cw)

    def predict_next(self) -> np.ndarray:
        """Distribution over the next slot's channel."""
        return self._forecast.copy()

    def predict_top1(self) -> int:
        return int(np.argmax(self._forecast))

    def predict_topk(self, k: int = 3) -> list[int]:
        return [int(c) for c in np.argsort(self._forecast)[::-1][:k]]

    def predictability(self) -> float:
        """kappa = 1 - H(p)/log M of the next-hop forecast, in [0, 1]."""
        p = self._forecast[self._forecast > 0]
        return float(1.0 - (-(p * np.log(p)).sum()) / np.log(self.M))

    def leading_segment(self) -> _Segment | None:
        """The most probable segment (its bank holds the recovered secret)."""
        return self.segments[int(np.argmax(self._weights()))] if self.segments else None


def _readonly_memo(hypotheses) -> dict:
    """``deepcopy`` memo that shares the experts' big read-only tables.

    Trajectory tables and chaos transition matrices are never written after
    construction, so every segment can point at the library's copy; all mutable
    state (weights, counts, clocks) is still copied per segment.
    """
    memo: dict = {}

    def share(a) -> None:
        memo[id(a)] = a

    for h in hypotheses:
        if isinstance(h, DeterministicHypothesis):
            share(h._filter.traj)
            share(h._filter._log_w0)
            if h.candidate_labels is not None:
                share(h.candidate_labels)
        elif isinstance(h, MixtureHMMHypothesis):
            share(h._T)
            for T in h._T:
                share(T)
            for f in h._filters:
                share(f.T)
                share(f._init)
            if h.cand_labels is not None:
                share(h.cand_labels)
    return memo
