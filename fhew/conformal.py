"""Conformal next-channel prediction sets with a distribution-free coverage
guarantee (thesis §4.2 / §6, the calibrated-set head the allocator consumes).

The :class:`~fhew.predict.ChannelPredictor` emits a *point* guess and a fixed
``top-k`` shortlist. Neither carries a guarantee: a fixed ``top-3`` covers the
true next hop a different fraction of the time on an LFSR (≈100 %) than on a
Markov hopper (≈30 %), and nothing ties the shortlist size to how predictable
the emitter actually is. This module adds a **conformal prediction set**: a set
``S_t`` of channels whose *size adapts* so that the true next channel lands in it
with a controlled long-run probability ``1 - alpha`` — on *any* sequence, with no
distributional assumption, including a generator the bank has no expert for.

Method — Adaptive Conformal Inference (ACI) over an APS-style set
----------------------------------------------------------------
Standard split-conformal assumes exchangeability, which a non-stationary FH
stream violates. We instead use the distribution-free *online* calibration of
Gibbs & Candès (Adaptive Conformal Inference): maintain an effective miscoverage
level ``alpha_t`` and, at each slot, form the smallest top-ranked set whose
predictive mass reaches ``1 - alpha_t`` (an adaptive-prediction-set, APS, rule)::

    S_t = smallest {channels by descending p} with cumulative mass >= 1 - alpha_t
    err_t = 0 if next channel in S_t else 1
    alpha_{t+1} = clip( alpha_t + gamma * (alpha - err_t),  0, 1 )

A miss (``err_t = 1``) lowers ``alpha_t`` and so *grows* the next set; a
comfortable hit shrinks it. The realised miscoverage converges to ``alpha`` for
an arbitrary sequence (the ACI guarantee), because ``alpha_t`` is bounded and the
update is a bounded-gradient online step — the same style of argument as the
CDT knob in :mod:`fhew.allocator`.

**Erasure feedback (the FH twist).** The calibration signal — "was the realised
next hop in the set?" — is only available on a *clean* slot (one single
detection). On a missed or false-alarm slot there is no ground truth to
calibrate against, so :meth:`ConformalChannelPredictor.calibrate` is simply not
called and ``alpha_t`` holds. Coverage is therefore guaranteed over the clean
sub-stream, which is all a passive node ever gets; this is stated honestly rather
than assuming full feedback.

Why this matters for action (coupling to Layer 4)
-------------------------------------------------
The set *is* the multi-tone jam plan: fund exactly the channels in ``S_t`` and
the realised next hop is denied at the conformal coverage rate ``1 - alpha`` —
turning a distribution-free statistical guarantee into an operational
"fraction of hops jammed" guarantee, with the number of tones set by the data
(one on a pinned-down emitter, up to the whole plan on an unpredictable one)
rather than by a hand-set ``max_tones``. See :func:`jam_set_from_conformal`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fhew.channels import ChannelSet
from fhew.predict import ChannelPredictor
from fhew.sensor import Observation


def aps_set(p: np.ndarray, level: float, *, min_size: int = 1) -> list[int]:
    """Adaptive-prediction-set: the smallest top-ranked channel set with mass ``level``.

    Channels are added in descending predictive probability until the cumulative
    mass first reaches ``level`` (in ``[0, 1]``). The result is a list of channel
    indices, most-probable first, of size at least ``min_size`` and at most ``M``.
    ``level = 0`` gives the single most-likely channel; ``level = 1`` the whole
    channel set. The set size is monotone non-decreasing in ``level``.
    """
    p = np.asarray(p, dtype=np.float64)
    s = p.sum()
    if s <= 0:
        p = np.full(p.size, 1.0 / p.size)
    else:
        p = p / s
    order = np.argsort(p)[::-1]
    cum = np.cumsum(p[order])
    level = float(np.clip(level, 0.0, 1.0))
    # first index whose cumulative mass reaches `level`, inclusive
    k = int(np.searchsorted(cum, level, side="left")) + 1
    k = int(np.clip(k, max(1, int(min_size)), p.size))
    return order[:k].astype(int).tolist()


class ConformalChannelPredictor:
    """Wrap a :class:`~fhew.predict.ChannelPredictor` with ACI prediction sets.

    Parameters
    ----------
    predictor:
        The underlying next-channel predictor (owns the Fixed-Share mixture and
        the universal expert). This class only reads its predictive distribution
        and never touches its belief state, so information parity is preserved.
    alpha:
        Target miscoverage in ``(0, 1)``; the set is calibrated to contain the
        true next channel ``1 - alpha`` of the time (default ``0.1`` → 90 %).
    gamma:
        ACI step size. Larger reacts faster to coverage drift (e.g. after a
        regime switch) at the cost of a noisier set size.
    min_size:
        Floor on the set size (default 1 — always name at least one channel).
    """

    def __init__(self, predictor: ChannelPredictor, *, alpha: float = 0.1,
                 gamma: float = 0.02, min_size: int = 1):
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha (target miscoverage) must be in (0, 1)")
        self.pred = predictor
        self.channels: ChannelSet = predictor.channels
        self.M = predictor.M
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.min_size = int(min_size)
        self.alpha_t = float(alpha)

    # ------------------------------------------------------------------ driving
    def reset(self) -> None:
        self.pred.reset()
        self.alpha_t = self.alpha

    def observe(self, obs: Observation) -> float:
        """Feed one slot's observation to the underlying predictor."""
        return self.pred.observe(obs)

    # --------------------------------------------------------------- prediction
    def level(self) -> float:
        """The current effective coverage target ``1 - alpha_t``."""
        return float(np.clip(1.0 - self.alpha_t, 0.0, 1.0))

    def prediction_set(self) -> list[int]:
        """The conformal set of channels for the *next* hop (most likely first)."""
        return aps_set(self.pred.predict_next(), self.level(), min_size=self.min_size)

    def predict_next(self) -> np.ndarray:
        return self.pred.predict_next()

    def predictability(self) -> float:
        return self.pred.predictability()

    # -------------------------------------------------------------- calibration
    def calibrate(self, covered: bool) -> float:
        """Update the effective level from one feedback bit (was the hop covered?).

        Call this only when a clean realised next channel is available (erasure
        feedback). Returns the new effective level ``1 - alpha_t``.
        """
        err = 0.0 if covered else 1.0
        self.alpha_t = float(np.clip(self.alpha_t + self.gamma * (self.alpha - err), 0.0, 1.0))
        return self.level()


def jam_set_from_conformal(channel_set: list[int], effect, budget: float, *,
                           channels: ChannelSet | None = None) -> np.ndarray:
    """Turn a conformal channel set into a multi-tone jam allocation (Layer 4 / D1).

    The conformal set *is* the jam plan: fund each channel in it to the denial
    threshold ``tau`` (cheapest-first so the budget stretches to the whole set
    when it can), spending any remainder partially on the next. Because the set
    is calibrated to contain the true next hop with probability ``1 - alpha``, a
    fully-funded set denies the realised next channel at that same rate — a
    distribution-free lower bound on the fraction of hops jammed. The number of
    lit tones is the (data-driven) set size, replacing the hand-set ``max_tones``.
    """
    from fhew.allocator import EffectModel  # local import to keep the module light

    assert isinstance(effect, EffectModel)
    channels = channels or effect.channels
    costs = effect.costs()
    power = np.zeros(channels.size, dtype=np.float64)
    remaining = float(budget)
    # cheapest channels in the set first, so a tight budget still covers more of it
    for c in sorted(channel_set, key=lambda i: costs[int(i)]):
        if remaining <= 0:
            break
        pay = min(float(costs[int(c)]), remaining)
        power[int(c)] = pay
        remaining -= pay
    return power


# ---------------------------------------------------------------- evaluation API
@dataclass
class ConformalScore:
    """Conformal next-channel set quality over one or more episodes."""

    n_slots: int
    target_coverage: float          # nominal 1 - alpha
    coverage: float                 # realised fraction of hops in the set
    mean_set_size: float            # mean |S_t| (number of tones)
    median_set_size: float
    mean_level: float               # mean effective 1 - alpha_t
    kappa_mean: float
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "n_slots": self.n_slots, "target_coverage": self.target_coverage,
            "coverage": self.coverage, "mean_set_size": self.mean_set_size,
            "median_set_size": self.median_set_size, "mean_level": self.mean_level,
            "kappa_mean": self.kappa_mean, **self.extra,
        }


def _clean_channel(obs: Observation) -> int | None:
    return obs.detections[0].channel if len(obs.detections) == 1 else None


def conformal_over_episode(cpred: ConformalChannelPredictor, true_channels, observations,
                           *, warmup: int = 5, reset: bool = True) -> ConformalScore:
    """Run a conformal predictor over one episode and score coverage + set size.

    At each slot ``t`` the predictor ingests ``observations[t]`` and then forms a
    set ``S_t`` for channel ``t+1``. Coverage is measured against the hidden
    ``true_channels`` (used only for scoring, never shown to the predictor). The
    ACI calibration step is driven by *erasure feedback*: it fires only when slot
    ``t+1`` is clean (a single detection), i.e. when a passive node would actually
    learn the realised hop. ``warmup`` slots are ingested but excluded from the
    score.
    """
    true = np.asarray(true_channels, dtype=np.int64)
    window = len(observations)
    if reset:
        cpred.reset()
    covered, sizes, levels, kappas = [], [], [], []
    for t, o in enumerate(observations):
        cpred.observe(o)
        if t >= window - 1:
            break
        S = cpred.prediction_set()
        nxt = int(true[t + 1])
        hit = nxt in S
        if t >= warmup:
            covered.append(int(hit))
            sizes.append(len(S))
            levels.append(cpred.level())
            kappas.append(cpred.predictability())
        # erasure feedback: calibrate only when the next slot is clean
        if _clean_channel(observations[t + 1]) is not None:
            cpred.calibrate(hit)
    n = len(covered)
    nan = float("nan")
    return ConformalScore(
        n_slots=n,
        target_coverage=1.0 - cpred.alpha,
        coverage=float(np.mean(covered)) if n else nan,
        mean_set_size=float(np.mean(sizes)) if n else nan,
        median_set_size=float(np.median(sizes)) if n else nan,
        mean_level=float(np.mean(levels)) if n else nan,
        kappa_mean=float(np.mean(kappas)) if n else nan,
    )
