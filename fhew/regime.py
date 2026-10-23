"""The regime monitor and the switch-response controller (thesis Section 6.4).

A frequency-hopping adversary that changes its generator family mid-engagement
(the "regime-switcher" red-team adversary, Section 11) makes the old expert's
predictions suddenly wrong: its log-loss spikes. The regime monitor watches the
mixture's one-step loss stream and raises an alarm on a likely switch. On an
alarm the controller does *two* things (Section 6.4):

    1. resets the Fixed-Share mixture weights (so a dead expert can win again);
    2. **forgets / windows the universal predictor's context** -- the step the
       merged architecture left implicit. Resetting the weights alone is not
       enough, because the context model's tables are still full of the old
       generator's patterns and keep mispredicting until they flush out. The
       controller calls ``mixture.forget(retain)`` so exponential forgetting
       actually happens.

It also reports the recovery metric the thesis asks for: the number of hops
after a switch until the mixture's top-1 next-hop prediction comes back
(:func:`recovery_time`, used by the regime experiment).

Detectors (all share ``update(x) -> bool`` + ``reset()``):

    * :class:`CUSUMDetector`  -- the cheap fallback: a one-sided CUSUM on the
      loss stream (the "CUSUM as cheap fallback" of Section 6.4).
    * :class:`BOCPDDetector`  -- a compact Bayesian online change-point detector
      (constant-hazard, Gaussian observation model) on the loss stream.
    * :class:`ConformalEDetector` -- a distribution-free betting e-detector with
      average-run-length control, matching the conformal spirit of Layer 3.
"""

from __future__ import annotations

import abc
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from fhew.mixture import FixedShareMixture
from fhew.sensor import Observation


class ChangeDetector(abc.ABC):
    """Common interface: feed a scalar from the loss/score stream, get an alarm."""

    name: str = "base"

    @abc.abstractmethod
    def update(self, x: float) -> bool:
        """Consume one scalar; return True iff a change-point is declared now."""

    @abc.abstractmethod
    def reset(self) -> None:
        ...


class CUSUMDetector(ChangeDetector):
    """One-sided CUSUM on the loss stream (cheap fallback, paper Section 6.4).

    Tracks a running baseline mean of the in-control loss and accumulates the
    excess ``S_t = max(0, S_{t-1} + (x_t - mean - k))``. Alarms when ``S_t > h``.
    Only *upward* shifts (loss getting worse) trigger it, which is the signature
    of an expert that no longer fits. The baseline is an EWMA that freezes while
    an alarm is pending so a sustained new regime does not get absorbed.

    ``k`` is the slack (in units of the loss std) and ``h`` the decision
    threshold (likewise). ``warmup`` slots seed the baseline before it can fire.
    """

    name = "cusum"

    def __init__(self, k: float = 0.5, h: float = 5.0, *, warmup: int = 8,
                 ewma: float = 0.05):
        self.k = float(k)
        self.h = float(h)
        self.warmup = int(warmup)
        self.ewma = float(ewma)
        self.reset()

    def reset(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0            # Welford running variance accumulator
        self.var = 1.0
        self.std = 1.0
        self.s_up = 0.0

    def update(self, x: float) -> bool:
        x = float(x)
        if not np.isfinite(x):
            return False
        self.n += 1
        if self.n <= self.warmup:
            # Seed the baseline with plain running mean/var.
            d = x - self.mean
            self.mean += d / self.n
            self.m2 = max(0.0, self.m2 + d * (x - self.mean))
            self.var = self.m2 / max(self.n - 1, 1)
            self.std = max(np.sqrt(self.var), 1e-6)
            return False
        resid = x - self.mean
        z = resid / self.std
        self.s_up = max(0.0, self.s_up + (z - self.k))
        if self.s_up > self.h:
            self.reset_after_alarm()
            return True
        # EWMA the baseline toward the current (in-control) stream. Track the
        # variance directly (not the mean-absolute-deviation, which would
        # underestimate sigma by ~0.8 and make the detector drift on noise).
        self.var = (1.0 - self.ewma) * self.var + self.ewma * resid * resid
        self.std = max(np.sqrt(self.var), 1e-6)
        self.mean = (1.0 - self.ewma) * self.mean + self.ewma * x
        return False

    def reset_after_alarm(self) -> None:
        # Keep the learned scale but drop the accumulator and re-seed the mean
        # to the new regime's current level over the next few warmup slots.
        self.s_up = 0.0
        self.n = 0
        self.m2 = 0.0


class BOCPDDetector(ChangeDetector):
    """Bayesian Online Change-Point Detection on the loss stream (Section 6.4).

    Standard constant-hazard BOCPD with a conjugate Gaussian observation model
    (known variance estimated at warmup, unknown mean with a Normal prior). It
    maintains the full run-length posterior with per-run sufficient statistics,
    so each run length's predictive tracks the data seen *since its own
    hypothesised change-point*. After a real change, mass flows from long run
    lengths to short ones over a few slots; the detector alarms when the MAP run
    length -- having grown through the first regime -- collapses. On a stationary
    stream the MAP run length only grows, so false alarms are rare. This is the
    Section 6.4 BOCPD, demoted to an auxiliary signal on the score/loss stream.
    """

    name = "bocpd"

    def __init__(self, hazard: float = 60.0, *, warmup: int = 10, max_run: int = 300,
                 kappa0: float = 0.2, map_min: int = 3, min_grow: int = 8,
                 var_floor: float = 1e-3):
        self.hazard = 1.0 / float(hazard)
        self.warmup = int(warmup)
        self.max_run = int(max_run)
        self.kappa0 = float(kappa0)         # prior precision (in obs-var units)
        self.map_min = int(map_min)
        self.min_grow = int(min_grow)
        self.var_floor = float(var_floor)
        self.reset()

    def reset(self) -> None:
        self.rl = np.array([1.0])           # P(run length = r), r = 0, 1, ...
        self.run_sum = np.array([0.0])      # sum of x over each run length
        self.run_n = np.array([0.0])        # observations in each run length
        self.n = 0
        self._buf: list[float] = []
        self.mu0 = 0.0
        self.obs_var = 1.0
        self._peak = 0

    def update(self, x: float) -> bool:
        x = float(x)
        if not np.isfinite(x):
            return False
        self.n += 1
        if self.n <= self.warmup:
            self._buf.append(x)
            self.mu0 = float(np.mean(self._buf))
            self.obs_var = max(float(np.var(self._buf)), self.var_floor)
            return False

        # Predictive of x for each current run length (conjugate Normal mean).
        post_mean = (self.kappa0 * self.mu0 + self.run_sum) / (self.kappa0 + self.run_n)
        post_var = self.obs_var * (1.0 + 1.0 / (self.kappa0 + self.run_n))
        pred = np.exp(-0.5 * (x - post_mean) ** 2 / post_var) / np.sqrt(2 * np.pi * post_var)

        growth = self.rl * pred * (1.0 - self.hazard)
        cp = float((self.rl * pred * self.hazard).sum())
        new = np.concatenate(([cp], growth))
        new_sum = np.concatenate(([0.0], self.run_sum + x))
        new_n = np.concatenate(([0.0], self.run_n + 1.0))
        if new.size > self.max_run:
            new = new[: self.max_run]; new_sum = new_sum[: self.max_run]; new_n = new_n[: self.max_run]
        total = new.sum()
        self.rl = new / total if total > 0 else np.array([1.0])
        self.run_sum, self.run_n = new_sum, new_n

        map_rl = int(np.argmax(self.rl))
        self._peak = max(self._peak, map_rl)
        if self._peak >= self.min_grow and map_rl <= self.map_min:
            # The run length grew through a regime and then collapsed -> a switch.
            self.rl = np.array([1.0]); self.run_sum = np.array([0.0])
            self.run_n = np.array([0.0]); self._peak = 0
            return True
        return False


class ConformalEDetector(ChangeDetector):
    """Distribution-free betting e-detector with ARL control (paper Section 6.4).

    Keeps a calibration window of recent losses and bets, via a non-negative
    e-process, that the newest loss exceeds its upper conformal quantile. The
    e-value grows multiplicatively while losses run high; an alarm fires when it
    crosses ``1/alpha``. By Ville's inequality the in-control average run length
    is at least ``1/alpha``, giving a distribution-free false-alarm guarantee
    without assuming a loss distribution.
    """

    name = "conformal_e"

    def __init__(self, alpha: float = 0.01, *, window: int = 40, quantile: float = 0.8,
                 bet: float = 0.5, warmup: int = 12):
        self.alpha = float(alpha)
        self.window = int(window)
        self.quantile = float(quantile)
        self.bet = float(bet)
        self.warmup = int(warmup)
        self.log_threshold = float(np.log(1.0 / alpha))
        self.reset()

    def reset(self) -> None:
        self.cal: deque[float] = deque(maxlen=self.window)
        self.log_e = 0.0
        self.n = 0

    def update(self, x: float) -> bool:
        x = float(x)
        if not np.isfinite(x):
            return False
        self.n += 1
        if self.n <= self.warmup or len(self.cal) < max(4, self.warmup // 2):
            self.cal.append(x)
            return False
        q = float(np.quantile(np.asarray(self.cal), self.quantile))
        indicator = 1.0 if x > q else 0.0
        # Bet factor: (1 + bet*(1[x>q] - (1-quantile))) is a non-negative
        # martingale increment under the in-control null (P(x>q) = 1-quantile).
        factor = 1.0 + self.bet * (indicator - (1.0 - self.quantile))
        self.log_e += np.log(max(factor, 1e-12))
        self.log_e = max(self.log_e, 0.0)       # reset to 1 on favourable evidence
        self.cal.append(x)
        if self.log_e >= self.log_threshold:
            self.log_e = 0.0
            return True
        return False


def make_detector(kind: str = "cusum", **kwargs) -> ChangeDetector:
    kind = kind.lower()
    if kind in ("cusum",):
        return CUSUMDetector(**kwargs)
    if kind in ("bocpd", "bayes"):
        return BOCPDDetector(**kwargs)
    if kind in ("conformal", "conformal_e", "e-detector", "edetector"):
        return ConformalEDetector(**kwargs)
    raise ValueError(f"unknown change detector '{kind}'")


@dataclass
class RegimeStep:
    """One slot's controller output."""

    slot: int
    loglik: float
    surprise: float
    alarmed: bool
    kappa: float
    top1: int                      # argmax of the mixture next-hop
    weights: np.ndarray


class RegimeController:
    """Wires a change detector to a :class:`FixedShareMixture` (Section 6.4).

    On each slot it drives the mixture, feeds the mixture's surprise (the
    one-step log-loss of the realised channel) to the detector, and on an alarm
    calls :meth:`FixedShareMixture.on_regime_alarm`, which both resets the
    weights and -- crucially -- forgets the context model.

    Parameters
    ----------
    retain:
        Fraction of context kept on an alarm (0 = hard flush, the default; a
        small positive value is exponential forgetting).
    reset_weights / forget_context:
        Toggle either half of the response. Setting ``forget_context=False``
        reproduces the "weights-only reset" the doc described, so the experiment
        can show the context-flush is what actually buys the fast recovery.
    cooldown:
        Minimum slots between alarms, so one switch does not fire repeatedly.
    """

    def __init__(self, mixture: FixedShareMixture, detector: ChangeDetector, *,
                 retain: float = 0.0, reset_weights: bool = True,
                 forget_context: bool = True, cooldown: int = 5):
        self.mixture = mixture
        self.detector = detector
        self.retain = float(retain)
        self.reset_weights = bool(reset_weights)
        self.forget_context = bool(forget_context)
        self.cooldown = int(cooldown)
        self.reset()

    def reset(self) -> None:
        self.mixture.reset()
        self.detector.reset()
        self.t = 0
        self._last_alarm = -(10 ** 9)
        self.alarms: list[int] = []

    def update(self, obs: Observation) -> RegimeStep:
        self.mixture.update(obs)
        step = self.mixture.last
        self.t += 1
        alarmed = False
        if step.scored:
            raised = self.detector.update(step.surprise)
            if raised and (self.t - self._last_alarm) > self.cooldown:
                self.mixture.on_regime_alarm(
                    self.retain, reset_weights=self.reset_weights,
                    forget_context=self.forget_context)
                self._last_alarm = self.t
                self.alarms.append(self.t)
                alarmed = True
        p = self.mixture.predict()
        return RegimeStep(slot=self.t, loglik=step.loglik, surprise=step.surprise,
                          alarmed=alarmed, kappa=self.mixture.predictability(),
                          top1=int(np.argmax(p)), weights=step.weights)


def recovery_time(top1_correct: list[bool] | np.ndarray, switch_at: int, *,
                  window: int = 5, thresh: float = 0.6, horizon: int | None = None) -> float:
    """Hops after a switch until the top-1 prediction reliably comes back.

    ``top1_correct[t]`` is whether the system's top-1 next-hop prediction for
    slot ``t+1`` matched the truth. The recovery time is the first offset
    ``d >= 0`` after ``switch_at`` such that the trailing ``window`` predictions
    are correct at least a fraction ``thresh`` of the time (a sustained return,
    not a lucky single hit). Returns ``nan`` if it never recovers within the
    observed horizon -- that is itself a reportable outcome (e.g. against a
    crypto regime after the switch).
    """
    arr = np.asarray(top1_correct, dtype=np.float64)
    n = arr.size
    hi = n if horizon is None else min(n, switch_at + horizon)
    for s in range(switch_at, hi):
        end = min(s + window, n)
        if end - s <= 0:
            break
        if float(arr[s:end].mean()) >= thresh:
            return float(s - switch_at)
    return float("nan")
