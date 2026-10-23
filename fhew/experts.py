"""The Expert interface, the hypotheses, the none-of-the-above class, and the bank.

This module merges what used to be the fhew/experts/ subpackage; the public
names are unchanged, so `from fhew.experts import X` still works.
"""

from __future__ import annotations

# ---- merged from fhew/experts/base.py ----
import abc

import numpy as np

from fhew.sensor import Observation


class Expert(abc.ABC):
    """A next-hop predictor that consumes the lossy observation stream.

    ``update`` returns the slot's reduced log-likelihood increment (useful to a
    log-loss mixture); ``predict`` returns the next-hop distribution over
    channels. Subclasses add their own state (a bank posterior, a neural hidden
    state, ...), but nothing outside this interface is required of them.
    """

    #: human-readable label, used in logs and metrics
    label: str = "expert"

    @abc.abstractmethod
    def update(self, obs: Observation) -> float:
        """Consume one slot's observation; return the reduced log-likelihood."""

    @abc.abstractmethod
    def predict(self) -> np.ndarray:
        """Return the next-hop distribution ``p(c)`` over channels."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Return to the pre-engagement state."""

    def forget(self, retain: float = 0.0) -> None:
        """Exponentially forget learned context after a regime switch.

        ``retain`` in ``[0, 1]`` is the fraction of accumulated evidence kept
        (0 = flush to the pre-engagement prior, 1 = no forgetting). Stateless
        experts need do nothing; online predictors shrink their count tables and
        windowed context toward the prior (paper Section 6.4). This is the hook
        the regime monitor calls so a switch flushes the *context model*, not
        only the mixture weights.
        """
        return None

# ---- merged from fhew/experts/hypothesis.py ----
import abc
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.special import log_softmax, logsumexp

from fhew.channels import ChannelSet
from fhew.generators import make_generator
from fhew.generators import WeakSeedGenerator
from fhew.generators import _PRIMITIVE_TAPS, _bits_per_channel
from fhew.likelihood import (
    ChannelHMMFilter,
    DeterministicTrajectoryFilter,
    StepResult,
)
from fhew.recovery import (
    bits_to_channels,
    channels_to_bits,
    recover_and_predict,
    run_lfsr_from_coeffs,
)


class Hypothesis(abc.ABC):
    """A single bank hypothesis over one generator family."""

    label: str

    @abc.abstractmethod
    def step(self, log_emission: np.ndarray) -> StepResult:
        ...

    @abc.abstractmethod
    def predict_next(self) -> np.ndarray:
        ...

    @abc.abstractmethod
    def reset(self) -> None:
        ...

    def forget(self, retain: float = 0.0) -> None:
        """Exponentially forget learned context (paper Section 6.4).

        Default no-op for stateless / enumerated hypotheses (their candidate
        posterior is already recovered by Fixed-Share on the mixture side).
        Online predictors override this to shrink their context toward the
        pre-engagement prior; see :meth:`Expert.forget`.
        """
        return None

    @property
    def n_particles(self) -> int:  # pragma: no cover - overridden
        return 0


class DeterministicHypothesis(Hypothesis):
    """Marginalises over a bank of candidate deterministic trajectories."""

    def __init__(self, label: str, trajectories: np.ndarray, n_channels: int,
                 candidate_labels: list | None = None):
        self.label = label
        self.n_channels = n_channels
        self._filter = DeterministicTrajectoryFilter(trajectories, n_channels)
        self.candidate_labels = candidate_labels
        self._last_predict = self._filter.predict_current()

    def step(self, log_emission: np.ndarray) -> StepResult:
        res = self._filter.step(log_emission)
        self._last_predict = res.predict_next
        return res

    def predict_next(self) -> np.ndarray:
        return self._last_predict

    def reset(self) -> None:
        self._filter.reset()
        self._last_predict = self._filter.predict_current()

    @property
    def n_particles(self) -> int:
        return self._filter.K

    def map_candidate(self) -> int:
        """Index of the most probable candidate (for diagnostics / recovery)."""
        return int(np.argmax(self._filter.log_w))

    def candidate_posterior(self) -> np.ndarray:
        return np.exp(self._filter.log_w)

    def map_trajectory(self) -> np.ndarray:
        """Channel trajectory of the most probable candidate."""
        return self._filter.traj[self.map_candidate()]


class MarkovHypothesis(Hypothesis):
    """First-order Markov hypothesis with an online-fitted transition matrix.

    Under information parity the transition cannot be handed over; it is learned
    prequentially from clean (single-detection) slots with Laplace smoothing. The
    likelihood each slot uses the transition fitted from *earlier* slots only
    (one-step-ahead), so it is an honest predictive score.
    """

    def __init__(self, label: str, channels: ChannelSet, alpha: float = 1.0):
        self.label = label
        self.channels = channels
        self.M = channels.size
        self.alpha = float(alpha)
        self.reset()

    def reset(self) -> None:
        self.counts = np.full((self.M, self.M), self.alpha, dtype=np.float64)
        self.prior = self.channels.uniform()
        self._last_clean = None
        self._last_predict = self.channels.uniform()

    def forget(self, retain: float = 0.0) -> None:
        retain = float(np.clip(retain, 0.0, 1.0))
        # shrink the fitted transition evidence back toward the Laplace prior
        self.counts = self.alpha + retain * (self.counts - self.alpha)
        self.prior = self.channels.uniform()
        self._last_clean = None
        self._last_predict = self.channels.uniform()

    def _transition(self) -> np.ndarray:
        return self.counts / self.counts.sum(axis=1, keepdims=True)

    def step(self, log_emission: np.ndarray) -> StepResult:
        m = float(np.max(log_emission))
        e = np.exp(log_emission - m)
        weighted = self.prior * e
        total = float(weighted.sum())
        if total <= 0.0:
            loglik = m + float(np.log(np.finfo(float).tiny))
            post = self.prior
        else:
            loglik = m + float(np.log(total))
            post = weighted / total
        T = self._transition()
        predict_next = T.T @ post
        self.prior = predict_next
        self._last_predict = predict_next
        return StepResult(loglik=loglik, predict_next=predict_next)

    def observe_clean(self, channel: int | None) -> None:
        """Feed a clean single-detection channel to the online transition fit.

        Called by the bank after :meth:`step` so the fit never uses the current
        slot to score itself.
        """
        if channel is not None and self._last_clean is not None:
            self.counts[self._last_clean, channel] += 1.0
        if channel is not None:
            self._last_clean = channel

    def predict_next(self) -> np.ndarray:
        return self._last_predict

    @property
    def n_particles(self) -> int:
        return self.M


# ======================================================================
# Additional experts completing the thesis bank (Section 5 expert table).
# Each is self-contained and plugs into the same forward-filter interface;
# the registry/dispatch at the bottom wires them into build_hypothesis.
# ======================================================================


class BarrageHypothesis(Hypothesis):
    """Prior-weighted barrage floor (the zero-hop expert, paper Section 5).

    Assumes no structure: every slot it predicts the reconnaissance-weighted
    barrage distribution over the effective channel set. It is the floor the
    system falls back to when nothing is predictable; as an explicit expert it
    lets the bank report "spread over the channel plan" as an outcome.
    """

    def __init__(self, label: str, channels: ChannelSet, floor: str = "barrage"):
        self.label = label
        self.channels = channels
        self._floor = channels.barrage_floor() if floor == "barrage" else channels.uniform()
        self.reset()

    def reset(self) -> None:
        self._last_predict = self._floor.copy()

    def step(self, log_emission: np.ndarray) -> StepResult:
        m = float(np.max(log_emission))
        e = np.exp(log_emission - m)
        tot = float((self._floor * e).sum())
        loglik = m + float(np.log(tot)) if tot > 0 else m + float(np.log(np.finfo(float).tiny))
        return StepResult(loglik=loglik, predict_next=self._floor.copy())

    def predict_next(self) -> np.ndarray:
        return self._floor.copy()

    @property
    def n_particles(self) -> int:
        return self.channels.size


class _PredictorHypothesis(Hypothesis):
    """Base for online predictor experts that learn from the clean hop stream.

    Shares the prequential pattern: ``step`` scores the current slot with a
    one-step-ahead predictive formed from *earlier* clean channels, then
    ``observe_clean`` ingests the slot's clean channel and refreshes the
    prediction for the next slot. Gap-robust because it reads only clean
    single-detection slots. Subclasses implement the model hooks.
    """

    def __init__(self, label: str, channels: ChannelSet):
        self.label = label
        self.channels = channels
        self.M = channels.size
        self.reset()

    def reset(self) -> None:
        self._reset_model()
        self.prior = self.channels.uniform()
        self._last_predict = self.channels.uniform()

    def step(self, log_emission: np.ndarray) -> StepResult:
        m = float(np.max(log_emission))
        e = np.exp(log_emission - m)
        tot = float((self.prior * e).sum())
        loglik = m + float(np.log(tot)) if tot > 0 else m + float(np.log(np.finfo(float).tiny))
        self._last_predict = self.prior.copy()
        return StepResult(loglik=loglik, predict_next=self.prior.copy())

    def observe_clean(self, channel: int | None) -> None:
        if channel is not None:
            self._ingest(int(channel))
        # self.prior now holds P(s_{t+1} | clean_{1:t}); surface it as the
        # next-hop (step() could only offer the current-slot prediction, since it
        # runs before this slot's clean channel is known).
        self.prior = self._predict_next_dist()
        self._last_predict = self.prior.copy()

    def predict_next(self) -> np.ndarray:
        return self._last_predict

    def forget(self, retain: float = 0.0) -> None:
        """Flush/window the learned context after a regime switch.

        Delegates the model-specific shrink to :meth:`_forget_model`, then
        rolls the predictive back to the prior. ``retain=0`` is a hard flush
        (the context model's stale patterns are the gap flagged in Section 6.4);
        ``retain`` in ``(0,1)`` is exponential forgetting.
        """
        retain = float(np.clip(retain, 0.0, 1.0))
        self._forget_model(retain)
        self.prior = self.channels.uniform()
        self._last_predict = self.channels.uniform()

    @property
    def n_particles(self) -> int:
        return self.M

    # ---- subclass hooks ----
    def _reset_model(self) -> None:  # pragma: no cover - overridden
        ...

    def _forget_model(self, retain: float) -> None:
        """Shrink learned state toward the prior. Default: a hard reset when
        flushing, otherwise leave the model (subclasses do proper decay)."""
        if retain <= 0.0:
            self._reset_model()

    def _ingest(self, channel: int) -> None:  # pragma: no cover - overridden
        ...

    def _predict_next_dist(self) -> np.ndarray:  # pragma: no cover - overridden
        return self.channels.uniform()


class ContextMixingHypothesis(_PredictorHypothesis):
    """Zero-shot, gap-robust in-context predictor (variable-order Markov mixture).

    A training-free universal predictor (PPM / context-mixing style): count
    models of orders ``0..max_order`` each predict the next channel from their
    context, and the mixture weights them by recent predictive log-loss
    (exponential weights). It adapts in-context across the whole adversary
    library with no per-sequence training, standing in for the journal's
    zero-shot neural expert without a deep-learning dependency. This is the
    expert that lets the system **predict a generator it has no dedicated expert
    for**: a variable-order rule (e.g. ``c_t = c_{t-1} + c_{t-2}``) is recovered
    in-context even though no bank hypothesis targets it.

    **Gap-robustness.** A context is only ever formed from a *contiguous run of
    clean single-detection slots* ending at the slot before the one being
    predicted. A missed hop or a false-alarm slot (which cannot be trusted as a
    channel) breaks the run, so the predictor never learns or predicts from a
    context that straddles a gap -- it falls back to lower orders until a clean
    run rebuilds. This keeps the order-``k`` tables uncorrupted under a lossy
    sensor, which is what makes it usable for exact deterministic rules through
    missed detections.
    """

    def __init__(self, label: str, channels: ChannelSet, max_order: int = 3,
                 eta: float = 0.5, laplace: float = 0.5):
        self.max_order = int(max_order)
        self.eta = float(eta)
        self.laplace = float(laplace)
        super().__init__(label, channels)

    def _reset_model(self) -> None:
        K = self.max_order
        self.weights = np.full(K + 1, 1.0 / (K + 1))
        self.tables = [dict() for _ in range(K + 1)]
        # the current contiguous run of clean channels (most recent last); a gap
        # or false-alarm slot clears it so no context ever straddles a gap.
        self.run: list[int] = []

    def _ctx(self, order: int) -> tuple | None:
        """Context for ``order`` from the contiguous clean run, or None if short."""
        if order == 0:
            return ()
        if len(self.run) >= order:
            return tuple(self.run[-order:])
        return None

    def _order_pred(self, order: int) -> np.ndarray:
        key = self._ctx(order)
        counts = self.tables[order].get(key) if key is not None else None
        if counts is None:
            return self.channels.uniform()
        sm = counts + self.laplace
        return sm / sm.sum()

    def observe_clean(self, channel: int | None) -> None:
        if channel is None:
            # a missed or false-alarm slot: the next channel cannot be predicted
            # from a contiguous context, so break the run.
            self.run = []
        else:
            self._ingest(int(channel))
            self.run.append(int(channel))
            if len(self.run) > self.max_order:
                self.run.pop(0)
        self.prior = self._predict_next_dist()
        self._last_predict = self.prior.copy()

    def _ingest(self, channel: int) -> None:
        # `self.run` holds the contiguous clean channels immediately *before* this
        # slot, so every (context -> channel) pair recorded here is genuine.
        preds = [self._order_pred(o) for o in range(self.max_order + 1)]
        losses = np.array([-math.log(max(float(p[channel]), 1e-12)) for p in preds])
        self.weights = self.weights * np.exp(-self.eta * losses)
        self.weights /= self.weights.sum()
        for o in range(self.max_order + 1):
            key = self._ctx(o)
            if key is None:
                continue
            tab = self.tables[o]
            if key not in tab:
                tab[key] = np.zeros(self.M)
            tab[key][channel] += 1.0

    def _predict_next_dist(self) -> np.ndarray:
        mix = np.zeros(self.M)
        for o in range(self.max_order + 1):
            mix += self.weights[o] * self._order_pred(o)
        s = mix.sum()
        return mix / s if s > 0 else self.channels.uniform()

    def _forget_model(self, retain: float) -> None:
        # Exponential forgetting of the context tables + a context-window flush.
        # This is the specific fix for Section 6.4: the universal predictor's
        # tables are full of the *old* generator's patterns right after a switch,
        # so they must be shrunk (not just the mixture weights reset) or the
        # predictor keeps mispredicting until they flush out on their own.
        if retain <= 0.0:
            self._reset_model()
            return
        for tab in self.tables:
            for key in list(tab.keys()):
                tab[key] *= retain
        K = self.max_order
        self.weights = np.full(K + 1, 1.0 / (K + 1))
        self.run = []                    # window: drop the stale contiguous context

    def map_order(self) -> int:
        return int(np.argmax(self.weights))


class PhaseSpacePredictorHypothesis(_PredictorHypothesis):
    """Short-horizon forecaster for unknown chaotic dynamics (paper Section 5).

    Takens-style delay embedding + nearest-neighbour prediction: it embeds the
    clean channel stream into ``dim``-dimensional delay vectors, finds the
    ``k`` past states nearest the current one, and predicts their successors. It
    needs no knowledge of the map family; it fails (error reaches uniform) when
    the dynamics are not low-dimensional.
    """

    def __init__(self, label: str, channels: ChannelSet, dim: int = 3, k: int = 8):
        self.dim = int(dim)
        self.k = int(k)
        super().__init__(label, channels)

    def _reset_model(self) -> None:
        self.hist: list[int] = []

    def _forget_model(self, retain: float) -> None:
        # Window the delay-embedding history: keep only the most recent fraction
        # so neighbours come from the new regime's trajectory, not the old one.
        if retain <= 0.0:
            self.hist = []
        else:
            keep = int(round(retain * len(self.hist)))
            self.hist = self.hist[-keep:] if keep > 0 else []

    def _ingest(self, channel: int) -> None:
        self.hist.append(channel)

    def _predict_next_dist(self) -> np.ndarray:
        h = self.hist
        n = len(h)
        d = self.dim
        if n < d + 1:
            return self.channels.uniform()
        query = np.asarray(h[-d:], dtype=np.float64)
        past = np.asarray(h, dtype=np.float64)
        # delay vectors end at indices i in [d-1, n-2]; successor is h[i+1]
        idx = np.arange(d - 1, n - 1)
        vecs = np.stack([past[i - d + 1:i + 1] for i in idx])
        dists = np.sum((vecs - query) ** 2, axis=1)
        order = np.argsort(dists)[: self.k]
        counts = np.full(self.M, 0.1)
        for j in order:
            counts[h[idx[j] + 1]] += 1.0
        return counts / counts.sum()


class BehaviourHypothesis(_PredictorHypothesis):
    """Reduced reactive-behaviour predictor (stand-in for the journal TD-MPC2).

    A recency-weighted first-order model: transition counts decay each step, so
    the prediction tracks a drifting or reactive policy rather than a stationary
    one. In the passive conference setting there are no own-actions to condition
    on, so this is intentionally a reduced stand-in for the journal's model-based
    behaviour expert; it degrades to uniform as a policy adapts away from its
    recent statistics.
    """

    def __init__(self, label: str, channels: ChannelSet, decay: float = 0.9,
                 laplace: float = 0.2):
        self.decay = float(decay)
        self.laplace = float(laplace)
        super().__init__(label, channels)

    def _reset_model(self) -> None:
        self.counts = np.full((self.M, self.M), self.laplace)
        self.last = None

    def _forget_model(self, retain: float) -> None:
        if retain <= 0.0:
            self._reset_model()
        else:
            self.counts = self.laplace + retain * (self.counts - self.laplace)
            self.last = None

    def _ingest(self, channel: int) -> None:
        if self.last is not None:
            self.counts *= self.decay
            self.counts[self.last, channel] += 1.0
        self.last = channel

    def _predict_next_dist(self) -> np.ndarray:
        if self.last is None:
            return self.channels.uniform()
        row = self.counts[self.last]
        return row / row.sum()


class MixtureHMMHypothesis(Hypothesis):
    """A posterior over candidate channel-transition models (used by chaos ID).

    Holds one :class:`ChannelHMMFilter` per candidate transition matrix and a
    posterior over candidates, updated by each candidate's one-step likelihood.
    The marginal is the hypothesis likelihood; the winning candidate names the
    recovered parameter.
    """

    def __init__(self, label: str, channels: ChannelSet, transitions: list,
                 cand_labels: list | None = None):
        self.label = label
        self.channels = channels
        self.M = channels.size
        self._T = [np.asarray(T, dtype=np.float64) for T in transitions]
        if not self._T:
            raise ValueError(f"MixtureHMMHypothesis '{label}' needs >=1 transition")
        self.cand_labels = cand_labels
        self.reset()

    def reset(self) -> None:
        self._filters = [ChannelHMMFilter(T) for T in self._T]
        self.log_w = np.full(len(self._T), -np.log(len(self._T)))
        self._last_predict = self.channels.uniform()

    def forget(self, retain: float = 0.0) -> None:
        # Flatten the candidate posterior toward uniform and restart the HMM
        # filters so a post-switch map lands fresh on the new regime.
        retain = float(np.clip(retain, 0.0, 1.0))
        for f in self._filters:
            f.reset()
        n = len(self._T)
        self.log_w = retain * self.log_w + (1.0 - retain) * np.full(n, -np.log(n))
        self.log_w -= logsumexp(self.log_w)
        self._last_predict = self.channels.uniform()

    def step(self, log_emission: np.ndarray) -> StepResult:
        lls = np.empty(len(self._filters))
        preds = np.empty((len(self._filters), self.M))
        for i, f in enumerate(self._filters):
            r = f.step(log_emission)
            lls[i] = r.loglik
            preds[i] = r.predict_next
        a = self.log_w + lls
        mix = float(logsumexp(a))
        self.log_w = a - mix
        w = np.exp(self.log_w)
        predict_next = w @ preds
        s = predict_next.sum()
        predict_next = predict_next / s if s > 0 else self.channels.uniform()
        self._last_predict = predict_next
        return StepResult(loglik=mix, predict_next=predict_next)

    def predict_next(self) -> np.ndarray:
        return self._last_predict

    def map_candidate(self):
        i = int(np.argmax(self.log_w))
        return self.cand_labels[i] if self.cand_labels else i

    @property
    def n_particles(self) -> int:
        return len(self._T) * self.M


def _induced_transition(family: str, kw: dict, M: int, n_sim: int = 6000,
                        starts=(0.1, 0.4, 0.7), alpha: float = 1.0) -> np.ndarray:
    """Estimate the quantised symbolic-dynamics transition matrix of a chaotic map.

    Simulates the map (over several initial conditions) and bins consecutive
    channels. Deterministic given ``(family, kw, M)``.
    """
    counts = np.full((M, M), alpha, dtype=np.float64)
    for x0 in starts:
        if family == "henon":
            g = make_generator(family, M, x0=float(x0) - 0.4, y0=0.0, **kw)
        else:
            g = make_generator(family, M, x0=float(x0), **kw)
        seq = g.sequence(n_sim)
        flat = np.ravel_multi_index((seq[:-1], seq[1:]), (M, M))
        np.add.at(counts.reshape(-1), flat, 1.0)
    return counts / counts.sum(axis=1, keepdims=True)


_DEFAULT_CHAOS_GRID = {
    "logistic": [3.6, 3.7, 3.8, 3.9, 3.95, 4.0],
    "tent": [1.9, 1.95, 1.99, 2.0],
    "henon": [(1.4, 0.3), (1.2, 0.3), (1.3, 0.3)],
}


def _build_chaos_id(spec: "HypothesisSpec", channels: ChannelSet) -> MixtureHMMHypothesis:
    grid = spec.grid or _DEFAULT_CHAOS_GRID
    n_sim = int(spec.options.get("n_sim", 6000))
    transitions, labels = [], []
    for family, params in grid.items():
        for p in params:
            if family == "logistic":
                kw = {"r": float(p)}
            elif family == "tent":
                kw = {"mu": float(p)}
            elif family == "henon":
                a, b = p if isinstance(p, (list, tuple)) else (1.4, 0.3)
                kw = {"a": float(a), "b": float(b)}
            else:
                continue
            transitions.append(_induced_transition(family, kw, channels.size, n_sim=n_sim))
            labels.append((family, p))
    return MixtureHMMHypothesis(spec.label, channels, transitions, labels)


def _longest_clean_run(clean: dict):
    """Return ``(start_slot, [channels])`` for the longest run of consecutive
    clean slots, or ``None`` if empty."""
    if not clean:
        return None
    slots = sorted(clean)
    best_start, best = slots[0], [clean[slots[0]]]
    cur_start, cur = slots[0], [clean[slots[0]]]
    for prev, s in zip(slots, slots[1:]):
        if s == prev + 1:
            cur.append(clean[s])
        else:
            if len(cur) > len(best):
                best_start, best = cur_start, cur
            cur_start, cur = s, [clean[s]]
    if len(cur) > len(best):
        best_start, best = cur_start, cur
    return best_start, best


class AlgebraicLFSRHypothesis(Hypothesis):
    """Plain Berlekamp-Massey expert: collapse an LFSR from clean *consecutive*
    outputs (paper Section 5).

    Unlike the gap-robust state-filtered variant (the enumeration
    :class:`DeterministicHypothesis`), this needs a run of consecutive clean
    slots of length about ``2L`` to recover the recurrence; any missed detection
    inside the run breaks it. Once recovered it rolls the register forward across
    gaps and predicts deterministically. This realises the brittle "Berlekamp-
    Massey" row distinct from "State-filtered BM".
    """

    def __init__(self, label: str, channels: ChannelSet, max_length: int = 9,
                 min_run: int | None = None):
        self.label = label
        self.channels = channels
        self.M = channels.size
        self.b = _bits_per_channel(self.M)
        self.max_length = int(max_length)
        self.min_run = int(min_run) if min_run else (2 * self.max_length + 4)
        self.reset()

    def reset(self) -> None:
        self.t = 0
        self._clean = {}
        self._recovered = False
        self._coeffs = None
        self._L = None
        self._s = None
        self._bits = None
        self._last_predict = self.channels.uniform()

    def step(self, log_emission: np.ndarray) -> StepResult:
        if self._recovered:
            ch = self._channel_at(self.t)                 # current slot: for loglik
            loglik = float(log_emission[ch])
            nxt = self._channel_at(self.t + 1)            # next slot: for the predictive
            pred = np.zeros(self.M)
            pred[nxt] = 1.0
        else:
            pred = self.channels.uniform()
            m = float(np.max(log_emission))
            e = np.exp(log_emission - m)
            loglik = m + float(np.log(float((pred * e).sum())))
        self._last_predict = pred
        self.t += 1
        return StepResult(loglik=loglik, predict_next=pred)

    def observe_clean(self, channel: int | None) -> None:
        if channel is not None:
            self._clean[self.t - 1] = int(channel)
            if not self._recovered:
                self._try_recover()

    def _try_recover(self) -> None:
        run = _longest_clean_run(self._clean)
        if run is None or len(run[1]) < self.min_run:
            return
        s, chans = run
        res = recover_and_predict(chans, b=self.b, horizon=1, max_length=self.max_length)
        if not res.recovered:
            return
        self._coeffs = res.coeffs
        self._L = res.length
        self._s = s
        self._bits = list(int(x) for x in channels_to_bits(np.asarray(chans), self.b))
        self._recovered = True

    def _channel_at(self, slot: int) -> int:
        need = (slot - self._s + 1) * self.b
        while len(self._bits) < need:
            self._bits.append(run_lfsr_from_coeffs(self._bits, self._coeffs, self._L))
        off = (slot - self._s) * self.b
        v = 0
        for k in range(self.b):
            v = (v << 1) | int(self._bits[off + k])
        return v

    def predict_next(self) -> np.ndarray:
        return self._last_predict

    @property
    def n_particles(self) -> int:
        return 1


def recover_lcg_full(outputs, m: int):
    """Recover ``(a, c)`` of a full-state LCG from consecutive outputs mod ``m``.

    Uses ``a = (x2 - x1)(x1 - x0)^{-1} mod m`` over a usable triple, then
    verifies against the whole run. Returns ``None`` if no consistent parameters
    are found (the "lattice reduction finds nothing" failure of Section 5).
    """
    xs = [int(v) for v in outputs]
    n = len(xs)
    if n < 4:
        return None
    for i in range(n - 2):
        d0 = (xs[i + 1] - xs[i]) % m
        d1 = (xs[i + 2] - xs[i + 1]) % m
        if math.gcd(d0, m) != 1:
            continue
        a = (d1 * pow(d0, -1, m)) % m
        c = (xs[i + 1] - a * xs[i]) % m
        if all((a * xs[j] + c) % m == xs[j + 1] for j in range(n - 1)):
            return a, c
    return None


class LCGCryptanalysisHypothesis(Hypothesis):
    """Algebraic LCG cryptanalysis expert (paper Section 5).

    Targets a full-state LCG (``lcg_full``) whose channel is the state itself.
    From a few consecutive clean outputs it recovers the multiplier and increment
    algebraically (modulus = M known), then predicts the future deterministically
    from the last known state -- robust to gaps once recovered. Distinct from the
    seed-search LCG hypothesis, which attacks the high-bit-truncated generator.
    """

    def __init__(self, label: str, channels: ChannelSet, min_run: int = 6):
        # min_run >= 6 keeps the chance of a spurious consistent (a,c) on a
        # non-LCG run negligible (~ M**-(min_run-3)), so this expert almost never
        # makes a false algebraic claim.
        self.label = label
        self.channels = channels
        self.M = channels.size
        self.min_run = int(min_run)
        self.reset()

    def reset(self) -> None:
        self.t = 0
        self._clean = {}
        self._recovered = False
        self._a = None
        self._c = None
        self._last_known_slot = None
        self._last_known_state = None
        self._last_predict = self.channels.uniform()

    def step(self, log_emission: np.ndarray) -> StepResult:
        if self._recovered and self._last_known_slot is not None:
            state = self._last_known_state
            for _ in range(self.t - self._last_known_slot):
                state = (self._a * state + self._c) % self.M   # state at current slot t
            loglik = float(log_emission[state])
            nxt = (self._a * state + self._c) % self.M         # state at slot t+1
            pred = np.zeros(self.M)
            pred[nxt] = 1.0
        else:
            pred = self.channels.uniform()
            m = float(np.max(log_emission))
            e = np.exp(log_emission - m)
            loglik = m + float(np.log(float((pred * e).sum())))
        self._last_predict = pred
        self.t += 1
        return StepResult(loglik=loglik, predict_next=pred)

    def observe_clean(self, channel: int | None) -> None:
        if channel is None:
            return
        slot = self.t - 1
        self._clean[slot] = int(channel)
        if not self._recovered:
            self._last_known_slot = slot
            self._last_known_state = int(channel)
            run = _longest_clean_run(self._clean)
            if run is not None and len(run[1]) >= self.min_run:
                res = recover_lcg_full(run[1], self.M)
                if res is not None:
                    self._a, self._c = res
                    self._recovered = True
        elif self._last_known_slot is not None:
            # Already recovered: only re-anchor when the clean channel agrees with
            # the recurrence. A lone false-alarm detection would otherwise corrupt
            # the anchor; keep rolling from the last trusted state instead.
            predicted = self._last_known_state
            for _ in range(slot - self._last_known_slot):
                predicted = (self._a * predicted + self._c) % self.M
            if int(channel) == predicted:
                self._last_known_slot = slot
                self._last_known_state = int(channel)

    def predict_next(self) -> np.ndarray:
        return self._last_predict

    @property
    def n_particles(self) -> int:
        return 1


# ----------------------------------------------------------------- spec / build
@dataclass
class HypothesisSpec:
    """Declarative description of a bank hypothesis (config-driven, Section 12)."""

    label: str
    family: str
    kind: str = "deterministic"           # deterministic | markov | uniform
    grid: dict = field(default_factory=dict)
    max_particles: int = 8192
    options: dict = field(default_factory=dict)


def _lfsr_trajectories(channels, horizon, grid, max_particles):
    M = channels.size
    b = _bits_per_channel(M)
    lengths = grid.get("length", [7])
    out = []
    labels = []
    for L in lengths:
        taps_list = grid.get("taps", {}).get(L, [_PRIMITIVE_TAPS[L]]) if isinstance(
            grid.get("taps"), dict) else [_PRIMITIVE_TAPS[L]]
        for taps in taps_list:
            for state in range(1, 1 << L):
                g = make_generator("lfsr", M, length=L, taps=taps, state=state)
                out.append(g.sequence(horizon))
                labels.append(("lfsr", L, taps, state))
                if len(out) >= max_particles:
                    return np.asarray(out, dtype=np.int64), labels
    return np.asarray(out, dtype=np.int64), labels


def _weak_seed_trajectories(channels, horizon, grid, max_particles):
    M = channels.size
    keyspace = int(grid.get("keyspace", 256))
    keyspace = min(keyspace, max_particles)
    t = np.arange(horizon)
    traj = np.empty((keyspace, horizon), dtype=np.int64)
    for s in range(keyspace):
        traj[s] = [WeakSeedGenerator.channel_for(s, int(tt), M) for tt in t]
    labels = [("weak_seed", s) for s in range(keyspace)]
    return traj, labels


def _lcg_trajectories(channels, horizon, grid, max_particles):
    M = channels.size
    modulus = int(grid.get("modulus", 2 ** 16))
    seed_max = int(grid.get("seed_max", min(modulus, max_particles)))
    seed_max = min(seed_max, max_particles)
    out = []
    labels = []
    for seed in range(seed_max):
        g = make_generator("lcg", M, modulus=modulus, seed=seed)
        out.append(g.sequence(horizon))
        labels.append(("lcg", seed))
    return np.asarray(out, dtype=np.int64), labels


def _generic_grid_trajectories(family, channels, horizon, grid, max_particles):
    """Cartesian-product enumeration for any registered deterministic family."""
    import itertools

    M = channels.size
    keys = list(grid.keys())
    value_lists = [grid[k] if isinstance(grid[k], (list, tuple)) else [grid[k]] for k in keys]
    out = []
    labels = []
    for combo in itertools.product(*value_lists):
        params = dict(zip(keys, combo))
        g = make_generator(family, M, **params)
        out.append(g.sequence(horizon))
        labels.append((family, combo))
        if len(out) >= max_particles:
            break
    return np.asarray(out, dtype=np.int64), labels


def _filtered_lfsr_trajectories(channels, horizon, grid, max_particles):
    """Correlation-attack expert: enumerate register states of a *known* filtered
    LFSR family and score each through the sensor-aware filter.

    This is the Bayesian-optimal analogue of a correlation attack on a
    nonlinearly-filtered LFSR: the recce supplies the filter family
    ``(length, taps, filter_taps)`` and we marginalise over the register state,
    recovering it under gaps where plain Berlekamp-Massey fails.
    """
    M = channels.size
    lengths = grid.get("length", [9])
    ft_list = grid.get("filter_taps") or [tuple(range(7))]
    ft_list = [tuple(ft) for ft in ft_list]
    out, labels = [], []
    for L in lengths:
        taps = grid.get("taps", {}).get(L, _PRIMITIVE_TAPS[L]) if isinstance(
            grid.get("taps"), dict) else _PRIMITIVE_TAPS[L]
        for ft in ft_list:
            for state in range(1, 1 << L):
                g = make_generator("filtered_lfsr", M, length=L, taps=taps,
                                   state=state, filter_taps=ft)
                out.append(g.sequence(horizon))
                labels.append(("filtered_lfsr", L, ft, state))
                if len(out) >= max_particles:
                    return np.asarray(out, dtype=np.int64), labels
    return np.asarray(out, dtype=np.int64), labels


_TRAJ_BUILDERS = {
    "lfsr": _lfsr_trajectories,
    "weak_seed": _weak_seed_trajectories,
    "lcg": _lcg_trajectories,
    "filtered_lfsr": _filtered_lfsr_trajectories,
}


def build_hypothesis(spec: HypothesisSpec, channels: ChannelSet, horizon: int,
                     rng: np.random.Generator | None = None) -> Hypothesis:
    """Instantiate a hypothesis from its spec and the recce ranges."""
    if spec.kind == "uniform":
        return H0Hypothesis(channels, mode="uniform",
                            floor=spec.options.get("floor", "uniform"), label=spec.label)
    if spec.kind == "markov":
        return MarkovHypothesis(spec.label, channels, alpha=spec.options.get("alpha", 1.0))
    if spec.kind == "barrage":
        return BarrageHypothesis(spec.label, channels,
                                 floor=spec.options.get("floor", "barrage"))
    if spec.kind == "context_mixing":
        return ContextMixingHypothesis(spec.label, channels,
                                       max_order=spec.options.get("max_order", 3),
                                       eta=spec.options.get("eta", 0.5))
    if spec.kind == "phase_space":
        return PhaseSpacePredictorHypothesis(spec.label, channels,
                                             dim=spec.options.get("dim", 3),
                                             k=spec.options.get("k", 8))
    if spec.kind == "behaviour":
        return BehaviourHypothesis(spec.label, channels,
                                   decay=spec.options.get("decay", 0.9))
    if spec.kind == "chaos_id":
        return _build_chaos_id(spec, channels)
    if spec.kind == "bm":
        return AlgebraicLFSRHypothesis(spec.label, channels,
                                       max_length=spec.options.get("max_length", 9))
    if spec.kind == "lcg_crypto":
        return LCGCryptanalysisHypothesis(spec.label, channels,
                                          min_run=spec.options.get("min_run", 6))
    if spec.kind == "deterministic":
        builder = _TRAJ_BUILDERS.get(spec.family)
        if builder is not None:
            traj, labels = builder(channels, horizon, spec.grid, spec.max_particles)
        else:
            traj, labels = _generic_grid_trajectories(
                spec.family, channels, horizon, spec.grid, spec.max_particles)
        if traj.size == 0:
            raise ValueError(f"hypothesis '{spec.label}' built zero candidates")
        return DeterministicHypothesis(spec.label, traj, channels.size, labels)
    raise ValueError(f"unknown hypothesis kind '{spec.kind}'")

# ---- merged from fhew/experts/h0.py ----
import numpy as np

from fhew.channels import ChannelSet
from fhew.likelihood import ChannelHMMFilter, StepResult


class _OccupancyFloor:
    """Order-0 learned floor: a fitted channel-occupancy (unigram) model.

    This is the "fitted low-order Markov" option the paper allows for H_0
    (Section 4.5). Unlike a flat uniform, it captures a biased channel occupancy,
    so an emitter with weak generic structure (e.g. a nonlinear-filtered LFSR)
    is absorbed by H_0 instead of being handed to the order-1 Markov *family*,
    which otherwise over-declares. It stays strictly weaker than the order-1
    Markov hypothesis, so a genuinely Markov emitter is still won by that family.
    """

    def __init__(self, channels: ChannelSet, alpha: float = 1.0):
        self.channels = channels
        self.M = channels.size
        self.alpha = float(alpha)
        self.reset()

    def reset(self) -> None:
        self.counts = np.full(self.M, self.alpha, dtype=np.float64)
        self._dist = self.counts / self.counts.sum()
        self.last_predict = self._dist.copy()

    def step(self, log_emission: np.ndarray) -> StepResult:
        m = float(np.max(log_emission))
        e = np.exp(log_emission - m)
        total = float((self._dist * e).sum())
        loglik = m + float(np.log(total)) if total > 0 else m + float(np.log(np.finfo(float).tiny))
        self.last_predict = self._dist.copy()
        return StepResult(loglik=loglik, predict_next=self._dist.copy())

    def observe_clean(self, channel: int | None) -> None:
        if channel is not None:
            self.counts[channel] += 1.0
            self._dist = self.counts / self.counts.sum()

    def forget(self, retain: float = 0.0) -> None:
        retain = float(np.clip(retain, 0.0, 1.0))
        self.counts = self.alpha + retain * (self.counts - self.alpha)
        self._dist = self.counts / self.counts.sum()
        self.last_predict = self._dist.copy()


class H0Hypothesis(Hypothesis):
    """None-of-the-above class (pluggable likelihood, paper Section 4.5 / 5).

    ``mode``:
        * ``'uniform'``  -- flat (or recce barrage-floor) channel distribution;
        * ``'occupancy'`` -- fitted order-0 (unigram) floor (default; absorbs
          weak generic structure so the Markov *family* does not over-declare);
        * ``'markov'``   -- fitted order-1 Markov floor.
    ``floor='barrage'`` uses the channel plan's effective set for the uniform
    case instead of a flat uniform.
    """

    def __init__(self, channels: ChannelSet, mode: str = "occupancy",
                 floor: str = "uniform", label: str = "H0", alpha: float = 1.0):
        self.label = label
        self.channels = channels
        self.mode = mode
        self.floor_kind = floor
        self._floor = channels.barrage_floor() if floor == "barrage" else channels.uniform()
        self._markov = None
        self._occ = None
        self._filter = None
        if mode == "markov":
            self._markov = MarkovHypothesis(label, channels, alpha=alpha)
        elif mode == "occupancy":
            self._occ = _OccupancyFloor(channels, alpha=alpha)
        else:
            T = np.tile(self._floor, (channels.size, 1))
            self._filter = ChannelHMMFilter(T, init=self._floor)
        self._last_predict = self._floor.copy()

    def _backend(self):
        return self._markov or self._occ or self._filter

    def step(self, log_emission: np.ndarray) -> StepResult:
        res = self._backend().step(log_emission)
        self._last_predict = res.predict_next
        return res

    def observe_clean(self, channel: int | None) -> None:
        backend = self._backend()
        observe = getattr(backend, "observe_clean", None)
        if observe is not None:
            observe(channel)

    def predict_next(self) -> np.ndarray:
        return self._last_predict

    def reset(self) -> None:
        self._backend().reset()
        self._last_predict = self._floor.copy()

    def forget(self, retain: float = 0.0) -> None:
        backend = self._backend()
        forget = getattr(backend, "forget", None)
        if forget is not None:
            forget(retain)
        self._last_predict = self._floor.copy()

    @property
    def n_particles(self) -> int:
        return self.channels.size

# ---- merged from fhew/experts/bank.py ----
from dataclasses import dataclass

import numpy as np
from scipy.special import log_softmax, logsumexp

from fhew.channels import ChannelSet
from fhew.likelihood import EmissionModel, SensorAwareEmission
from fhew.sensor import Observation, SensorModel
from fhew.stopping import MSPRT, StoppingRule

H0_INDEX = 0
UNKNOWN_LABEL = "unknown"


@dataclass
class BankOutput:
    """Everything the system emits at a slot (paper Section 8)."""

    slot: int
    posterior: np.ndarray          # pi_t over {0..K}, index 0 = H_0
    labels: list[str]              # labels aligned with posterior; labels[0] = 'H0'
    next_hop: np.ndarray           # posterior-weighted next-channel distribution
    kappa: float                   # predictability score 1 - H(next_hop)/log M
    declared: bool                 # has the stop rule fired?
    declared_index: int | None     # argmax at declaration (0 => unknown)
    declared_label: str | None     # 'unknown' when H_0 wins
    tau_stop: int | None           # slot of declaration
    stop_score: float              # the stopping statistic this slot

    @property
    def top_index(self) -> int:
        return int(np.argmax(self.posterior))

    @property
    def top_label(self) -> str:
        return self.labels[self.top_index]


class HypothesisBank(Expert):
    """Posterior over ``{H_0, H_1, .., H_K}`` with MSPRT / e-process stopping."""

    label = "bank"

    def __init__(
        self,
        channels: ChannelSet,
        h0: H0Hypothesis,
        hypotheses: list[Hypothesis],
        *,
        emission: EmissionModel | None = None,
        sensor: SensorModel | None = None,
        stopping: StoppingRule | None = None,
        prior: np.ndarray | None = None,
        h0_floor: float = 0.1,
    ):
        if emission is None:
            if sensor is None:
                raise ValueError("pass either an emission model or a sensor")
            emission = SensorAwareEmission(sensor)
        self.channels = channels
        self.emission = emission
        self.stopping = stopping or MSPRT(delta=0.05)
        # Ordered hypothesis list: H_0 first.
        self._hyps: list[Hypothesis] = [h0] + list(hypotheses)
        self.labels = [h.label for h in self._hyps]
        self.labels[H0_INDEX] = h0.label  # typically 'H0'
        self.n = len(self._hyps)
        self._log_prior0 = self._make_prior(prior, h0_floor)
        self.reset()

    # ------------------------------------------------------------------ priors
    def _make_prior(self, prior: np.ndarray | None, h0_floor: float) -> np.ndarray:
        if prior is None:
            p = np.full(self.n, 1.0 / self.n)
        else:
            p = np.asarray(prior, dtype=np.float64)
            if p.shape[0] != self.n:
                raise ValueError("prior length must match number of hypotheses")
            p = p / p.sum()
        # Floor on H_0 so the unknown class is never ruled out at the start.
        if p[H0_INDEX] < h0_floor:
            p[H0_INDEX] = h0_floor
            p = p / p.sum()
        return np.log(p)

    # ------------------------------------------------------------------ Expert
    def reset(self) -> None:
        for h in self._hyps:
            h.reset()
        self.log_prior = self._log_prior0.copy()
        self.cum_loglik = np.zeros(self.n, dtype=np.float64)
        self.t = 0
        self.declared = False
        self.declared_index: int | None = None
        self.tau_stop: int | None = None
        self._stop_score = float("nan")
        self._next_hops = np.tile(self.channels.uniform(), (self.n, 1))

    def forget(self, retain: float = 0.0) -> None:
        """Re-flatten the bank after a regime alarm (paper Section 6.4).

        Shrinks the cumulative evidence toward zero (so the posterior relaxes
        back to the prior and a previously-losing generator can win again) and
        forgets each sub-hypothesis's online-fitted context. Stopping state is
        left untouched; the regime controller decides whether to re-open it.
        """
        retain = float(np.clip(retain, 0.0, 1.0))
        self.cum_loglik *= retain
        for h in self._hyps:
            h.forget(retain)
        self._next_hops = np.tile(self.channels.uniform(), (self.n, 1))

    def update(self, obs: Observation) -> float:
        log_e = self.emission.log_emission(obs)
        prev_logpost = self.log_prior + self.cum_loglik  # unnormalised log-posterior
        clean = self._clean_channel(obs)
        for k, h in enumerate(self._hyps):
            res = h.step(log_e)
            self.cum_loglik[k] += res.loglik
            # Feed online-fitted hypotheses (Markov / learned H_0) *after* scoring.
            observe_clean = getattr(h, "observe_clean", None)
            if observe_clean is not None:
                observe_clean(clean)
            # Read the next-hop AFTER observe_clean: the online predictor experts
            # can only form the genuine P(s_{t+1} | o_{1:t}) once they have ingested
            # this slot's clean channel. Deterministic/Markov/H_0 set predict_next
            # in step() and observe_clean leaves it untouched, so this is correct
            # and uniform across every expert.
            self._next_hops[k] = h.predict_next()
        self.t += 1
        # Bank's one-step mixture predictive log-likelihood (for a log-loss mixer).
        new_logpost = self.log_prior + self.cum_loglik
        bank_loglik = float(logsumexp(new_logpost) - logsumexp(prev_logpost))
        if not self.declared:
            decision = self.stopping.decide(self.log_prior, self.cum_loglik)
            self._stop_score = decision.score
            if decision.stop:
                self.declared = True
                self.declared_index = decision.k_star
                self.tau_stop = self.t
        return bank_loglik

    def predict(self) -> np.ndarray:
        """Posterior-weighted next-hop distribution (paper Section 8)."""
        post = self.posterior()
        p_hat = post @ self._next_hops
        s = p_hat.sum()
        return p_hat / s if s > 0 else self.channels.uniform()

    @property
    def hypotheses(self) -> list[Hypothesis]:
        """The bank's hypotheses aligned with :attr:`labels` (index 0 = H_0)."""
        return list(self._hyps)

    # --------------------------------------------------------------- posterior
    def posterior(self) -> np.ndarray:
        return np.exp(log_softmax(self.log_prior + self.cum_loglik))

    def log_posterior(self) -> np.ndarray:
        return log_softmax(self.log_prior + self.cum_loglik)

    def predictability(self) -> float:
        """kappa = 1 - H(p_hat)/log M, in [0, 1] (paper Section 4.7)."""
        p = self.predict()
        nz = p[p > 0]
        H = float(-(nz * np.log(nz)).sum())
        return float(1.0 - H / np.log(self.channels.size))

    # ------------------------------------------------------------------ output
    def output(self) -> BankOutput:
        post = self.posterior()
        declared_label = None
        if self.declared and self.declared_index is not None:
            declared_label = (UNKNOWN_LABEL if self.declared_index == H0_INDEX
                              else self.labels[self.declared_index])
        return BankOutput(
            slot=self.t,
            posterior=post,
            labels=list(self.labels),
            next_hop=self.predict(),
            kappa=self.predictability(),
            declared=self.declared,
            declared_index=self.declared_index,
            declared_label=declared_label,
            tau_stop=self.tau_stop,
            stop_score=self._stop_score,
        )

    def declared_identity(self) -> str | None:
        if not self.declared:
            return None
        return (UNKNOWN_LABEL if self.declared_index == H0_INDEX
                else self.labels[self.declared_index])

    @staticmethod
    def _clean_channel(obs: Observation) -> int | None:
        if len(obs.detections) == 1:
            return obs.detections[0].channel
        return None

# ---- the Fixed-Share expert mixture: the next-channel predictor (thesis §4.2) ----


class FixedShareMixture(Expert):
    r"""Weights every expert by its *recent* next-channel predictive skill.

    This is the thesis Section 4.2 combiner, and the object the system uses to
    **predict the next channel**. It differs from :class:`HypothesisBank` in what
    it optimises: the bank accumulates likelihood to *identify* a generator and
    can only predict well once an in-bank hypothesis wins; the mixture instead
    tracks whichever expert is predicting best *right now*, so it keeps producing
    a sharp next-hop distribution even when no identification expert matches ---
    the universal (context-mixing) expert then carries the prediction.

    Weights update multiplicatively on each expert's per-slot sensor-aware
    predictive log-likelihood ``\ell_t(i)`` (information-parity safe: it is the
    log-likelihood of the *observation*, never the hidden channel), followed by a
    Fixed-Share floor so a newly-relevant expert recovers in a logarithmic number
    of steps after a regime switch:

        w_{t+1}(i) \propto w_t(i) \exp(\eta\,\ell_t(i)),
        w_{t+1} \leftarrow (1-\alpha)\,w_{t+1} + \alpha/N.

    The per-slot additive constant that the reduced emission drops is identical
    across experts, so it cancels in the weight normalisation; the mixture is
    therefore invariant to it.
    """

    label = "mixture"

    def __init__(
        self,
        channels: ChannelSet,
        experts: list[Hypothesis],
        *,
        emission: EmissionModel | None = None,
        sensor: SensorModel | None = None,
        eta: float = 1.0,
        alpha: float = 0.02,
        labels: list[str] | None = None,
    ):
        if not experts:
            raise ValueError("FixedShareMixture needs at least one expert")
        if emission is None:
            if sensor is None:
                raise ValueError("pass either an emission model or a sensor")
            emission = SensorAwareEmission(sensor)
        if not 0.0 <= alpha < 1.0:
            raise ValueError("alpha (share floor) must be in [0, 1)")
        if eta <= 0.0:
            raise ValueError("eta must be positive")
        self.channels = channels
        self.M = channels.size
        self.emission = emission
        self.experts = list(experts)
        self.n = len(self.experts)
        self.eta = float(eta)
        self.alpha = float(alpha)
        self.labels = list(labels) if labels is not None else [
            getattr(h, "label", f"expert{i}") for i, h in enumerate(self.experts)
        ]
        self.reset()

    # ------------------------------------------------------------------ Expert
    def reset(self) -> None:
        for h in self.experts:
            h.reset()
        self.log_w = np.full(self.n, -np.log(self.n))
        self.t = 0
        self._next_hops = np.tile(self.channels.uniform(), (self.n, 1))
        self._last_predict = self.channels.uniform()

    def update(self, obs: Observation) -> float:
        log_e = self.emission.log_emission(obs)
        clean = self._clean_channel(obs)
        lls = np.empty(self.n, dtype=np.float64)
        for k, h in enumerate(self.experts):
            res = h.step(log_e)
            lls[k] = res.loglik
            observe_clean = getattr(h, "observe_clean", None)
            if observe_clean is not None:
                observe_clean(clean)
            # read the next-hop AFTER observe_clean so online predictors surface
            # the genuine P(s_{t+1} | o_{1:t}) (same convention as HypothesisBank).
            self._next_hops[k] = h.predict_next()
        # mixture predictive log-likelihood for this slot (before the weight step)
        mix_loglik = float(logsumexp(self.log_w + lls))
        # Fixed-Share weight update (multiplicative weights + share floor).
        a = self.log_w + self.eta * lls
        a -= logsumexp(a)
        w = np.exp(a)
        w = (1.0 - self.alpha) * w + self.alpha / self.n
        self.log_w = np.log(w)
        self.t += 1
        self._last_predict = self._mix_next_hop()
        return mix_loglik

    def _mix_next_hop(self) -> np.ndarray:
        w = np.exp(self.log_w)
        p = w @ self._next_hops
        s = p.sum()
        return p / s if s > 0 else self.channels.uniform()

    def predict(self) -> np.ndarray:
        """The posterior-over-experts weighted next-channel distribution."""
        return self._last_predict.copy()

    # ------------------------------------------------------------- prediction API
    def predict_topk(self, k: int = 1) -> list[int]:
        """The ``k`` most probable next channels, most likely first."""
        p = self.predict()
        return [int(c) for c in np.argsort(p)[::-1][:k]]

    def predict_argmax(self) -> int:
        return int(np.argmax(self.predict()))

    def weights(self) -> np.ndarray:
        return np.exp(self.log_w)

    def dominant_label(self) -> str:
        """Label of the expert currently carrying the most weight (diagnostic)."""
        return self.labels[int(np.argmax(self.log_w))]

    def predictability(self) -> float:
        """kappa = 1 - H(predict)/log M, in [0, 1]: how sharp the prediction is."""
        p = self.predict()
        nz = p[p > 0]
        H = float(-(nz * np.log(nz)).sum())
        return float(1.0 - H / np.log(self.M))

    def forget(self, retain: float = 0.0) -> None:
        """Flush the mixture after a regime alarm (thesis Section 6.4).

        Re-flattens the expert weights toward uniform and forgets each expert's
        learned context (every expert implements ``forget``). The share floor
        alone lets a dead expert recover, but the universal predictor's context
        tables are still full of the *old* generator's patterns; forgetting them
        is what makes the re-point fast rather than drawn out.
        """
        retain = float(np.clip(retain, 0.0, 1.0))
        for h in self.experts:
            h.forget(retain)
        self.log_w = retain * self.log_w + (1.0 - retain) * np.full(self.n, -np.log(self.n))
        self.log_w -= logsumexp(self.log_w)
        self._next_hops = np.tile(self.channels.uniform(), (self.n, 1))
        self._last_predict = self.channels.uniform()

    @staticmethod
    def _clean_channel(obs: Observation) -> int | None:
        if len(obs.detections) == 1:
            return obs.detections[0].channel
        return None
