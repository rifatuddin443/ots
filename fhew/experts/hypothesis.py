"""Hypotheses: one generator family each, built from the reconnaissance prior.

A hypothesis is a class with named parameters and two jobs (paper Section 5):
score how well observed hops match it (the per-slot likelihood) and predict the
next-channel distribution. Both run through the shared forward filter, so gaps
and phantoms are handled inside the likelihood.

Crucially, a hypothesis is built from the recce *ranges* (which lengths, which
keyspace), never from the true generator instance. The candidate enumeration
*is* the sensor-aware, gap-robust "state-filtered Berlekamp-Massey": it keeps a
posterior over register states through missed and false detections, instead of
needing clean consecutive outputs the way plain BM does.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp

from fhew.channels import ChannelSet
from fhew.generators import make_generator
from fhew.generators.lcg import WeakSeedGenerator
from fhew.generators.lfsr import _PRIMITIVE_TAPS, _bits_per_channel
from fhew.likelihood.forward import (
    ChannelHMMFilter,
    DeterministicTrajectoryFilter,
    StepResult,
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


_TRAJ_BUILDERS = {
    "lfsr": _lfsr_trajectories,
    "weak_seed": _weak_seed_trajectories,
    "lcg": _lcg_trajectories,
}


def build_hypothesis(spec: HypothesisSpec, channels: ChannelSet, horizon: int,
                     rng: np.random.Generator | None = None) -> Hypothesis:
    """Instantiate a hypothesis from its spec and the recce ranges."""
    if spec.kind == "uniform":
        from fhew.experts.h0 import H0Hypothesis
        return H0Hypothesis(channels, mode="uniform",
                            floor=spec.options.get("floor", "uniform"), label=spec.label)
    if spec.kind == "markov":
        return MarkovHypothesis(spec.label, channels, alpha=spec.options.get("alpha", 1.0))
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
