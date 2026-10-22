"""Sensor-aware likelihood: emission strategies and the forward filter.

This module merges what used to be the fhew/likelihood/ subpackage; the public
names are unchanged, so `from fhew.likelihood import X` still works.
"""

from __future__ import annotations

# ---- merged from fhew/likelihood/emission.py ----
import abc

import numpy as np

from fhew.sensor import Observation, SensorModel


class EmissionModel(abc.ABC):
    """Maps an observation to a reduced log-emission vector over channels."""

    name: str = "base"

    @abc.abstractmethod
    def log_emission(self, obs: Observation) -> np.ndarray:
        ...


class SensorAwareEmission(EmissionModel):
    """The method: emission built from the sensor's ``P_d`` and ``P_fa``.

    Gaps (missed hops) and phantoms (false alarms) are handled inside the
    likelihood instead of corrupting it. Delegates to
    :meth:`SensorModel.emission_loglikelihood`.
    """

    name = "sensor_aware"

    def __init__(self, sensor: SensorModel):
        self.sensor = sensor

    def log_emission(self, obs: Observation) -> np.ndarray:
        return self.sensor.emission_loglikelihood(obs)


class NaiveEmission(EmissionModel):
    """The baseline: ignore ``P_d`` / ``P_fa`` and score only clean matches.

    A channel that produced a detection scores ``log 1 = 0``; a channel with no
    detection scores ``log(eps_miss)``. There is no false-alarm discounting, so a
    missed true hop is (wrongly) treated as evidence against the true generator
    and a phantom as evidence for a wrong one. The gap to
    :class:`SensorAwareEmission`, measured as hops-to-identify and false-ID rate,
    is the ablation that justifies the sensor-aware approach.
    """

    name = "naive"

    def __init__(self, eps_miss: float = 1e-3):
        if not 0.0 < eps_miss < 1.0:
            raise ValueError("eps_miss must be in (0, 1)")
        self.log_eps = float(np.log(eps_miss))

    def log_emission(self, obs: Observation) -> np.ndarray:
        detected = obs.detection_mask()
        log_e = np.full(obs.n_channels, self.log_eps, dtype=np.float64)
        log_e[detected] = 0.0
        return log_e


def make_emission(kind: str, sensor: SensorModel, **kwargs) -> EmissionModel:
    """Factory used by configs: ``'sensor_aware'`` or ``'naive'``."""
    if kind == "sensor_aware":
        return SensorAwareEmission(sensor)
    if kind == "naive":
        return NaiveEmission(**kwargs)
    raise ValueError(f"unknown emission kind '{kind}'")

# ---- merged from fhew/likelihood/forward.py ----
import abc
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp


@dataclass
class StepResult:
    """One slot's forward-filter output."""

    loglik: float               # reduced log P(o_t | H_k, o_{1:t-1})
    predict_next: np.ndarray    # P(s_{t+1} = c | o_{1:t}) over channels


class ForwardFilter(abc.ABC):
    """Common interface for the two filter flavours."""

    n_channels: int

    @abc.abstractmethod
    def step(self, log_emission: np.ndarray) -> StepResult:
        """Consume one slot's reduced log-emission; return loglik + next-hop."""

    @abc.abstractmethod
    def predict_current(self) -> np.ndarray:
        """Prior over the *current* slot's true channel, before observing it."""

    @abc.abstractmethod
    def reset(self) -> None:
        ...


class DeterministicTrajectoryFilter(ForwardFilter):
    """Particle filter over candidate deterministic trajectories.

    ``trajectories`` is a ``(K, horizon)`` int array: row ``j`` is candidate
    ``j``'s channel at each slot. ``log_w0`` is the prior over candidates
    (defaults to uniform). The hidden state is the candidate index; the forward
    pass marginalises over it exactly. Used for LFSR state enumeration and
    LCG / weak-seed searches.
    """

    def __init__(self, trajectories: np.ndarray, n_channels: int,
                 log_w0: np.ndarray | None = None):
        self.traj = np.ascontiguousarray(trajectories, dtype=np.int64)
        if self.traj.ndim != 2:
            raise ValueError("trajectories must be a (K, horizon) array")
        self.K, self.horizon = self.traj.shape
        self.n_channels = int(n_channels)
        if log_w0 is None:
            log_w0 = np.full(self.K, -np.log(self.K))
        self._log_w0 = log_w0 - logsumexp(log_w0)
        self.reset()

    def reset(self) -> None:
        self.t = 0
        self.log_w = self._log_w0.copy()

    def predict_current(self) -> np.ndarray:
        return self._weighted_channel_dist(self.t, self.log_w)

    def _weighted_channel_dist(self, t: int, log_w: np.ndarray) -> np.ndarray:
        if t >= self.horizon:
            # Beyond the precomputed horizon: fall back to the marginal channel
            # usage of the candidates at the last slot (rarely hit; the episode
            # window T sizes the horizon).
            t = self.horizon - 1
        dist = np.zeros(self.n_channels, dtype=np.float64)
        w = np.exp(log_w)
        np.add.at(dist, self.traj[:, t], w)
        s = dist.sum()
        return dist / s if s > 0 else np.full(self.n_channels, 1.0 / self.n_channels)

    def step(self, log_emission: np.ndarray) -> StepResult:
        if self.t >= self.horizon:
            # No trajectory information left; emit a flat predictive and a
            # neutral likelihood (log of the uniform-mix emission).
            loglik = float(logsumexp(log_emission) - np.log(self.n_channels))
            return StepResult(loglik=loglik,
                              predict_next=np.full(self.n_channels, 1.0 / self.n_channels))
        cur = self.traj[:, self.t]                      # (K,) channel per candidate
        a = self.log_w + log_emission[cur]              # log(w_j * ratio_j)
        loglik = float(logsumexp(a))                    # log sum w_j ratio_j (w normalised)
        self.log_w = a - loglik                         # posterior over candidates
        self.t += 1
        predict_next = self._weighted_channel_dist(self.t, self.log_w)
        return StepResult(loglik=loglik, predict_next=predict_next)


class ChannelHMMFilter(ForwardFilter):
    """Channel-level HMM forward filter for Markov / n-gram and the H_0 floor.

    The hidden state is the current true channel ``s_t`` with transition matrix
    ``T`` (row-stochastic, ``T[i, j] = P(s_{t+1}=j | s_t=i)``) and initial
    distribution ``init``. For the uniform H_0 every row of ``T`` is uniform, so
    the predictive is always uniform.
    """

    def __init__(self, transition: np.ndarray, init: np.ndarray | None = None):
        self.T = np.asarray(transition, dtype=np.float64)
        if self.T.ndim != 2 or self.T.shape[0] != self.T.shape[1]:
            raise ValueError("transition must be a square matrix")
        self.n_channels = self.T.shape[0]
        row = self.T.sum(axis=1, keepdims=True)
        self.T = np.divide(self.T, row, out=np.full_like(self.T, 1.0 / self.n_channels),
                           where=row > 0)
        if init is None:
            init = np.full(self.n_channels, 1.0 / self.n_channels)
        self._init = np.asarray(init, dtype=np.float64)
        self._init = self._init / self._init.sum()
        self.reset()

    def reset(self) -> None:
        self.prior = self._init.copy()   # distribution over current slot's s_t

    def predict_current(self) -> np.ndarray:
        return self.prior.copy()

    def step(self, log_emission: np.ndarray) -> StepResult:
        m = float(np.max(log_emission))
        e = np.exp(log_emission - m)
        weighted = self.prior * e
        total = float(weighted.sum())
        if total <= 0.0:
            # Degenerate slot; keep belief, emit flat predictive.
            loglik = m + np.log(np.finfo(float).tiny)
            return StepResult(loglik=loglik, predict_next=self.T.T @ self.prior)
        loglik = m + float(np.log(total))     # reduced log L_t (prior sums to 1)
        post = weighted / total
        predict_next = self.T.T @ post         # P(s_{t+1} | o_{1:t})
        self.prior = predict_next
        return StepResult(loglik=loglik, predict_next=predict_next)
