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

# ---- merged from fhew/experts/hypothesis.py ----
import abc
from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp

from fhew.channels import ChannelSet
from fhew.generators import make_generator
from fhew.generators import WeakSeedGenerator
from fhew.generators import _PRIMITIVE_TAPS, _bits_per_channel
from fhew.likelihood import (
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

    def update(self, obs: Observation) -> float:
        log_e = self.emission.log_emission(obs)
        prev_logpost = self.log_prior + self.cum_loglik  # unnormalised log-posterior
        clean = self._clean_channel(obs)
        for k, h in enumerate(self._hyps):
            res = h.step(log_e)
            self.cum_loglik[k] += res.loglik
            self._next_hops[k] = res.predict_next
            # Feed online-fitted hypotheses (Markov / learned H_0) *after* scoring.
            observe_clean = getattr(h, "observe_clean", None)
            if observe_clean is not None:
                observe_clean(clean)
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
