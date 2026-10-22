"""The mandatory none-of-the-above hypothesis H_0 (paper Section 5).

H_0 keeps the bank from naming the wrong generator with false confidence when
the true one is absent (an out-of-bank generator, or a keyed CSPRNG). Its
likelihood is flat or learned -- uniform over channels, the reconnaissance
barrage floor, or a fitted low-order Markov model -- and it is made *pluggable*
(design rule 4, Section 12) so the journal can swap in a learned open-set score
without touching the bank. When H_0 wins, the declared outcome is "unknown
generator", which is a correct and useful result, not a failure.
"""

from __future__ import annotations

import numpy as np

from fhew.channels import ChannelSet
from fhew.experts.hypothesis import Hypothesis, MarkovHypothesis
from fhew.likelihood.forward import ChannelHMMFilter, StepResult


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
