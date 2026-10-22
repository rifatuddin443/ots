"""Pairwise SPRT generalised to a bank by a leader-vs-all rule.

The classic SPRT is a two-hypothesis test; with ``K`` candidates the usual
generalisation declares hypothesis ``k`` once its cumulative log-likelihood
ratio against *every* competitor clears Wald's upper bound
``A = log((1 - beta) / alpha)``. This is a fair baseline -- same data, same
sensor-aware likelihoods -- whose one structural difference from the bank is the
absence of a none-of-the-above class. On an out-of-bank or crypto emitter it is
therefore forced to name a generator, which is exactly where it fails (paper
Section 11, item 4).
"""

from __future__ import annotations

import numpy as np

from fhew.baselines.common import LikelihoodTracker
from fhew.experts.hypothesis import Hypothesis
from fhew.likelihood.emission import EmissionModel
from fhew.sensor import Observation


class PairwiseSPRT:
    label = "pairwise_sprt"

    def __init__(self, hypotheses: list[Hypothesis], emission: EmissionModel,
                 n_channels: int, alpha: float = 0.05, beta: float = 0.05):
        self.tracker = LikelihoodTracker(hypotheses, emission, n_channels)
        self.labels = self.tracker.labels
        # Wald's A bound on the log-likelihood-ratio for a target error.
        self.A = float(np.log((1.0 - beta) / alpha))
        self.reset()

    def reset(self) -> None:
        self.tracker.reset()
        self.declared = False
        self.declared_index: int | None = None
        self.tau_stop: int | None = None

    def update(self, obs: Observation) -> None:
        self.tracker.update(obs)
        if self.declared:
            return
        cl = self.tracker.cum_loglik
        k = int(np.argmax(cl))
        # leader's smallest margin over any competitor
        margins = cl[k] - cl
        margins[k] = np.inf
        if float(margins.min()) >= self.A:
            self.declared = True
            self.declared_index = k
            self.tau_stop = self.tracker.t

    def declared_identity(self) -> str | None:
        if self.declared_index is None:
            return None
        return self.labels[self.declared_index]

    def map_identity(self) -> str:
        return self.labels[self.tracker.map_index()]
