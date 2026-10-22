"""Fixed-length likelihood-ratio test.

Observe a fixed number of slots ``n_fixed``, then declare the maximum-likelihood
hypothesis. No early stopping and no none-of-the-above class: the simplest
non-sequential baseline, which pays the full observation cost on every episode
regardless of how quickly the evidence resolves (paper Section 11).
"""

from __future__ import annotations

from fhew.baselines.common import LikelihoodTracker
from fhew.experts.hypothesis import Hypothesis
from fhew.likelihood.emission import EmissionModel
from fhew.sensor import Observation


class FixedLengthTest:
    label = "fixed_length"

    def __init__(self, hypotheses: list[Hypothesis], emission: EmissionModel,
                 n_channels: int, n_fixed: int = 30):
        self.tracker = LikelihoodTracker(hypotheses, emission, n_channels)
        self.labels = self.tracker.labels
        self.n_fixed = int(n_fixed)
        self.reset()

    def reset(self) -> None:
        self.tracker.reset()
        self.declared = False
        self.declared_index: int | None = None
        self.tau_stop: int | None = None

    def update(self, obs: Observation) -> None:
        if self.tracker.t >= self.n_fixed:
            return
        self.tracker.update(obs)
        if self.tracker.t >= self.n_fixed and not self.declared:
            self.declared = True
            self.declared_index = self.tracker.map_index()
            self.tau_stop = self.n_fixed

    def declared_identity(self) -> str | None:
        if self.declared_index is None:
            return None
        return self.labels[self.declared_index]
