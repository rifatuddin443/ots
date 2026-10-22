"""Shared likelihood driver for the SPRT / fixed-length baselines.

Both baselines run the *same* hypotheses with the *same* sensor-aware emission
as the bank, so they see identical evidence. Only the decision rule differs.
This driver keeps the per-hypothesis cumulative reduced log-likelihood and the
next-hop predictives, exactly like the bank but without H_0 or a stop rule.
"""

from __future__ import annotations

import numpy as np

from fhew.experts.hypothesis import Hypothesis
from fhew.likelihood.emission import EmissionModel
from fhew.sensor import Observation


class LikelihoodTracker:
    """Drives a list of hypotheses and accumulates reduced log-likelihoods."""

    def __init__(self, hypotheses: list[Hypothesis], emission: EmissionModel,
                 n_channels: int):
        self.hyps = list(hypotheses)
        self.labels = [h.label for h in self.hyps]
        self.emission = emission
        self.n = len(self.hyps)
        self.n_channels = n_channels
        self.reset()

    def reset(self) -> None:
        for h in self.hyps:
            h.reset()
        self.cum_loglik = np.zeros(self.n, dtype=np.float64)
        self.t = 0
        self._next_hops = np.tile(np.full(self.n_channels, 1.0 / self.n_channels),
                                  (self.n, 1))

    def update(self, obs: Observation) -> None:
        log_e = self.emission.log_emission(obs)
        clean = obs.detections[0].channel if len(obs.detections) == 1 else None
        for k, h in enumerate(self.hyps):
            res = h.step(log_e)
            self.cum_loglik[k] += res.loglik
            self._next_hops[k] = res.predict_next
            observe_clean = getattr(h, "observe_clean", None)
            if observe_clean is not None:
                observe_clean(clean)
        self.t += 1

    def map_index(self) -> int:
        return int(np.argmax(self.cum_loglik))

    def next_hop(self, index: int) -> np.ndarray:
        return self._next_hops[index]
