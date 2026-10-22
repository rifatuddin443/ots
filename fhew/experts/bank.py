"""The sensor-aware Bayesian hypothesis bank with anytime stopping.

The bank keeps a posterior over all hypotheses -- the ``K`` named generators
plus the mandatory none-of-the-above class ``H_0`` at index 0 -- and updates it
after every slot (paper Section 7):

    pi_t(k)  propto  pi_{t-1}(k) * P(o_t | o_{1:t-1}, H_k),    k in {0, 1, .., K}.

It implements the :class:`~fhew.experts.base.Expert` interface, so in the thesis
it drops into the Fixed-Share mixture as one expert. Three switches carry the
contribution and make the ablation table one codebase, not several: the
``emission`` strategy (sensor-aware vs naive), the presence of ``H_0``, and the
``stopping`` rule (MSPRT vs e-process).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import log_softmax, logsumexp

from fhew.channels import ChannelSet
from fhew.experts.base import Expert
from fhew.experts.h0 import H0Hypothesis
from fhew.experts.hypothesis import Hypothesis
from fhew.likelihood.emission import EmissionModel, SensorAwareEmission
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
