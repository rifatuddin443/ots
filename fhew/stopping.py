"""Stopping rules: MSPRT and the anytime-valid e-process (paper Section 7).

The bank keeps, for every hypothesis ``k`` in ``{0, 1, .., K}``, a cumulative
*reduced* log-likelihood ``Lambda_t(k) = sum_s loglik_k(s)`` (the common
all-false-alarm constant has already cancelled). Both stopping rules read from
``(log_prior, cum_loglik)``:

    * :class:`MSPRT` declares at the first slot where the top posterior crosses
      ``1 - delta``. This is the multi-hypothesis SPRT, asymptotically optimal
      as ``delta`` shrinks.
    * :class:`EProcess` declares when a hypothesis's e-value crosses ``1/beta``.
      Ville's inequality then bounds the false-identification probability by
      ``beta`` *whenever* the run stops -- the right object for a short,
      variable-length engagement window.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np
from scipy.special import log_softmax, logsumexp


@dataclass
class StopDecision:
    stop: bool
    k_star: int | None
    score: float          # the quantity compared against the threshold
    threshold: float


class StoppingRule(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def decide(self, log_prior: np.ndarray, cum_loglik: np.ndarray) -> StopDecision:
        ...


class MSPRT(StoppingRule):
    """Declare when the top posterior exceeds ``1 - delta``."""

    name = "msprt"

    def __init__(self, delta: float = 0.05):
        if not 0.0 < delta < 1.0:
            raise ValueError("delta must be in (0, 1)")
        self.delta = float(delta)

    def posterior(self, log_prior: np.ndarray, cum_loglik: np.ndarray) -> np.ndarray:
        return np.exp(log_softmax(log_prior + cum_loglik))

    def decide(self, log_prior: np.ndarray, cum_loglik: np.ndarray) -> StopDecision:
        post = self.posterior(log_prior, cum_loglik)
        k = int(np.argmax(post))
        top = float(post[k])
        return StopDecision(stop=top >= 1.0 - self.delta, k_star=k, score=top,
                            threshold=1.0 - self.delta)


class EProcess(StoppingRule):
    r"""Anytime-valid e-process stopping.

    For hypothesis ``k`` the e-value against the mixture of its competitors is

        E_t(k) = p_k(o_{1:t}) / sum_{j != k} rho_j p_j(o_{1:t}),
        rho_j = pi_0(j) / (1 - pi_0(k)).

    ``E_t(k)`` is a non-negative martingale under the mixture-of-competitors
    null, so by Ville's inequality ``P(exists t: E_t(k) >= 1/beta) <= beta`` when
    that null holds. We declare the ``argmax_k E_t(k)`` once it crosses
    ``1/beta``. The guarantee is stated honestly against the mixture null; the
    per-generator false-identification rate is validated empirically against the
    nominal ``beta`` line (paper Section 10).
    """

    name = "eprocess"

    def __init__(self, beta: float = 0.05):
        if not 0.0 < beta < 1.0:
            raise ValueError("beta must be in (0, 1)")
        self.beta = float(beta)
        self.log_threshold = float(np.log(1.0 / beta))

    def log_evalues(self, log_prior: np.ndarray, cum_loglik: np.ndarray) -> np.ndarray:
        K = cum_loglik.shape[0]
        log_e = np.full(K, -np.inf)
        for k in range(K):
            mask = np.arange(K) != k
            # rho_j = pi_0(j) / (1 - pi_0(k)); in log: log_prior[j] - log(1-pi0_k)
            log_pi0_k = log_prior[k]
            log_1m = np.log1p(-min(np.exp(log_pi0_k), 1 - 1e-12))
            log_rho = log_prior[mask] - log_1m
            denom = logsumexp(log_rho + cum_loglik[mask])
            log_e[k] = cum_loglik[k] - denom
        return log_e

    def decide(self, log_prior: np.ndarray, cum_loglik: np.ndarray) -> StopDecision:
        log_e = self.log_evalues(log_prior, cum_loglik)
        k = int(np.argmax(log_e))
        return StopDecision(stop=log_e[k] >= self.log_threshold, k_star=k,
                            score=float(log_e[k]), threshold=self.log_threshold)


def make_stopping_rule(kind: str, **kwargs) -> StoppingRule:
    kind = kind.lower()
    if kind in ("msprt", "posterior"):
        return MSPRT(**kwargs)
    if kind in ("eprocess", "e-process", "ville"):
        return EProcess(**kwargs)
    raise ValueError(f"unknown stopping rule '{kind}'")
