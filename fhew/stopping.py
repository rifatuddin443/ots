"""Stopping rules: MSPRT and the anytime-valid e-process (paper Section 7).

The bank keeps, for every hypothesis ``k`` in ``{0, 1, .., K}``, a cumulative
*reduced* log-likelihood ``Lambda_t(k) = sum_s loglik_k(s)`` (the common
all-false-alarm constant has already cancelled). Both stopping rules read from
``(log_prior, cum_loglik)``:

    * :class:`MSPRT` declares at the first slot where the top posterior crosses
      ``1 - delta``. This is the multi-hypothesis SPRT, asymptotically optimal
      as ``delta`` shrinks.
    * :class:`EProcess` declares when a hypothesis's e-value crosses ``1/beta``.
      Ville's inequality then bounds the probability of declaring it over any
      given in-bank competitor by ``beta`` *whenever* the run stops -- the right
      object for a short, variable-length engagement window.
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

    For hypothesis ``k`` the default e-value is against its strongest competitor

        E_t(k) = p_k(o_{1:t}) / max_{j != k} p_j(o_{1:t}).

    If the truth is ``j != k`` then ``E_t(k) <= p_k / p_j``, a non-negative
    martingale with mean 1 under ``H_j``, so by Ville's inequality
    ``P(exists t: E_t(k) >= 1/beta) <= beta`` against *each* in-bank competitor,
    whatever its prior weight. (Each ``p_j`` marginalises ``H_j``'s own seeds,
    so "under ``H_j``" means a seed drawn from the bank's prior within ``j``.)
    The chance of declaring *some* wrong hypothesis is at most ``(K-1) beta`` by
    the union bound. We declare the ``argmax_k E_t(k)`` once it crosses
    ``1/beta``.

    ``against="mixture"`` restores the older, faster-stopping e-value against the
    prior-weighted mixture of competitors,
    ``p_k / sum_{j != k} rho_j p_j`` with ``rho_j = pi_0(j) / (1 - pi_0(k))``,
    whose ``beta`` bound holds only when the truth is drawn from that mixture;
    against one competitor ``j`` it weakens to ``beta / rho_j``.
    """

    name = "eprocess"

    def __init__(self, beta: float = 0.05, against: str = "max"):
        if not 0.0 < beta < 1.0:
            raise ValueError("beta must be in (0, 1)")
        if against not in ("max", "mixture"):
            raise ValueError("against must be 'max' or 'mixture'")
        self.beta = float(beta)
        self.against = against
        self.log_threshold = float(np.log(1.0 / beta))

    def log_evalues(self, log_prior: np.ndarray, cum_loglik: np.ndarray) -> np.ndarray:
        K = cum_loglik.shape[0]
        log_e = np.full(K, -np.inf)
        for k in range(K):
            mask = np.arange(K) != k
            if self.against == "max":
                denom = np.max(cum_loglik[mask])
            else:
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
