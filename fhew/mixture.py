"""The Fixed-Share expert mixture (thesis Section 4.2).

Layer 2 never classifies the adversary and commits. The Bayesian hypothesis
bank (``fhew.experts.HypothesisBank``) weights known generators by *cumulative*
evidence, so once a generator wins the bank stays stuck on it -- which is
exactly wrong when the adversary changes its generator mid-engagement. The
Fixed-Share mixture weights every expert by its *recent* predictive skill and
keeps a share floor ``alpha > 0`` on every weight, so a "dead" expert can come
back to life within a few hops after a switch. The bank enters this mixture as
*one* expert (``HypothesisBank`` already implements the :class:`Expert`
interface), so it is weighted by realised skill like any other.

Weights update multiplicatively on the one-step log-loss (paper eq. in 4.2):

    w_{t+1}(i) ∝ w_t(i) · exp(-η · ℓ_t(i)),     ℓ_t(i) = -log p_i(c_t | H_{t-1})

then Fixed-Share mixes a fraction ``alpha`` of the mass back to uniform:

    w_{t+1}(i) ← (1 - alpha) · w_{t+1}(i) + alpha / N.

The share floor is what gives the *tracking* regret bound against the best
segmentation of the engagement into ``k`` regimes -- and, operationally, what
lets the mixture re-point itself onto the expert that fits the new generator
with no manual switching (thesis Section 4.1 / 4.2).

This module is deliberately additive: it consumes the conference-slice experts
unchanged and is in turn consumed by :mod:`fhew.regime`, which wires a
change-detector to :meth:`FixedShareMixture.on_regime_alarm` so a switch flushes
the universal predictor's context, not only the mixture weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp

from fhew.channels import ChannelSet
from fhew.experts import Expert, Hypothesis
from fhew.likelihood import EmissionModel, SensorAwareEmission
from fhew.sensor import Observation, SensorModel


def _clean_channel(obs: Observation) -> int | None:
    """The slot's channel iff it has exactly one detection (the clean surrogate
    for the true channel used for prequential scoring; same rule as the bank)."""
    if len(obs.detections) == 1:
        return obs.detections[0].channel
    return None


class HypothesisExpert(Expert):
    """Adapter: expose a single :class:`~fhew.experts.Hypothesis` as an Expert.

    The standalone experts (context-mixing, Markov, phase-space, ...) implement
    the ``Hypothesis`` interface (``step`` on a reduced log-emission +
    ``observe_clean``). The mixture wants the ``Expert`` interface
    (``update(obs)`` + ``predict()``). This adapter supplies the emission model
    and threads the clean channel through, so any hypothesis can be a mixture
    member next to the whole :class:`~fhew.experts.HypothesisBank`.
    """

    def __init__(self, hyp: Hypothesis, emission: EmissionModel, channels: ChannelSet):
        self.hyp = hyp
        self.emission = emission
        self.channels = channels
        self.label = hyp.label

    def reset(self) -> None:
        self.hyp.reset()

    def update(self, obs: Observation) -> float:
        log_e = self.emission.log_emission(obs)
        res = self.hyp.step(log_e)
        observe_clean = getattr(self.hyp, "observe_clean", None)
        if observe_clean is not None:
            observe_clean(_clean_channel(obs))
        return float(res.loglik)

    def predict(self) -> np.ndarray:
        return self.hyp.predict_next()

    def forget(self, retain: float = 0.0) -> None:
        self.hyp.forget(retain)


@dataclass
class MixtureStep:
    """One slot's mixture diagnostics."""

    loglik: float                     # mixture one-step predictive log-lik of c_t
    surprise: float                   # -loglik (the loss stream the monitor reads)
    weights: np.ndarray               # post-update expert weights
    member_losses: np.ndarray         # per-expert one-step log-loss (nan if no c_t)
    scored: bool                      # was there a clean channel to score on?


class FixedShareMixture(Expert):
    """Fixed-Share mixture of experts over the next channel (paper Section 4.2).

    Parameters
    ----------
    members:
        The experts to combine. Each must implement :class:`Expert`. Wrap a bare
        hypothesis with :class:`HypothesisExpert` first.
    eta:
        Learning rate in the multiplicative weight update.
    share:
        The share floor ``alpha`` in ``[0, 1)``. Strictly positive is what lets
        a dead expert recover after a switch; ``0`` reduces to plain
        exponential-weights (Hedge) and gets stuck, which the ablation reports.
    channels:
        The channel set (for the uniform fallback).
    """

    label = "mixture"

    def __init__(self, members: list[Expert], *, channels: ChannelSet,
                 eta: float = 0.5, share: float = 0.02):
        if not members:
            raise ValueError("mixture needs at least one member expert")
        if not 0.0 <= share < 1.0:
            raise ValueError("share floor alpha must be in [0, 1)")
        self.members = list(members)
        self.labels = [m.label for m in self.members]
        self.N = len(self.members)
        self.eta = float(eta)
        self.share = float(share)
        self.channels = channels
        self.reset()

    # ------------------------------------------------------------------ Expert
    def reset(self) -> None:
        for m in self.members:
            m.reset()
        self.w = np.full(self.N, 1.0 / self.N)
        self.t = 0
        self._last = MixtureStep(float("nan"), float("nan"), self.w.copy(),
                                 np.full(self.N, np.nan), False)

    def _member_preds(self) -> np.ndarray:
        P = np.stack([np.asarray(m.predict(), dtype=np.float64) for m in self.members])
        s = P.sum(axis=1, keepdims=True)
        return np.divide(P, s, out=np.tile(self.channels.uniform(), (self.N, 1)), where=s > 0)

    def update(self, obs: Observation) -> float:
        # 1. Each member's one-step-ahead prediction for THIS slot, formed from
        #    o_{1:t-1} (before it consumes o_t). This is what we score.
        preds = self._member_preds()
        c = _clean_channel(obs)

        # 2. Members consume the slot (advances their state + next prediction).
        for m in self.members:
            m.update(obs)
        self.t += 1

        # 3. Mixture predictive of the realised channel, and the reweighting.
        if c is not None:
            member_p = np.clip(preds[:, c], 1e-12, 1.0)
            mix_p = float(np.dot(self.w, member_p))
            loglik = float(np.log(max(mix_p, 1e-300)))
            losses = -np.log(member_p)
            log_w = np.log(np.clip(self.w, 1e-300, 1.0)) - self.eta * losses
            log_w -= logsumexp(log_w)
            w = np.exp(log_w)
            # Fixed-Share: bleed a fraction alpha of the mass back to uniform.
            w = (1.0 - self.share) * w + self.share / self.N
            self.w = w / w.sum()
            scored = True
            member_losses = losses
        else:
            loglik = float("nan")
            scored = False
            member_losses = np.full(self.N, np.nan)

        self._last = MixtureStep(loglik=loglik, surprise=-loglik if scored else float("nan"),
                                 weights=self.w.copy(), member_losses=member_losses,
                                 scored=scored)
        return loglik

    def predict(self) -> np.ndarray:
        """The mixture next-hop distribution ``p_t^mix`` over channels."""
        P = self._member_preds()
        mix = self.w @ P
        s = mix.sum()
        return mix / s if s > 0 else self.channels.uniform()

    # --------------------------------------------------------------- diagnostics
    @property
    def last(self) -> MixtureStep:
        return self._last

    def predictability(self) -> float:
        """kappa = 1 - H(p_t^mix)/log M, in [0, 1] (paper Section 4.7)."""
        p = self.predict()
        nz = p[p > 0]
        H = float(-(nz * np.log(nz)).sum())
        return float(1.0 - H / np.log(self.channels.size))

    def weight_of(self, label: str) -> float:
        return float(self.w[self.labels.index(label)])

    # ------------------------------------------------------- regime-alarm hooks
    def reset_weights(self) -> None:
        """Flatten the mixture weights back to uniform (regime alarm, part 1)."""
        self.w = np.full(self.N, 1.0 / self.N)

    def forget(self, retain: float = 0.0) -> None:
        """Flush/window every member's learned context (regime alarm, part 2).

        This is the step the doc's Section 6.4 left implicit: resetting the
        mixture weights alone is not enough, because the universal predictor's
        context tables are still full of the *old* generator's patterns and keep
        mispredicting until they flush. Here we propagate ``forget`` to every
        member so the context model is forgotten too.
        """
        for m in self.members:
            m.forget(retain)

    def on_regime_alarm(self, retain: float = 0.0, *, reset_weights: bool = True,
                        forget_context: bool = True) -> None:
        """Both halves of the switch response in one call (paper Section 6.4)."""
        if reset_weights:
            self.reset_weights()
        if forget_context:
            self.forget(retain)


def build_default_mixture(channels: ChannelSet, window: int, sensor: SensorModel,
                          *, specs=None, eta: float = 0.5, share: float = 0.02,
                          emission: EmissionModel | None = None) -> FixedShareMixture:
    """A ready-made mixture: the Bayesian bank + a universal (context-mixing)
    predictor + an online Markov expert + the barrage floor.

    These four cover the thesis' "two mechanisms side by side" (the bank and the
    live experts) plus the floor, and are exactly the members whose recovery
    after a switch the regime experiments measure.
    """
    from fhew.config import default_bank_specs
    from fhew.experts import (
        BarrageHypothesis,
        ContextMixingHypothesis,
        MarkovHypothesis,
    )
    from fhew.config import build_bank

    emission = emission or SensorAwareEmission(sensor)
    specs = specs or default_bank_specs(channels.size)
    bank = build_bank(channels, window, sensor, specs,
                      stopping={"kind": "msprt", "delta": 0.05})
    universal = HypothesisExpert(
        ContextMixingHypothesis("context_mixing", channels, max_order=3), emission, channels)
    markov = HypothesisExpert(MarkovHypothesis("markov_live", channels, alpha=1.0),
                              emission, channels)
    barrage = HypothesisExpert(BarrageHypothesis("barrage", channels, floor="uniform"),
                               emission, channels)
    return FixedShareMixture([bank, universal, markov, barrage],
                             channels=channels, eta=eta, share=share)
