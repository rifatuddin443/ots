"""The next-channel predictor: a ready-to-use front end over the Fixed-Share mix.

This is the object that answers the headline question — *what channel will the
emitter hop to next?* — rather than *which generator is it?*. It wraps a
:class:`~fhew.experts.FixedShareMixture` over a bank of experts that always
includes a zero-shot universal predictor, so it keeps producing a sharp next-hop
distribution even for a generator no expert was built for (thesis §4.2).

Usage
-----
>>> from fhew import ChannelSet, SensorModel
>>> from fhew.predict import ChannelPredictor
>>> ch = ChannelSet.of_size(16)
>>> sensor = SensorModel(p_d=0.95, p_fa=0.01)
>>> pred = ChannelPredictor(ch, sensor)
>>> for obs in stream:              # feed the lossy sensor stream slot by slot
...     pred.observe(obs)
...     guess = pred.predict_top1()  # best guess for the next channel
...     topk = pred.predict_topk(3)  # or a shortlist of channels to cover

The predictor never sees the true channel or the generator; it consumes only
:class:`~fhew.sensor.Observation` objects, so the information-parity rule holds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fhew.channels import ChannelSet
from fhew.config import prediction_expert_specs
from fhew.experts import FixedShareMixture, HypothesisSpec, build_hypothesis
from fhew.sensor import Observation, SensorModel


class ChannelPredictor:
    """Predict the next hop channel from a lossy observation stream.

    Parameters
    ----------
    channels:
        The channel set ``C``.
    sensor:
        The sensor model (its ``P_d`` / ``P_fa`` enter every expert's likelihood).
    specs:
        Expert specs. Defaults to :func:`fhew.config.prediction_expert_specs`,
        which carries a universal predictor so out-of-bank generators are still
        predicted.
    horizon:
        Trajectory horizon for the enumeration experts; defaults to a generous
        bound so deterministic rolls never run short.
    eta, alpha:
        Fixed-Share learning rate and share floor. Larger ``eta`` locks onto the
        best expert faster; ``alpha`` sets how quickly the mixture can switch
        experts after a regime change.
    """

    def __init__(
        self,
        channels: ChannelSet,
        sensor: SensorModel,
        *,
        specs: list[HypothesisSpec] | None = None,
        horizon: int = 256,
        eta: float = 1.0,
        alpha: float = 0.02,
    ):
        self.channels = channels
        self.M = channels.size
        self.sensor = sensor
        self.specs = specs if specs is not None else prediction_expert_specs(channels.size)
        self.horizon = int(horizon)
        experts = [build_hypothesis(s, channels, self.horizon) for s in self.specs]
        self.mixture = FixedShareMixture(
            channels, experts, sensor=sensor, eta=eta, alpha=alpha,
            labels=[s.label for s in self.specs],
        )

    # ------------------------------------------------------------------ driving
    def observe(self, obs: Observation) -> float:
        """Consume one slot's observation; return the mixture's predictive loglik."""
        return self.mixture.update(obs)

    def reset(self) -> None:
        self.mixture.reset()

    # --------------------------------------------------------------- prediction
    def predict_next(self) -> np.ndarray:
        """Full distribution ``p(c)`` over the next channel."""
        return self.mixture.predict()

    def predict_top1(self) -> int:
        """Single best guess for the next channel."""
        return self.mixture.predict_argmax()

    def predict_topk(self, k: int = 3) -> list[int]:
        """Shortlist of the ``k`` most likely next channels, most likely first."""
        return self.mixture.predict_topk(k)

    def predictability(self) -> float:
        """How sharp the current prediction is: ``kappa = 1 - H(p)/log M`` in [0,1]."""
        return self.mixture.predictability()

    def dominant_expert(self) -> str:
        """Which expert is currently carrying the prediction (diagnostic only)."""
        return self.mixture.dominant_label()

    def expert_weights(self) -> dict[str, float]:
        return {lab: float(w) for lab, w in zip(self.mixture.labels, self.mixture.weights())}


# ---------------------------------------------------------------- evaluation API
@dataclass
class PredictionScore:
    """Next-channel prediction quality over one or more episodes."""

    n_slots: int
    top1: float                 # headline metric: P(argmax == true next channel)
    top3: float
    logloss: float              # mean -log p(true next channel), in nats
    top1_steady: float          # top-1 over the final `steady_window` slots
    uniform_logloss: float      # log M, the no-skill floor
    lift_top1: float            # top1 - 1/M, accuracy above chance
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "n_slots": self.n_slots, "top1": self.top1, "top3": self.top3,
            "logloss": self.logloss, "top1_steady": self.top1_steady,
            "uniform_logloss": self.uniform_logloss, "lift_top1": self.lift_top1,
            **self.extra,
        }


def predict_over_episode(predictor: ChannelPredictor, true_channels, observations,
                         *, warmup: int = 0, steady_window: int = 40,
                         reset: bool = True) -> PredictionScore:
    """Run a predictor over one episode and score its next-channel predictions.

    At each slot ``t`` the predictor ingests ``observations[t]`` and then predicts
    channel ``t+1``; the prediction is scored against the hidden ``true_channels``
    (used only for scoring, never shown to the predictor). ``warmup`` slots are
    ingested but excluded from the score.
    """
    true = np.asarray(true_channels, dtype=np.int64)
    window = len(observations)
    if reset:
        predictor.reset()
    hits1, hits3, lls = [], [], []
    M = predictor.M
    for t, o in enumerate(observations):
        predictor.observe(o)
        if t >= window - 1 or t < warmup:
            continue
        p = predictor.predict_next()
        nxt = int(true[t + 1])
        order = np.argsort(p)[::-1]
        hits1.append(int(nxt == order[0]))
        hits3.append(int(nxt in order[:3]))
        lls.append(-float(np.log(max(float(p[nxt]), 1e-12))))
    n = len(hits1)
    top1 = float(np.mean(hits1)) if n else float("nan")
    steady = float(np.mean(hits1[-steady_window:])) if n else float("nan")
    return PredictionScore(
        n_slots=n,
        top1=top1,
        top3=float(np.mean(hits3)) if n else float("nan"),
        logloss=float(np.mean(lls)) if n else float("nan"),
        top1_steady=steady,
        uniform_logloss=float(np.log(M)),
        lift_top1=top1 - 1.0 / M if n else float("nan"),
    )
