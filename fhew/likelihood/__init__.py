"""Sensor-aware likelihood: the forward filter and emission strategies.

The per-slot likelihood ``P(o_t | H_k, o_{1:t-1})`` is the one quantity the
posterior needs, and computing it correctly under a lossy sensor is the paper's
technical core (Section 6). Two filter flavours share one interface:

    * :class:`DeterministicTrajectoryFilter` -- a particle filter over candidate
      deterministic trajectories (LFSR states, LCG/weak seeds). The hidden state
      is the candidate parameterisation; the forward pass marginalises over it.
    * :class:`ChannelHMMFilter` -- a channel-level HMM for Markov / n-gram and
      the uniform H_0 floor.

Both consume a *reduced* log-emission vector (``fhew.sensor`` or the naive
baseline in :mod:`fhew.likelihood.emission`) so the comparison across every
hypothesis is fair -- the same emission model is used for all of them.
"""

from fhew.likelihood.emission import (
    EmissionModel,
    NaiveEmission,
    SensorAwareEmission,
)
from fhew.likelihood.forward import (
    ChannelHMMFilter,
    DeterministicTrajectoryFilter,
    ForwardFilter,
    StepResult,
)

__all__ = [
    "ForwardFilter",
    "StepResult",
    "DeterministicTrajectoryFilter",
    "ChannelHMMFilter",
    "EmissionModel",
    "SensorAwareEmission",
    "NaiveEmission",
]
