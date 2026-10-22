"""Emission strategies: the sensor-aware likelihood and the naive baseline.

An *emission model* turns one slot's :class:`~fhew.sensor.Observation` into a
reduced log-emission vector ``log e_t(s)`` over the true channel ``s``. Making
this a strategy object is what lets the ablation (paper Section 11, item 7) be a
single switch rather than a second codebase: the forward filter is identical,
only the emission differs.
"""

from __future__ import annotations

import abc

import numpy as np

from fhew.sensor import Observation, SensorModel


class EmissionModel(abc.ABC):
    """Maps an observation to a reduced log-emission vector over channels."""

    name: str = "base"

    @abc.abstractmethod
    def log_emission(self, obs: Observation) -> np.ndarray:
        ...


class SensorAwareEmission(EmissionModel):
    """The method: emission built from the sensor's ``P_d`` and ``P_fa``.

    Gaps (missed hops) and phantoms (false alarms) are handled inside the
    likelihood instead of corrupting it. Delegates to
    :meth:`SensorModel.emission_loglikelihood`.
    """

    name = "sensor_aware"

    def __init__(self, sensor: SensorModel):
        self.sensor = sensor

    def log_emission(self, obs: Observation) -> np.ndarray:
        return self.sensor.emission_loglikelihood(obs)


class NaiveEmission(EmissionModel):
    """The baseline: ignore ``P_d`` / ``P_fa`` and score only clean matches.

    A channel that produced a detection scores ``log 1 = 0``; a channel with no
    detection scores ``log(eps_miss)``. There is no false-alarm discounting, so a
    missed true hop is (wrongly) treated as evidence against the true generator
    and a phantom as evidence for a wrong one. The gap to
    :class:`SensorAwareEmission`, measured as hops-to-identify and false-ID rate,
    is the ablation that justifies the sensor-aware approach.
    """

    name = "naive"

    def __init__(self, eps_miss: float = 1e-3):
        if not 0.0 < eps_miss < 1.0:
            raise ValueError("eps_miss must be in (0, 1)")
        self.log_eps = float(np.log(eps_miss))

    def log_emission(self, obs: Observation) -> np.ndarray:
        detected = obs.detection_mask()
        log_e = np.full(obs.n_channels, self.log_eps, dtype=np.float64)
        log_e[detected] = 0.0
        return log_e


def make_emission(kind: str, sensor: SensorModel, **kwargs) -> EmissionModel:
    """Factory used by configs: ``'sensor_aware'`` or ``'naive'``."""
    if kind == "sensor_aware":
        return SensorAwareEmission(sensor)
    if kind == "naive":
        return NaiveEmission(**kwargs)
    raise ValueError(f"unknown emission kind '{kind}'")
