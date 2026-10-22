"""An engagement episode: one emitter, its lossy observation stream, a window.

An episode bundles the ground truth (never shown to the bank) with the sensor
output (all the bank sees). ``target_label`` is what a *correct* system should
output: the true family when it is in the bank's label set, else ``'unknown'``
-- so an out-of-bank or crypto emitter is scored as a correct detection of the
none-of-the-above condition, not as a miss.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fhew.channels import ChannelSet
from fhew.generators import Generator, make_generator
from fhew.sensor import Observation, SensorModel

UNKNOWN_LABEL = "unknown"


@dataclass
class Episode:
    true_label: str                     # the generator family that produced it
    true_channels: np.ndarray           # hidden ground-truth channel sequence
    observations: list[Observation]     # the lossy stream (all the bank sees)
    window: int
    channels: ChannelSet
    p_d: float
    p_fa: float
    seed: int
    gen_params: dict = field(default_factory=dict)
    target_label: str = ""              # what a correct system should output

    def set_target(self, in_bank_labels: set[str]) -> "Episode":
        self.target_label = self.true_label if self.true_label in in_bank_labels else UNKNOWN_LABEL
        return self


def make_episode(
    gen_name: str,
    channels: ChannelSet,
    sensor: SensorModel,
    window: int,
    rng: np.random.Generator,
    *,
    gen_params: dict | None = None,
    seed: int | None = None,
    label: str | None = None,
) -> Episode:
    """Draw a generator, run it for ``window`` slots, and sense it."""
    gen_params = dict(gen_params or {})
    g: Generator = make_generator(gen_name, channels.size, rng=rng, **gen_params)
    true = g.sequence(window)
    obs = sensor.observe(true, channels, rng)
    return Episode(
        true_label=label or gen_name,
        true_channels=true,
        observations=obs,
        window=window,
        channels=channels,
        p_d=sensor.p_d,
        p_fa=sensor.p_fa,
        seed=int(seed) if seed is not None else -1,
        gen_params=gen_params,
    )
