"""Config dataclasses, YAML loading, and bank builders (design rule 5).

Everything runs from config files with fixed seeds so results regenerate. A
config is a plain dict (loaded from YAML); the builders here turn it into the
runtime objects. The same builders back the programmatic API the experiments
use, so a YAML config and a Python call produce identical banks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from fhew.channels import ChannelSet
from fhew.experts.bank import HypothesisBank
from fhew.experts.h0 import H0Hypothesis
from fhew.experts.hypothesis import HypothesisSpec, build_hypothesis
from fhew.likelihood.emission import make_emission
from fhew.sensor import SensorModel
from fhew.stopping import make_stopping_rule


# --------------------------------------------------------------- default banks
def default_bank_specs(n_channels: int, *, lfsr_lengths=(5, 6, 7),
                       weak_keyspace=256, lcg_seed_max=2048,
                       include_markov=True) -> list[HypothesisSpec]:
    """The standard conference bank: LFSR, weak-seed, LCG (+ Markov).

    These are the generator families the reconnaissance prior supplies. H_0 is
    added separately by the bank builder. Ranges are deliberately bounded (the
    recce prior); a true generator outside them should route to H_0.
    """
    specs = [
        HypothesisSpec("lfsr", "lfsr", "deterministic", grid={"length": list(lfsr_lengths)}),
        HypothesisSpec("weak_seed", "weak_seed", "deterministic",
                       grid={"keyspace": weak_keyspace}),
        HypothesisSpec("lcg", "lcg", "deterministic", grid={"seed_max": lcg_seed_max}),
    ]
    if include_markov:
        specs.append(HypothesisSpec("markov", "markov", "markov", options={"alpha": 1.0}))
    return specs


def specs_from_list(items: list[dict]) -> list[HypothesisSpec]:
    out = []
    for it in items:
        out.append(HypothesisSpec(
            label=it["label"], family=it["family"], kind=it.get("kind", "deterministic"),
            grid=it.get("grid", {}), max_particles=it.get("max_particles", 8192),
            options=it.get("options", {}),
        ))
    return out


def build_bank(
    channels: ChannelSet,
    horizon: int,
    sensor: SensorModel,
    specs: list[HypothesisSpec] | None = None,
    *,
    h0_mode: str = "uniform",
    h0_floor_kind: str = "uniform",
    stopping: dict | None = None,
    emission: dict | None = None,
    prior: np.ndarray | None = None,
    h0_prior_floor: float = 0.1,
) -> HypothesisBank:
    """Assemble a bank from specs + runtime settings.

    ``emission`` selects the sensor-aware method or the naive baseline;
    ``stopping`` selects MSPRT or the e-process. Making both switches part of the
    builder is what keeps the ablation one codebase.
    """
    if specs is None:
        specs = default_bank_specs(channels.size)
    emission = emission or {"kind": "sensor_aware"}
    stopping = stopping or {"kind": "msprt", "delta": 0.05}

    emit = make_emission(sensor=sensor, **emission)
    stop = make_stopping_rule(**stopping)
    h0 = H0Hypothesis(channels, mode=h0_mode, floor=h0_floor_kind)
    hyps = [build_hypothesis(s, channels, horizon) for s in specs]
    return HypothesisBank(channels, h0, hyps, emission=emit, stopping=stop,
                          prior=prior, h0_floor=h0_prior_floor)


def in_bank_labels(specs: list[HypothesisSpec]) -> set[str]:
    return {s.label for s in specs} | {"unknown"}


# ------------------------------------------------------------------- YAML load
def load_yaml(path: str | Path) -> dict:
    with Path(path).open() as fh:
        return yaml.safe_load(fh)


@dataclass
class ExperimentConfig:
    """Top-level experiment config (sensor sweeps, adversaries, seeds)."""

    name: str
    n_channels: int = 16
    window: int = 100
    seeds: int = 30
    p_d_grid: list[float] = field(default_factory=lambda: [1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
    p_fa_grid: list[float] = field(default_factory=lambda: [0.0, 0.01, 0.02, 0.05, 0.1])
    adversaries: list[dict] = field(default_factory=list)
    bank: dict = field(default_factory=dict)
    output_dir: str = "results"
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        d = load_yaml(path)
        known = {f for f in cls.__dataclass_fields__}
        extra = {k: v for k, v in d.items() if k not in known}
        init = {k: v for k, v in d.items() if k in known}
        return cls(**init, extra=extra)

    def channels(self) -> ChannelSet:
        return ChannelSet.of_size(self.n_channels)

    def bank_specs(self) -> list[HypothesisSpec]:
        if self.bank.get("specs"):
            return specs_from_list(self.bank["specs"])
        return default_bank_specs(self.n_channels,
                                  include_markov=self.bank.get("include_markov", True))
