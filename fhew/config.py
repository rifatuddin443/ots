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
from fhew.experts import HypothesisBank
from fhew.experts import H0Hypothesis
from fhew.experts import HypothesisSpec, build_hypothesis
from fhew.likelihood import make_emission
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


# Canonical filtered-LFSR family the recce prior supplies to the correlation
# attack; episodes of 'filtered_lfsr' in the coverage experiments use the same
# (length, filter_taps) so the attack can enumerate the register state.
CORRELATION_FILTER_TAPS = (0, 1, 2, 3, 4, 5, 6)
CORRELATION_LFSR_LENGTH = 9


def identification_bank_specs(n_channels: int, *, lfsr_lengths=(5, 6, 7),
                              weak_keyspace=256, lcg_seed_max=1024) -> list[HypothesisSpec]:
    """One best expert per generator family -- the full identifiable bank.

    A clean partition for MSPRT identification across every family the thesis
    bank covers (paper Section 5): algebraic/state-filtered LFSR recovery, LCG
    seed search and algebraic LCG cryptanalysis, weak-seed search, the
    correlation attack on a filtered LFSR, parametric chaos ID, and the Markov /
    context-mixing predictors. H_0 is added by the builder.
    """
    return [
        HypothesisSpec("lfsr", "lfsr", "deterministic", grid={"length": list(lfsr_lengths)}),
        HypothesisSpec("filtered_lfsr", "filtered_lfsr", "deterministic",
                       grid={"length": [CORRELATION_LFSR_LENGTH],
                             "filter_taps": [CORRELATION_FILTER_TAPS]}),
        HypothesisSpec("lcg", "lcg", "deterministic", grid={"seed_max": lcg_seed_max}),
        HypothesisSpec("lcg_full", "lcg_full", "lcg_crypto"),
        HypothesisSpec("weak_seed", "weak_seed", "deterministic",
                       grid={"keyspace": weak_keyspace}),
        HypothesisSpec("chaos", "logistic", "chaos_id"),
        HypothesisSpec("markov", "markov", "markov", options={"alpha": 1.0}),
    ]


def full_bank_specs(n_channels: int, *, lfsr_lengths=(5, 6, 7),
                    weak_keyspace=256, lcg_seed_max=1024) -> list[HypothesisSpec]:
    """Every expert in the thesis bank, including redundant ones per family.

    Used for the expert-coverage confusion matrix, where it matters which
    *expert* fires for each generator (e.g. plain Berlekamp-Massey vs the
    gap-robust state-filtered variant, both targeting an LFSR).
    """
    return [
        HypothesisSpec("bm", "lfsr", "bm", options={"max_length": max(lfsr_lengths)}),
        HypothesisSpec("state_filtered_bm", "lfsr", "deterministic",
                       grid={"length": list(lfsr_lengths)}),
        HypothesisSpec("correlation_attack", "filtered_lfsr", "deterministic",
                       grid={"length": [CORRELATION_LFSR_LENGTH],
                             "filter_taps": [CORRELATION_FILTER_TAPS]}),
        HypothesisSpec("lcg_seed_search", "lcg", "deterministic", grid={"seed_max": lcg_seed_max}),
        HypothesisSpec("lcg_cryptanalysis", "lcg_full", "lcg_crypto"),
        HypothesisSpec("seed_search", "weak_seed", "deterministic",
                       grid={"keyspace": weak_keyspace}),
        HypothesisSpec("chaos_id", "logistic", "chaos_id"),
        HypothesisSpec("phase_space", "logistic", "phase_space"),
        HypothesisSpec("markov", "markov", "markov", options={"alpha": 1.0}),
        HypothesisSpec("context_mixing", "markov", "context_mixing"),
        HypothesisSpec("behaviour", "markov", "behaviour"),
        HypothesisSpec("barrage", "none", "barrage"),
    ]


def prediction_expert_specs(n_channels: int, *, lfsr_lengths=(5, 6, 7),
                            weak_keyspace=256, lcg_seed_max=1024,
                            max_order: int = 4) -> list[HypothesisSpec]:
    """Experts for the Fixed-Share **next-channel predictor** (thesis §4.2).

    The identification experts (one per known family) plus a *zero-shot universal
    predictor* and the model-free fallbacks, so the next-channel prediction keeps
    working on a generator the bank has no dedicated expert for ("if a new algo
    is used, still predict the next channel"). The universal `context_mixing`
    expert is a variable-order Markov mixture: it recovers in-context regularity
    (including higher-order rules a first-order Markov expert misses), `phase_space`
    carries unseen chaotic dynamics, and `barrage` is the uniform floor for the
    genuinely unpredictable (keyed CSPRNG).
    """
    specs = identification_bank_specs(
        n_channels, lfsr_lengths=lfsr_lengths, weak_keyspace=weak_keyspace,
        lcg_seed_max=lcg_seed_max)
    specs += [
        HypothesisSpec("context_mixing", "markov", "context_mixing",
                       options={"max_order": max_order}),
        HypothesisSpec("phase_space", "logistic", "phase_space"),
        HypothesisSpec("barrage", "none", "barrage"),
    ]
    return specs


#: which generator family each expert label targets (for coverage scoring)
EXPERT_TARGET_FAMILY = {
    "lfsr": "lfsr", "bm": "lfsr", "state_filtered_bm": "lfsr",
    "filtered_lfsr": "filtered_lfsr", "correlation_attack": "filtered_lfsr",
    "lcg": "lcg", "lcg_seed_search": "lcg",
    "lcg_full": "lcg_full", "lcg_cryptanalysis": "lcg_full",
    "weak_seed": "weak_seed", "seed_search": "weak_seed",
    "chaos": "chaos", "chaos_id": "chaos", "phase_space": "chaos",
    "markov": "markov", "context_mixing": "markov", "behaviour": "markov",
    "barrage": "unknown", "H0": "unknown",
}

#: the family a correct system should report for each true generator
GENERATOR_TARGET_FAMILY = {
    "lfsr": "lfsr", "filtered_lfsr": "filtered_lfsr", "lcg": "lcg", "lcg_full": "lcg_full",
    "weak_seed": "weak_seed", "markov": "markov",
    "logistic": "chaos", "tent": "chaos", "henon": "chaos",
    "csprng": "unknown",
    # "new algorithm" families: no dedicated expert, so identification routes to
    # unknown; they exist to test next-channel *prediction*, not identification.
    "lagged_fib": "unknown", "quadratic_cg": "unknown", "sine_map": "unknown",
}

#: "new algorithm" generators held out of every bank, used by the prediction
#: benchmark to show the universal predictor generalises beyond the bank.
NEW_ALGORITHM_GENERATORS = ("lagged_fib", "quadratic_cg", "sine_map")


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
    h0_prior_floor: float = 0.15,
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
