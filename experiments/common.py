"""Shared builders and conventions for the experiment scripts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fhew.channels import ChannelSet
from fhew.config import (
    CORRELATION_FILTER_TAPS,
    CORRELATION_LFSR_LENGTH,
    EXPERT_TARGET_FAMILY,
    GENERATOR_TARGET_FAMILY,
    build_bank,
    default_bank_specs,
    full_bank_specs,
    identification_bank_specs,
    in_bank_labels,
)
from fhew.experts import HypothesisSpec
from fhew.sensor import SensorModel
from fhew.sim import Episode, make_episode

# Every generator family the full thesis bank covers, grouped by whether it is
# identifiable (an expert targets it) or an out-of-bank / unbreakable adversary.
ALL_GENERATORS = (
    "lfsr", "filtered_lfsr", "lcg", "lcg_full", "weak_seed",
    "logistic", "tent", "henon", "markov", "csprng",
)
IDENTIFIABLE_GENERATORS = (
    "lfsr", "filtered_lfsr", "lcg", "lcg_full", "weak_seed",
    "logistic", "tent", "henon", "markov",
)

RESULTS_DIR = Path("results")
FIG_DIR = RESULTS_DIR / "figures"

# The families the bank can name; everything else should route to H_0 ('unknown').
IN_BANK_GENERATORS = ("lfsr", "weak_seed", "lcg", "markov")
OUT_OF_BANK_GENERATORS = ("csprng", "filtered_lfsr")

# Families that assert an exact algebraic / deterministic break. Naming one of
# these when it is not the truth is the *dangerous* false identification the
# paper warns against ("name the wrong generator with false confidence").
# Falling back to 'markov' or 'unknown' on an out-of-bank emitter is graceful,
# not dangerous, so it is tracked separately.
DETERMINISTIC_FAMILIES = frozenset({"lfsr", "filtered_lfsr", "lcg", "weak_seed"})


def is_hard_false_id(result) -> bool:
    return (result.decision_label in DETERMINISTIC_FAMILIES
            and result.decision_label != result.target_label)

DEFAULT_M = 16
DEFAULT_WINDOW = 100


def default_specs(M: int = DEFAULT_M, lcg_seed_max: int = 1024) -> list[HypothesisSpec]:
    return default_bank_specs(M, lcg_seed_max=lcg_seed_max)


def make_bank(channels: ChannelSet, window: int, sensor: SensorModel,
              specs: list[HypothesisSpec], *, emission_kind: str = "sensor_aware",
              stopping: dict | None = None, h0_mode: str = "uniform"):
    return build_bank(
        channels, window, sensor, specs,
        emission={"kind": emission_kind},
        stopping=stopping or {"kind": "msprt", "delta": 0.05},
        h0_mode=h0_mode,
    )


#: the LCG seed keyspace the bank searches; identifiable-LCG episodes draw within it
LCG_KEYSPACE = 1024


def draw_episode_params(gen_name: str, rng: np.random.Generator, *,
                        lcg_keyspace: int = LCG_KEYSPACE) -> dict:
    """Draw a generator's hidden parameters *within* the bank's recce ranges.

    This is what makes a generator identifiable: the recce prior bounds the
    secret (an LCG's seed, a weak PRNG's keyspace) so a short search can cover
    it. The stale-bank experiment deliberately draws *outside* these ranges.
    """
    if gen_name == "lfsr":
        return {"length": int(rng.choice([5, 6, 7]))}
    if gen_name == "filtered_lfsr":
        # canonical filter family the correlation attack knows (length + taps)
        return {"length": CORRELATION_LFSR_LENGTH, "filter_taps": CORRELATION_FILTER_TAPS}
    if gen_name == "weak_seed":
        return {"keyspace": 256}                 # matches the bank's weak-seed search
    if gen_name == "lcg":
        return {"seed": int(rng.integers(0, lcg_keyspace))}  # in-range secret
    if gen_name == "lcg_full":
        return {"seed": int(rng.integers(0, 256))}  # full-state LCG; a,c recovered
    if gen_name == "markov":
        return {"alpha": float(rng.choice([0.15, 0.2, 0.3]))}
    if gen_name == "logistic":
        return {"r": float(rng.choice([3.7, 3.8, 3.9, 3.95, 4.0])),
                "x0": float(rng.uniform(0.1, 0.9))}
    if gen_name == "tent":
        return {"mu": float(rng.choice([1.9, 1.95, 1.99, 2.0])),
                "x0": float(rng.uniform(0.1, 0.9))}
    if gen_name == "henon":
        return {"a": float(rng.choice([1.2, 1.3, 1.4])), "b": 0.3,
                "x0": float(rng.uniform(-0.3, 0.3)), "y0": float(rng.uniform(-0.3, 0.3))}
    return {}


def episode_for(gen_name: str, channels: ChannelSet, sensor: SensorModel, window: int,
                seed: int, *, gen_params: dict | None = None,
                target_labels: set[str] | None = None,
                lcg_keyspace: int = LCG_KEYSPACE) -> Episode:
    rng = np.random.default_rng(seed)
    if gen_params is None:
        gen_params = draw_episode_params(gen_name, rng, lcg_keyspace=lcg_keyspace)
    ep = make_episode(gen_name, channels, sensor, window, rng,
                      gen_params=gen_params, seed=seed)
    if target_labels is not None:
        ep.set_target(target_labels)
    return ep


def default_target_labels(specs: list[HypothesisSpec]) -> set[str]:
    return in_bank_labels(specs)


def id_specs(M: int = DEFAULT_M) -> list[HypothesisSpec]:
    """One best expert per family -- the full identifiable bank."""
    return identification_bank_specs(M)


def full_specs(M: int = DEFAULT_M) -> list[HypothesisSpec]:
    """Every expert (redundant per family) -- for the coverage confusion matrix."""
    return full_bank_specs(M)


def seeds(n: int, base: int = 1000) -> list[int]:
    """Deterministic, well-separated seeds so conditions use disjoint streams."""
    return [base + 7919 * i for i in range(n)]
