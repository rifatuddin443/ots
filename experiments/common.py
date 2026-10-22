"""Shared builders and conventions for the experiment scripts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fhew.channels import ChannelSet
from fhew.config import build_bank, default_bank_specs, in_bank_labels
from fhew.experts.hypothesis import HypothesisSpec
from fhew.sensor import SensorModel
from fhew.sim.episode import Episode, make_episode

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
        return {"length": int(rng.choice([9, 11]))}
    if gen_name == "weak_seed":
        return {"keyspace": 256}                 # matches the bank's weak-seed search
    if gen_name == "lcg":
        return {"seed": int(rng.integers(0, lcg_keyspace))}  # in-range secret
    if gen_name == "markov":
        return {"alpha": float(rng.choice([0.15, 0.2, 0.3]))}
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


def seeds(n: int, base: int = 1000) -> list[int]:
    """Deterministic, well-separated seeds so conditions use disjoint streams."""
    return [base + 7919 * i for i in range(n)]
