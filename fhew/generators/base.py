"""Generator interface and the name-keyed registry.

A hop generator maps an internal state to a stream of channel indices. Every
generator family lives in a registry keyed by name (paper Section 5, "Registry
design"), so the journal adds chaotic and learned adversaries by registering
them, with no change to the bank or the stopping rule.

Two roles are deliberately separated:

    * :class:`Generator` -- the *truth* side: an emitter the gym runs to produce
      a channel sequence. It knows its own seed and state.
    * the hypothesis side (``fhew.experts``) only ever sees observations and the
      recce *ranges*, never a :class:`Generator` instance, which is how the
      information-parity rule is enforced structurally.
"""

from __future__ import annotations

import abc
from typing import Callable, Dict, Iterable

import numpy as np


class Generator(abc.ABC):
    """A frequency-hopping channel generator (the truth side).

    Subclasses implement :meth:`next_channel`; the base class turns that into a
    reproducible sequence. ``name`` is the registry key and the ground-truth
    label used only for scoring, never exposed to the bank.
    """

    #: registry key; set by @register
    name: str = "base"

    def __init__(self, n_channels: int):
        if n_channels <= 0:
            raise ValueError("n_channels must be positive")
        self.n_channels = int(n_channels)

    @abc.abstractmethod
    def next_channel(self) -> int:
        """Advance the internal state by one slot and return the channel index."""

    def reset(self) -> None:  # pragma: no cover - optional override
        """Reset internal state to the configured seed, if meaningful."""

    def sequence(self, length: int) -> np.ndarray:
        """Produce ``length`` channel indices, advancing the state each slot."""
        out = np.empty(length, dtype=np.int64)
        for i in range(length):
            c = int(self.next_channel())
            if not 0 <= c < self.n_channels:
                raise ValueError(f"generator {self.name} produced out-of-range channel {c}")
            out[i] = c
        return out

    def describe(self) -> dict:  # pragma: no cover - cosmetic / logging
        return {"name": self.name, "n_channels": self.n_channels}


# --------------------------------------------------------------------- registry
_REGISTRY: Dict[str, type[Generator]] = {}


def register(name: str) -> Callable[[type[Generator]], type[Generator]]:
    """Class decorator registering a generator family under ``name``."""

    def deco(cls: type[Generator]) -> type[Generator]:
        if name in _REGISTRY:
            raise KeyError(f"generator '{name}' already registered")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


def get_generator_cls(name: str) -> type[Generator]:
    try:
        return _REGISTRY[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"unknown generator '{name}'; registered: {sorted(_REGISTRY)}") from exc


def make_generator(name: str, n_channels: int, rng: np.random.Generator | None = None,
                   **params) -> Generator:
    """Instantiate a registered generator.

    ``rng`` is used to draw any unspecified hidden parameters (seed, register
    state, taps), so the *truth* can be randomised across seeds while the
    hypothesis side only knows the ranges.
    """
    cls = get_generator_cls(name)
    return cls.create(n_channels=n_channels, rng=rng, **params)  # type: ignore[attr-defined]


def registered_names() -> list[str]:
    return sorted(_REGISTRY)


def iter_registry() -> Iterable[tuple[str, type[Generator]]]:
    return _REGISTRY.items()
