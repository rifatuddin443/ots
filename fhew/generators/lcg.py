"""Linear congruential generator (LCG) and weak-seed hop generators."""

from __future__ import annotations

import numpy as np

from fhew.generators.base import Generator, register


@register("lcg")
class LCGGenerator(Generator):
    """Linear congruential generator: ``x <- (a*x + c) mod m``.

    The channel is the high bits of the state (``x >> shift) mod M``), which is
    the usual way LCG output is consumed and avoids the well-known low-bit
    periodicity. Hidden parameters are ``(m, a, c, seed)``; the hypothesis side
    searches over configured ranges.
    """

    def __init__(self, n_channels: int, modulus: int, mult: int, incr: int, seed: int):
        super().__init__(n_channels)
        self.m = int(modulus)
        self.a = int(mult)
        self.c = int(incr)
        self._seed = int(seed) % self.m
        self.x = self._seed
        # Use the top bits for channel selection.
        self.shift = max(0, self.m.bit_length() - n_channels.bit_length() - 1)

    @classmethod
    def create(cls, n_channels: int, rng: np.random.Generator | None = None,
               modulus: int = 2 ** 16, mult: int = 25214903917 & (2 ** 16 - 1),
               incr: int = 11, seed: int | None = None):
        rng = rng or np.random.default_rng()
        # A small, well-known-style LCG (glibc/POSIX-like constants, reduced).
        if modulus == 2 ** 16:
            mult, incr = 1103515245 % modulus, 12345 % modulus
        if seed is None:
            seed = int(rng.integers(0, modulus))
        return cls(n_channels=n_channels, modulus=modulus, mult=mult, incr=incr, seed=seed)

    def reset(self) -> None:
        self.x = self._seed

    def next_channel(self) -> int:
        self.x = (self.a * self.x + self.c) % self.m
        return (self.x >> self.shift) % self.n_channels

    def describe(self) -> dict:
        return {"name": self.name, "n_channels": self.n_channels,
                "modulus": self.m, "mult": self.a, "incr": self.c}


def _splitmix64(x: int) -> int:
    """SplitMix64 finalizer -- a good avalanche mix for a small-seed generator."""
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return z ^ (z >> 31)


@register("weak_seed")
class WeakSeedGenerator(Generator):
    """A well-mixed PRNG whose *keyspace* (seed range) is small.

    The per-slot output looks random, but because the seed lives in a small set
    ``[0, keyspace)`` the whole future is determined by a short search. This is
    the most realistic route to a genuine break: results are graded by key
    strength (``keyspace``). The hypothesis-side seed search enumerates the same
    range -- information parity holds because the *range* is a recce prior while
    the actual seed is never exposed.
    """

    def __init__(self, n_channels: int, seed: int, keyspace: int):
        super().__init__(n_channels)
        self.keyspace = int(keyspace)
        self._seed = int(seed) % self.keyspace
        self.counter = 0

    @classmethod
    def create(cls, n_channels: int, rng: np.random.Generator | None = None,
               keyspace: int = 256, seed: int | None = None):
        rng = rng or np.random.default_rng()
        if seed is None:
            seed = int(rng.integers(0, keyspace))
        return cls(n_channels=n_channels, seed=seed, keyspace=keyspace)

    def reset(self) -> None:
        self.counter = 0

    def next_channel(self) -> int:
        val = _splitmix64((self._seed << 20) ^ self.counter)
        self.counter += 1
        return val % self.n_channels

    @staticmethod
    def channel_for(seed: int, counter: int, n_channels: int) -> int:
        """Deterministic channel for a hypothesised seed -- used by seed search."""
        return _splitmix64((int(seed) << 20) ^ int(counter)) % n_channels

    def describe(self) -> dict:
        return {"name": self.name, "n_channels": self.n_channels, "keyspace": self.keyspace}
