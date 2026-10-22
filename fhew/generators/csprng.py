"""Keyed cryptographic hop generator -- the unpredictability ceiling.

A keyed CSPRNG is unpredictable by design: no expert beats uniform against it
(paper Section 5). It is included as a *test adversary*, not as a bank
hypothesis -- the correct outcome is that the none-of-the-above class H_0 wins
and the system stops chasing structure. Implemented as SHA-256 in counter mode
keyed by a secret key.
"""

from __future__ import annotations

import hashlib

import numpy as np

from fhew.generators.base import Generator, register


@register("csprng")
class CSPRNGGenerator(Generator):
    """SHA-256 counter-mode keyed generator."""

    def __init__(self, n_channels: int, key: bytes):
        super().__init__(n_channels)
        self.key = bytes(key)
        self.counter = 0

    @classmethod
    def create(cls, n_channels: int, rng: np.random.Generator | None = None,
               key: bytes | None = None, key_bytes: int = 16):
        rng = rng or np.random.default_rng()
        if key is None:
            key = bytes(int(x) for x in rng.integers(0, 256, size=key_bytes))
        return cls(n_channels=n_channels, key=key)

    def reset(self) -> None:
        self.counter = 0

    def next_channel(self) -> int:
        h = hashlib.sha256(self.key + self.counter.to_bytes(8, "big")).digest()
        self.counter += 1
        val = int.from_bytes(h[:8], "big")
        return val % self.n_channels

    def describe(self) -> dict:
        return {"name": self.name, "n_channels": self.n_channels, "keyed": True}
