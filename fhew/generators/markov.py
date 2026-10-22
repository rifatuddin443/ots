"""Markov / n-gram hop generator with empirical transition regularity."""

from __future__ import annotations

import numpy as np

from fhew.generators.base import Generator, register


@register("markov")
class MarkovGenerator(Generator):
    """First-order Markov hopper: ``c_t ~ P(. | c_{t-1})``.

    The transition matrix is drawn Dirichlet with concentration ``alpha``.
    Small ``alpha`` makes transitions peaky (predictable); large ``alpha`` makes
    them near-uniform (which should pin the predictability low and route the
    posterior toward H_0). A ``blacklist`` can forbid channels (AFH-style).
    """

    def __init__(self, n_channels: int, transition: np.ndarray, start: int):
        super().__init__(n_channels)
        T = np.asarray(transition, dtype=np.float64)
        if T.shape != (n_channels, n_channels):
            raise ValueError("transition matrix shape mismatch")
        self.T = T / T.sum(axis=1, keepdims=True)
        self._start = int(start)
        self.state = self._start
        self._rng = np.random.default_rng()

    @classmethod
    def create(cls, n_channels: int, rng: np.random.Generator | None = None,
               alpha: float = 0.3, start: int | None = None,
               blacklist: tuple[int, ...] = ()):
        rng = rng or np.random.default_rng()
        T = rng.dirichlet(np.full(n_channels, alpha), size=n_channels)
        if blacklist:
            T[:, list(blacklist)] = 0.0
            T = T / T.sum(axis=1, keepdims=True)
        if start is None:
            start = int(rng.integers(0, n_channels))
        gen = cls(n_channels=n_channels, transition=T, start=start)
        gen._rng = rng
        return gen

    def reset(self) -> None:
        self.state = self._start

    def next_channel(self) -> int:
        self.state = int(self._rng.choice(self.n_channels, p=self.T[self.state]))
        return self.state

    def describe(self) -> dict:
        return {"name": self.name, "n_channels": self.n_channels,
                "alpha_effective": float(np.mean(self.T.max(axis=1)))}
