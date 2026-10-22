"""Hop-generator library + registry (LFSR, filtered-LFSR, LCG, weak-seed, Markov, CSPRNG).

This module merges what used to be the fhew/generators/ subpackage; the public
names are unchanged, so `from fhew.generators import X` still works.
"""

from __future__ import annotations

# ---- merged from fhew/generators/base.py ----
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

# ---- merged from fhew/generators/lfsr.py ----
import math

import numpy as np



def _bits_per_channel(n_channels: int) -> int:
    b = int(round(math.log2(n_channels)))
    if 2 ** b != n_channels:
        raise ValueError(
            f"LFSR generators need a power-of-two channel count; got M={n_channels}"
        )
    return b


def _parity(x: int) -> int:
    return bin(x).count("1") & 1


# Default maximal-length feedback polynomials (tap masks) by register length.
# Bit i set means register position i feeds back. These are standard primitive
# polynomials, so the LFSR has maximal period 2**L - 1.
_PRIMITIVE_TAPS = {
    3: 0b101,
    4: 0b1001,
    5: 0b10010,
    6: 0b100001,
    7: 0b1000001,
    8: 0b10001110,
    9: 0b100010000,
    10: 0b1000000100,
    11: 0b10000000010,
    12: 0b100000101001,
}


@register("lfsr")
class LFSRGenerator(Generator):
    """Fibonacci LFSR hop generator.

    Parameters
    ----------
    length:
        Register length ``L``.
    taps:
        Feedback tap mask. Defaults to a primitive polynomial for ``length``.
    state:
        Initial non-zero register state. Defaults to a random non-zero state.
    """

    def __init__(self, n_channels: int, length: int, taps: int, state: int):
        super().__init__(n_channels)
        self.b = _bits_per_channel(n_channels)
        self.length = int(length)
        self.taps = int(taps)
        self.mask = (1 << self.length) - 1
        if state == 0:
            raise ValueError("LFSR initial state must be non-zero")
        self._init_state = int(state) & self.mask
        self.state = self._init_state

    @classmethod
    def create(cls, n_channels: int, rng: np.random.Generator | None = None,
               length: int = 7, taps: int | None = None, state: int | None = None):
        rng = rng or np.random.default_rng()
        if taps is None:
            taps = _PRIMITIVE_TAPS.get(int(length))
            if taps is None:
                raise ValueError(f"no default primitive taps for L={length}; pass taps=")
        if state is None:
            state = int(rng.integers(1, 1 << int(length)))
        return cls(n_channels=n_channels, length=length, taps=taps, state=state)

    def reset(self) -> None:
        self.state = self._init_state

    def _step_bit(self) -> int:
        out = self.state & 1
        fb = _parity(self.state & self.taps)
        self.state = (self.state >> 1) | (fb << (self.length - 1))
        return out

    def _step_bit_filtered(self) -> int:  # overridden by filtered variant
        return self._step_bit()

    def next_channel(self) -> int:
        c = 0
        for _ in range(self.b):
            c = (c << 1) | self._step_bit_filtered()
        return c

    def describe(self) -> dict:
        return {"name": self.name, "n_channels": self.n_channels,
                "length": self.length, "taps": self.taps}


@register("filtered_lfsr")
class FilteredLFSRGenerator(LFSRGenerator):
    """LFSR driven through a nonlinear boolean combiner.

    The output bit is a nonlinear function of several register positions, so the
    *channel* bit stream no longer obeys a low-degree linear recurrence. A plain
    LFSR likelihood / Berlekamp-Massey should fail here (its linear complexity
    stays near ``n/2``); the point of the generator is to be an LFSR underneath
    that algebraic recovery cannot see.

    ``filter_taps`` selects which register positions feed the combiner. The
    combiner is a fixed, well-studied correlation-immune-ish function: a
    majority of three taps XORed with the product of two others.
    """

    N_FILTER_TAPS = 7  # degree-3 combiner below needs 7 distinct tap positions

    def __init__(self, n_channels: int, length: int, taps: int, state: int,
                 filter_taps: tuple[int, ...]):
        super().__init__(n_channels, length, taps, state)
        if len(filter_taps) < self.N_FILTER_TAPS:
            raise ValueError(
                f"filtered LFSR combiner needs at least {self.N_FILTER_TAPS} tap positions"
            )
        for p in filter_taps:
            if not 0 <= p < length:
                raise ValueError("filter tap position out of range")
        self.filter_taps = tuple(int(p) for p in filter_taps)

    @classmethod
    def create(cls, n_channels: int, rng: np.random.Generator | None = None,
               length: int = 11, taps: int | None = None, state: int | None = None,
               filter_taps: tuple[int, ...] | None = None):
        rng = rng or np.random.default_rng()
        if taps is None:
            taps = _PRIMITIVE_TAPS.get(int(length))
            if taps is None:
                raise ValueError(f"no default primitive taps for L={length}; pass taps=")
        if state is None:
            state = int(rng.integers(1, 1 << int(length)))
        if filter_taps is None:
            filter_taps = tuple(
                sorted(rng.choice(int(length), size=cls.N_FILTER_TAPS, replace=False).tolist())
            )
        return cls(n_channels=n_channels, length=length, taps=taps, state=state,
                   filter_taps=filter_taps)

    def _combine(self) -> int:
        # Algebraic-degree-3 Boolean combiner. A degree-d filter on a length-L
        # m-sequence raises the linear complexity up to sum_{i<=d} C(L,i), which
        # for L=11, d=3 is 231 -- far beyond what a short clean window can fit,
        # so Berlekamp-Massey cannot recover it (paper Section 5, "strong filter
        # hides the linear structure").
        p = self.filter_taps
        bit = lambda i: (self.state >> i) & 1  # noqa: E731
        a, b, c, d, e, f, g = (bit(p[i]) for i in range(7))
        return (a & b) ^ (c & d) ^ (e & f & g) ^ a ^ g

    def _step_bit_filtered(self) -> int:
        out = self._combine()
        # advance the underlying register one clock
        self._step_bit()
        return out

    def describe(self) -> dict:
        d = super().describe()
        d["filter_taps"] = self.filter_taps
        return d

# ---- merged from fhew/generators/lcg.py ----
import numpy as np



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

# ---- merged from fhew/generators/markov.py ----
import numpy as np



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

# ---- merged from fhew/generators/csprng.py ----
import hashlib

import numpy as np



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

# ---- chaotic map generators (for the parametric-chaos-ID and phase-space experts) ----


def _chaos_channel(x: float, n_channels: int) -> int:
    """Quantise a map state ``x in [0,1)`` to a channel index (symbolic dynamics)."""
    c = int(x * n_channels)
    if c < 0:
        return 0
    if c >= n_channels:
        return n_channels - 1
    return c


@register("logistic")
class LogisticMapGenerator(Generator):
    """Logistic map ``x <- r x (1 - x)`` quantised to channels.

    Chaotic for ``r`` in roughly ``[3.57, 4]`` (fully chaotic at ``r = 4``). The
    hidden parameters are the map parameter ``r`` and the initial condition
    ``x0``. The parametric-chaos-ID expert recovers ``r``; the phase-space
    predictor forecasts short-horizon without knowing the family.
    """

    def __init__(self, n_channels: int, r: float, x0: float, burn_in: int = 64):
        super().__init__(n_channels)
        self.r = float(r)
        self._x0 = float(x0)
        self.burn_in = int(burn_in)
        self.reset()

    @classmethod
    def create(cls, n_channels: int, rng: np.random.Generator | None = None,
               r: float = 3.95, x0: float | None = None, burn_in: int = 64):
        rng = rng or np.random.default_rng()
        if x0 is None:
            x0 = float(rng.uniform(0.05, 0.95))
        return cls(n_channels=n_channels, r=r, x0=x0, burn_in=burn_in)

    def reset(self) -> None:
        x = self._x0
        for _ in range(self.burn_in):  # discard the transient
            x = self.r * x * (1.0 - x)
        self.x = x

    def next_channel(self) -> int:
        self.x = self.r * self.x * (1.0 - self.x)
        return _chaos_channel(self.x, self.n_channels)

    def describe(self) -> dict:
        return {"name": self.name, "n_channels": self.n_channels, "r": self.r}


@register("tent")
class TentMapGenerator(Generator):
    """Tent map ``x <- mu * min(x, 1 - x)`` quantised to channels.

    Chaotic for ``mu`` near 2 (fully chaotic at ``mu = 2``). Parameter ``mu``.
    """

    def __init__(self, n_channels: int, mu: float, x0: float, burn_in: int = 64):
        super().__init__(n_channels)
        self.mu = float(mu)
        self._x0 = float(x0)
        self.burn_in = int(burn_in)
        self.reset()

    @classmethod
    def create(cls, n_channels: int, rng: np.random.Generator | None = None,
               mu: float = 1.99, x0: float | None = None, burn_in: int = 64):
        rng = rng or np.random.default_rng()
        if x0 is None:
            x0 = float(rng.uniform(0.05, 0.95))
        return cls(n_channels=n_channels, mu=mu, x0=x0, burn_in=burn_in)

    def _map(self, x: float) -> float:
        return self.mu * (x if x < 0.5 else 1.0 - x)

    def reset(self) -> None:
        x = self._x0
        for _ in range(self.burn_in):
            x = self._map(x)
        self.x = min(max(x, 0.0), 1.0 - 1e-12)

    def next_channel(self) -> int:
        self.x = min(max(self._map(self.x), 0.0), 1.0 - 1e-12)
        return _chaos_channel(self.x, self.n_channels)

    def describe(self) -> dict:
        return {"name": self.name, "n_channels": self.n_channels, "mu": self.mu}


@register("henon")
class HenonMapGenerator(Generator):
    """Hénon map ``x <- 1 - a x^2 + y, y <- b x`` with the x-coordinate quantised.

    Chaotic at the classic ``a = 1.4, b = 0.3``. The x-coordinate (roughly in
    ``[-1.5, 1.5]``) is affinely mapped to ``[0, 1)`` before quantisation.
    """

    X_LO, X_HI = -1.5, 1.5

    def __init__(self, n_channels: int, a: float, b: float, x0: float, y0: float,
                 burn_in: int = 64):
        super().__init__(n_channels)
        self.a = float(a)
        self.b = float(b)
        self._x0 = float(x0)
        self._y0 = float(y0)
        self.burn_in = int(burn_in)
        self.reset()

    @classmethod
    def create(cls, n_channels: int, rng: np.random.Generator | None = None,
               a: float = 1.4, b: float = 0.3, x0: float | None = None,
               y0: float | None = None, burn_in: int = 64):
        rng = rng or np.random.default_rng()
        if x0 is None:
            x0 = float(rng.uniform(-0.3, 0.3))
        if y0 is None:
            y0 = float(rng.uniform(-0.3, 0.3))
        return cls(n_channels=n_channels, a=a, b=b, x0=x0, y0=y0, burn_in=burn_in)

    def reset(self) -> None:
        x, y = self._x0, self._y0
        for _ in range(self.burn_in):
            x, y = 1.0 - self.a * x * x + y, self.b * x
        self.x, self.y = x, y

    def next_channel(self) -> int:
        self.x, self.y = 1.0 - self.a * self.x * self.x + self.y, self.b * self.x
        u = (self.x - self.X_LO) / (self.X_HI - self.X_LO)
        return _chaos_channel(min(max(u, 0.0), 1.0 - 1e-12), self.n_channels)

    def describe(self) -> dict:
        return {"name": self.name, "n_channels": self.n_channels, "a": self.a, "b": self.b}

# ---- full-state LCG (for the algebraic LCG-cryptanalysis expert) ----


@register("lcg_full")
class FullStateLCGGenerator(Generator):
    """LCG whose modulus equals the channel count, so the channel IS the state.

    ``x <- (a x + c) mod M`` with the channel reported as ``x`` itself. Because
    the full state is observed, the algebraic LCG-cryptanalysis expert can
    recover ``(a, c)`` from a few consecutive clean outputs (unlike the
    high-bit-truncated :class:`LCGGenerator`, which hides the low bits and is
    attacked by seed search instead). Defaults give a full-period generator for a
    power-of-two ``M`` (Hull-Dobell: ``a = 1 mod 4``, ``c`` odd).
    """

    def __init__(self, n_channels: int, mult: int, incr: int, seed: int):
        super().__init__(n_channels)
        self.m = int(n_channels)
        self.a = int(mult) % self.m
        self.c = int(incr) % self.m
        self._seed = int(seed) % self.m
        self.x = self._seed

    @classmethod
    def create(cls, n_channels: int, rng: np.random.Generator | None = None,
               mult: int = 5, incr: int = 3, seed: int | None = None):
        rng = rng or np.random.default_rng()
        if seed is None:
            seed = int(rng.integers(0, n_channels))
        return cls(n_channels=n_channels, mult=mult, incr=incr, seed=seed)

    def reset(self) -> None:
        self.x = self._seed

    def next_channel(self) -> int:
        self.x = (self.a * self.x + self.c) % self.m
        return self.x

    def describe(self) -> dict:
        return {"name": self.name, "n_channels": self.n_channels,
                "modulus": self.m, "mult": self.a, "incr": self.c}
