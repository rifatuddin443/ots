"""LFSR and nonlinear-filtered LFSR hop generators.

A length-``L`` Fibonacci LFSR over GF(2) produces a bit stream obeying a linear
recurrence of degree ``L``. When the channel-set size ``M`` is a power of two, a
channel is ``b = log2(M)`` consecutive output bits, so the *channel* stream
expands losslessly to the *bit* stream and Berlekamp-Massey can recover the
recurrence from clean outputs (paper Sections 6 and 10). The filtered variant
passes the register through a nonlinear boolean combiner, which hides the linear
structure and is where plain BM should fail.
"""

from __future__ import annotations

import math

import numpy as np

from fhew.generators.base import Generator, register


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
