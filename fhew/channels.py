"""The channel set C over which the emitter hops.

The emitter hops over a known channel set ``C`` of size ``M`` (paper Section 3).
Channels are indexed ``0 .. M-1``. The channel set is deliberately tiny: it
carries only the count plus optional metadata (a reconnaissance channel plan and
blacklist) that the journal's barrage floor reads. Keeping it a plain value
object means the gym core and the bank share exactly one notion of "channel".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ChannelSet:
    """A frozen description of the hop channel set.

    Parameters
    ----------
    size:
        Number of channels ``M``.
    plan:
        Optional reconnaissance channel plan: the subset of channel indices the
        emitter is believed to use. Used by the journal's barrage floor; here it
        only shapes the H_0 / barrage uniform distribution when supplied.
    blacklist:
        Channel indices the emitter never uses (guard channels). Excluded from
        the effective channel set.
    """

    size: int
    plan: tuple[int, ...] = field(default=())
    blacklist: tuple[int, ...] = field(default=())

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError("channel-set size must be positive")
        for name, idxs in (("plan", self.plan), ("blacklist", self.blacklist)):
            for c in idxs:
                if not 0 <= c < self.size:
                    raise ValueError(f"{name} channel {c} out of range [0, {self.size})")

    @property
    def M(self) -> int:  # noqa: N802 - matches the paper's notation
        return self.size

    @property
    def indices(self) -> np.ndarray:
        return np.arange(self.size, dtype=np.int64)

    def effective_indices(self) -> np.ndarray:
        """Channels the emitter may actually use: ``plan`` minus ``blacklist``.

        Falls back to all non-blacklisted channels when no plan is supplied.
        """
        black = set(self.blacklist)
        if self.plan:
            idxs = [c for c in self.plan if c not in black]
        else:
            idxs = [c for c in range(self.size) if c not in black]
        return np.asarray(idxs, dtype=np.int64)

    def barrage_floor(self) -> np.ndarray:
        """Uniform distribution over the effective channel set (the H_0 floor)."""
        eff = self.effective_indices()
        u = np.zeros(self.size, dtype=np.float64)
        u[eff] = 1.0 / len(eff)
        return u

    def uniform(self) -> np.ndarray:
        """Uniform distribution over all channels."""
        return np.full(self.size, 1.0 / self.size, dtype=np.float64)

    @classmethod
    def of_size(cls, size: int) -> "ChannelSet":
        return cls(size=size)

    @classmethod
    def from_config(cls, cfg: dict | int | "ChannelSet") -> "ChannelSet":
        if isinstance(cfg, ChannelSet):
            return cfg
        if isinstance(cfg, int):
            return cls(size=cfg)
        return cls(
            size=int(cfg["size"]),
            plan=tuple(int(c) for c in cfg.get("plan", ())),
            blacklist=tuple(int(c) for c in cfg.get("blacklist", ())),
        )


def as_channel_set(spec: "ChannelSet | int | dict") -> ChannelSet:
    return ChannelSet.from_config(spec)


def validate_sequence(seq: Sequence[int], channels: ChannelSet) -> None:
    arr = np.asarray(seq, dtype=np.int64)
    if arr.size and (arr.min() < 0 or arr.max() >= channels.size):
        raise ValueError("channel sequence contains out-of-range indices")
