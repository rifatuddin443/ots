"""The Expert interface shared by the bank and every thesis expert.

Keeping one interface is design rule 1 of the reuse plan (paper Section 12): the
bank must look like any other predictor so the journal's Fixed-Share mixture can
weight it by realised skill with no special-casing.
"""

from __future__ import annotations

import abc

import numpy as np

from fhew.sensor import Observation


class Expert(abc.ABC):
    """A next-hop predictor that consumes the lossy observation stream.

    ``update`` returns the slot's reduced log-likelihood increment (useful to a
    log-loss mixture); ``predict`` returns the next-hop distribution over
    channels. Subclasses add their own state (a bank posterior, a neural hidden
    state, ...), but nothing outside this interface is required of them.
    """

    #: human-readable label, used in logs and metrics
    label: str = "expert"

    @abc.abstractmethod
    def update(self, obs: Observation) -> float:
        """Consume one slot's observation; return the reduced log-likelihood."""

    @abc.abstractmethod
    def predict(self) -> np.ndarray:
        """Return the next-hop distribution ``p(c)`` over channels."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Return to the pre-engagement state."""
