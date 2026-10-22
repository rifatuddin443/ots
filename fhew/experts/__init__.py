"""The expert interface, the hypothesis wrappers, H_0, and the bank.

The reuse rule (paper Section 12, item 1) is that the bank has the *same*
interface as every other expert in the thesis: ``update(observation)``,
``predict()`` returning a next-hop distribution, plus ``posterior()`` and a
``stopped`` flag. That is the :class:`~fhew.experts.base.Expert` protocol here,
and :class:`~fhew.experts.bank.HypothesisBank` implements it, so the journal's
Fixed-Share mixture can consume the bank as one expert with no change.
"""

from fhew.experts.base import Expert
from fhew.experts.bank import BankOutput, HypothesisBank
from fhew.experts.hypothesis import (
    DeterministicHypothesis,
    Hypothesis,
    HypothesisSpec,
    MarkovHypothesis,
    build_hypothesis,
)
from fhew.experts.h0 import H0Hypothesis

__all__ = [
    "Expert",
    "Hypothesis",
    "HypothesisSpec",
    "DeterministicHypothesis",
    "MarkovHypothesis",
    "H0Hypothesis",
    "build_hypothesis",
    "HypothesisBank",
    "BankOutput",
]
