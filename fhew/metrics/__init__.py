"""Metrics: identification, calibration, and next-hop prediction.

Metrics are computed from lists of :class:`~fhew.sim.engine.EpisodeResult`, so
every method and every experiment reports the same quantities the same way
(paper Section 11): hops-to-identify, accuracy, false-identification rate with
the nominal bound overlaid, posterior calibration (reliability + ECE), and the
next-hop log-loss / top-k that the journal reuses.
"""

from fhew.metrics.identification import (
    IdentificationSummary,
    bootstrap_ci,
    mcnemar_test,
    paired_bootstrap_diff,
    summarize,
    summarize_by_class,
)
from fhew.metrics.calibration import ece, reliability_curve
from fhew.metrics.prediction import nexthop_summary

__all__ = [
    "IdentificationSummary",
    "summarize",
    "summarize_by_class",
    "bootstrap_ci",
    "paired_bootstrap_diff",
    "mcnemar_test",
    "ece",
    "reliability_curve",
    "nexthop_summary",
]
