"""Posterior calibration: reliability curve and expected calibration error.

The bank reports a posterior, so it can be over- or under-confident. ECE and the
reliability diagram (paper Section 11, "posterior calibration") measure whether
a declared confidence of, say, 0.9 corresponds to being right 90% of the time.
Inputs are per-episode ``(confidence, correct)`` pairs, where confidence is the
posterior mass on the decided label.
"""

from __future__ import annotations

import numpy as np


def reliability_curve(confidences, correct, n_bins: int = 10):
    """Return ``(bin_centers, bin_accuracy, bin_confidence, bin_count)``."""
    conf = np.asarray(confidences, dtype=np.float64)
    corr = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    acc = np.full(n_bins, np.nan)
    avg_conf = np.full(n_bins, np.nan)
    count = np.zeros(n_bins, dtype=np.int64)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, n_bins - 1)
    for b in range(n_bins):
        mask = idx == b
        count[b] = int(mask.sum())
        if count[b] > 0:
            acc[b] = float(corr[mask].mean())
            avg_conf[b] = float(conf[mask].mean())
    return centers, acc, avg_conf, count


def ece(confidences, correct, n_bins: int = 10) -> float:
    """Expected calibration error: weighted mean ``|accuracy - confidence|``."""
    conf = np.asarray(confidences, dtype=np.float64)
    _, acc, avg_conf, count = reliability_curve(conf, correct, n_bins)
    total = count.sum()
    if total == 0:
        return float("nan")
    mask = count > 0
    return float(np.sum(count[mask] / total * np.abs(acc[mask] - avg_conf[mask])))
