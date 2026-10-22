"""Hand features of an observation prefix for the learned classifier.

A fixed-length feature vector summarises what a lossy observation prefix looks
like, so a plain classifier can be trained on it. The features deliberately echo
the structure the hypotheses test for -- channel occupancy, transition
regularity, repeat rate, and (for power-of-two channel sets) the linear
complexity of the clean-bit expansion that separates LFSR-like from random
streams. Under information parity these are computed from the *sensor output*
only, never from the true channel sequence.
"""

from __future__ import annotations

import math

import numpy as np

from fhew.channels import ChannelSet
from fhew.recovery import channels_to_bits, linear_complexity_profile
from fhew.sensor import Observation


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def episode_features(obs_list: list[Observation], channels: ChannelSet) -> np.ndarray:
    """Return a fixed-length feature vector for an observation prefix."""
    M = channels.size
    n_slots = max(1, len(obs_list))

    hist = np.zeros(M, dtype=np.float64)
    n_det_total = 0
    n_empty = 0
    clean_channels: list[int] = []
    for o in obs_list:
        nd = len(o.detections)
        n_det_total += nd
        if nd == 0:
            n_empty += 1
        for d in o.detections:
            hist[d.channel] += 1.0
        if nd == 1:
            clean_channels.append(o.detections[0].channel)

    hist_norm = hist / hist.sum() if hist.sum() > 0 else np.full(M, 1.0 / M)
    occ_entropy = _entropy(hist_norm) / math.log(M)

    frac_empty = n_empty / n_slots
    mean_det = n_det_total / n_slots

    clean = np.asarray(clean_channels, dtype=np.int64)
    n_clean = clean.size
    if n_clean >= 2:
        repeat_rate = float(np.mean(clean[1:] == clean[:-1]))
        bigram_entropy = _bigram_entropy(clean, M) / math.log(M)
    else:
        repeat_rate = 0.0
        bigram_entropy = 1.0

    lc_ratio = 0.0
    if _is_power_of_two(M) and n_clean >= 8:
        b = int(round(math.log2(M)))
        bits = channels_to_bits(clean, b)
        prof = linear_complexity_profile(bits)
        lc = prof[-1] if prof else 0
        lc_ratio = min(1.0, lc / max(1, bits.size / 2))

    frac_clean = n_clean / n_slots
    log_slots = math.log1p(n_slots) / math.log1p(500)

    scalars = np.array([
        occ_entropy, frac_empty, mean_det, repeat_rate, bigram_entropy,
        lc_ratio, frac_clean, log_slots,
    ], dtype=np.float64)
    return np.concatenate([hist_norm, scalars])


def feature_dim(channels: ChannelSet) -> int:
    return channels.size + 8


def _entropy(p: np.ndarray) -> float:
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum())


def _bigram_entropy(clean: np.ndarray, M: int) -> float:
    counts = np.ones((M, M), dtype=np.float64)  # Laplace
    for a, b in zip(clean[:-1], clean[1:]):
        counts[a, b] += 1.0
    T = counts / counts.sum(axis=1, keepdims=True)
    # weight rows by how often each state is visited
    visit = np.bincount(clean[:-1], minlength=M).astype(np.float64)
    visit = visit / visit.sum() if visit.sum() > 0 else np.full(M, 1.0 / M)
    row_ent = np.array([_entropy(T[i]) for i in range(M)])
    return float((visit * row_ent).sum())
