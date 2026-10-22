"""Berlekamp-Massey over GF(2) and LFSR prediction from observed channels.

This module merges what used to be the fhew/recovery/ subpackage; the public
names are unchanged, so `from fhew.recovery import X` still works.
"""

from __future__ import annotations

# ---- merged from fhew/recovery/berlekamp_massey.py ----
from dataclasses import dataclass

import numpy as np


def channels_to_bits(channels, b: int) -> np.ndarray:
    """Expand a channel sequence to the underlying output bit stream (MSB-first).

    Matches :class:`fhew.generators.lfsr.LFSRGenerator`, which assembles each
    channel from ``b`` consecutive output bits most-significant first.
    """
    channels = np.asarray(channels, dtype=np.int64)
    bits = np.empty(channels.size * b, dtype=np.int8)
    idx = 0
    for c in channels:
        for k in range(b - 1, -1, -1):
            bits[idx] = (int(c) >> k) & 1
            idx += 1
    return bits


def bits_to_channels(bits, b: int) -> np.ndarray:
    """Inverse of :func:`channels_to_bits` (drops a trailing partial channel)."""
    bits = np.asarray(bits, dtype=np.int8)
    n = bits.size // b
    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        v = 0
        for k in range(b):
            v = (v << 1) | int(bits[i * b + k])
        out[i] = v
    return out


def berlekamp_massey_gf2(s) -> tuple[int, list[int]]:
    """Return ``(L, coeffs)`` for the shortest LFSR over GF(2) generating ``s``.

    ``coeffs`` are ``c_1..c_L`` in the recurrence ``s_n = XOR_i c_i s_{n-i}``.
    """
    s = [int(x) & 1 for x in s]
    n = len(s)
    C = [0] * (n + 1)
    B = [0] * (n + 1)
    C[0] = B[0] = 1
    L = 0
    m = 1
    for N in range(n):
        d = s[N]
        for i in range(1, L + 1):
            d ^= C[i] & s[N - i]
        if d == 0:
            m += 1
        elif 2 * L <= N:
            T = C[:]
            for i in range(n + 1 - m):
                C[i + m] ^= B[i]
            L = N + 1 - L
            B = T
            m = 1
        else:
            for i in range(n + 1 - m):
                C[i + m] ^= B[i]
            m += 1
    return L, C[1 : L + 1]


def linear_complexity_profile(s) -> list[int]:
    """Linear complexity after each prefix -- climbs to ~n/2 for random input.

    A genuine LFSR plateaus at its length ``L``; a nonlinear-filtered or random
    stream keeps climbing, which is the signature that separates "no linear
    structure" from "not enough data yet" (paper Section 5).
    """
    s = [int(x) & 1 for x in s]
    n = len(s)
    C = [0] * (n + 1)
    B = [0] * (n + 1)
    C[0] = B[0] = 1
    L = 0
    m = 1
    profile = []
    for N in range(n):
        d = s[N]
        for i in range(1, L + 1):
            d ^= C[i] & s[N - i]
        if d == 0:
            m += 1
        elif 2 * L <= N:
            T = C[:]
            for i in range(n + 1 - m):
                C[i + m] ^= B[i]
            L = N + 1 - L
            B = T
            m = 1
        else:
            for i in range(n + 1 - m):
                C[i + m] ^= B[i]
            m += 1
        profile.append(L)
    return profile


def run_lfsr_from_coeffs(seed_bits, coeffs, length: int) -> int:
    """Predict the next bit from the last ``len(coeffs)`` bits via the recurrence."""
    acc = 0
    L = len(coeffs)
    for i in range(1, L + 1):
        if coeffs[i - 1]:
            acc ^= int(seed_bits[-i])
    return acc


@dataclass
class BMResult:
    """Outcome of an algebraic recovery attempt on a clean channel prefix."""

    length: int
    coeffs: list[int]
    consistent: bool
    predicted_channels: np.ndarray  # continuation predicted beyond the input
    bits_used: int

    @property
    def recovered(self) -> bool:
        return self.consistent and self.length > 0


def recover_and_predict(
    clean_channels,
    b: int,
    horizon: int,
    *,
    max_length: int | None = None,
    verify_tail: int | None = None,
) -> BMResult:
    """Recover an LFSR from clean channels and roll it forward ``horizon`` slots.

    Berlekamp-Massey always regenerates its own input, so regenerating the fit
    data proves nothing. Genuine recovery is accepted only when the recurrence
    fitted on a *prefix* correctly predicts a *held-out tail* that it never saw,
    and the recovered length satisfies the classic ``L <= (fit bits)/2`` rule.
    This rejects the spurious high-degree fits produced by nonlinear-filtered or
    random streams, whose linear complexity grows with the data. This is the
    algebraic path; under a lossy sensor the hypothesis bank uses the gap-robust
    particle filter instead.
    """
    bits = channels_to_bits(clean_channels, b)
    if bits.size < 4:
        return BMResult(0, [], False, np.empty(0, dtype=np.int64), bits.size)

    # Hold out a verification tail that BM never fits on.
    if verify_tail is None:
        verify_tail = max(b, bits.size // 4)
    verify_tail = min(verify_tail, bits.size - 2)
    fit_bits = bits[: bits.size - verify_tail]
    tail_bits = bits[bits.size - verify_tail :]

    L, coeffs = berlekamp_massey_gf2(fit_bits)
    if L == 0 or (max_length is not None and L > max_length):
        return BMResult(L, coeffs, False, np.empty(0, dtype=np.int64), fit_bits.size)

    # Classic genuineness rule: a real degree-L LFSR needs ~2L bits to pin down.
    if L > fit_bits.size // 2:
        return BMResult(L, coeffs, False, np.empty(0, dtype=np.int64), fit_bits.size)

    # Predict the held-out tail from the fitted recurrence.
    stream = list(fit_bits.tolist())
    for _ in range(verify_tail):
        stream.append(run_lfsr_from_coeffs(stream, coeffs, L))
    predicted_tail = stream[fit_bits.size :]
    consistent = predicted_tail == tail_bits.tolist()
    if not consistent:
        return BMResult(L, coeffs, False, np.empty(0, dtype=np.int64), fit_bits.size)

    # Recovery genuine: roll forward from the full clean stream to predict.
    full = list(bits.tolist())
    for _ in range(horizon * b):
        full.append(run_lfsr_from_coeffs(full, coeffs, L))
    pred_bits = np.asarray(full[bits.size :], dtype=np.int8)
    pred_channels = bits_to_channels(pred_bits, b)
    return BMResult(L, coeffs, True, pred_channels, fit_bits.size)
