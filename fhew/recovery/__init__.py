"""Algebraic and gap-robust recovery for LFSR-type generators."""

from fhew.recovery.berlekamp_massey import (
    BMResult,
    berlekamp_massey_gf2,
    bits_to_channels,
    channels_to_bits,
    linear_complexity_profile,
    recover_and_predict,
    run_lfsr_from_coeffs,
)

__all__ = [
    "berlekamp_massey_gf2",
    "linear_complexity_profile",
    "channels_to_bits",
    "bits_to_channels",
    "run_lfsr_from_coeffs",
    "recover_and_predict",
    "BMResult",
]
