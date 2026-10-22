"""Generator library and Berlekamp-Massey recovery."""

import numpy as np
import pytest

from fhew.generators import make_generator, registered_names
from fhew.recovery import (
    berlekamp_massey_gf2,
    bits_to_channels,
    channels_to_bits,
    recover_and_predict,
)

M, B = 16, 4


def test_all_generators_registered():
    for name in ("lfsr", "filtered_lfsr", "lcg", "weak_seed", "markov", "csprng"):
        assert name in registered_names()


@pytest.mark.parametrize("name,kw", [
    ("lfsr", {"length": 7}), ("filtered_lfsr", {"length": 11}), ("lcg", {}),
    ("weak_seed", {"keyspace": 256}), ("markov", {"alpha": 0.2}), ("csprng", {}),
])
def test_generator_sequences_in_range(name, kw):
    rng = np.random.default_rng(0)
    g = make_generator(name, M, rng=rng, **kw)
    seq = g.sequence(50)
    assert seq.min() >= 0 and seq.max() < M
    assert len(seq) == 50


def test_lfsr_reset_reproducible():
    g = make_generator("lfsr", M, length=7, state=1, taps=0b1000001)
    a = g.sequence(30)
    g.reset()
    b = g.sequence(30)
    assert np.array_equal(a, b)


def test_bit_channel_roundtrip():
    rng = np.random.default_rng(3)
    chans = rng.integers(0, M, size=40)
    bits = channels_to_bits(chans, B)
    back = bits_to_channels(bits, B)
    assert np.array_equal(chans, back)


def test_bm_recovers_lfsr_bitstream():
    g = make_generator("lfsr", M, length=7, state=1, taps=0b1000001)
    bits = channels_to_bits(g.sequence(40), B)
    L, coeffs = berlekamp_massey_gf2(bits[:60])
    assert L <= 7


@pytest.mark.parametrize("seed", range(10))
def test_bm_recovers_and_predicts_true_lfsr(seed):
    rng = np.random.default_rng(seed)
    g = make_generator("lfsr", M, rng=rng, length=7)
    seq = g.sequence(50)
    res = recover_and_predict(seq[:24], b=B, horizon=20, max_length=16)
    assert res.recovered
    assert np.array_equal(res.predicted_channels[:20], seq[24:44])


@pytest.mark.parametrize("name,kw", [("filtered_lfsr", {"length": 11}), ("csprng", {})])
def test_bm_does_not_recover_nonlinear(name, kw):
    rng = np.random.default_rng(5)
    g = make_generator(name, M, rng=rng, **kw)
    seq = g.sequence(90)
    res = recover_and_predict(seq[:70], b=B, horizon=20, max_length=16)
    assert not res.recovered
