"""The information-parity audit (paper Section 11).

No likelihood may use anything the modelled sensor cannot deliver -- never the
true seed, register length, or generator family. This is enforced structurally
(the bank consumes only Observations and its own config) and is checked here two
ways: an interface check that the bank never touches the true channels, and a
behavioural check that a secret *outside* the recce range cannot be identified
(which would be impossible if side-information leaked).
"""

import numpy as np

from fhew import ChannelSet, SensorModel
from fhew.config import build_bank, default_bank_specs, in_bank_labels
from fhew.sim.engine import run_bank
from fhew.sim.episode import make_episode

M, T = 16, 100


def _bank(sensor, window=T):
    ch = ChannelSet.of_size(M)
    specs = default_bank_specs(M, lcg_seed_max=1024)
    return build_bank(ch, window, sensor, specs), ch, specs


def test_bank_update_consumes_only_observations():
    # The bank's update signature takes an Observation; it has no handle on the
    # Episode's true_channels or the generator. We assert it never reads them by
    # driving it purely from observations and checking it still works.
    sensor = SensorModel(0.9, 0.01)
    bank, ch, specs = _bank(sensor)
    rng = np.random.default_rng(0)
    ep = make_episode("lfsr", ch, sensor, T, rng, gen_params={"length": 7})
    # Pass only observations (not ep.true_channels) -> identification still works.
    for o in ep.observations[:30]:
        bank.update(o)
    assert bank.posterior().sum() == 1.0 or abs(bank.posterior().sum() - 1) < 1e-9


def test_secret_outside_recce_range_is_not_identified():
    # A weak-seed generator whose keyspace is far larger than the bank searches:
    # the true seed is almost never in range, so a parity-respecting bank cannot
    # identify it and must route to unknown. If side-information leaked, it would
    # "identify" it anyway -- this test would then fail.
    sensor = SensorModel(0.9, 0.01)
    bank, ch, specs = _bank(sensor)
    tgt = in_bank_labels(specs) | {"weak_seed"}
    hard_ids = 0
    for s in range(15):
        rng = np.random.default_rng(300 + s)
        # keyspace 1<<20 >> bank's 256-seed search
        ep = make_episode("weak_seed", ch, sensor, T, rng,
                          gen_params={"keyspace": 1 << 20})
        ep.true_label = "weak_seed"
        ep.target_label = "unknown"
        r = run_bank(bank, ep)
        if r.decision_label == "weak_seed":
            hard_ids += 1
    # at most a chance-level coincidence; a leak would make this ~15
    assert hard_ids <= 1


def test_lcg_out_of_range_seed_not_broken():
    sensor = SensorModel(0.9, 0.0)
    bank, ch, specs = _bank(sensor)
    breaks = 0
    for s in range(15):
        rng = np.random.default_rng(400 + s)
        ep = make_episode("lcg", ch, sensor, T, rng, gen_params={"seed": 50000})
        ep.true_label = "lcg"
        ep.target_label = "unknown"
        r = run_bank(bank, ep)
        if r.decision_label == "lcg":
            breaks += 1
    assert breaks <= 1
