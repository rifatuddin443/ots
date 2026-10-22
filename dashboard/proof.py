"""Small, dashboard-sized reruns of the paper's two clearest ablations.

These follow the protocols of ``experiments/exp6_bm_vs_filtered.py`` and
``experiments/exp7_sensor_ablation.py`` with fewer seeds, built from the same
``fhew`` primitives, so the dashboard does not need the plotting extras the
experiment modules import:

* :func:`bm_vs_filtered` -- textbook Berlekamp-Massey on the clean detections
  versus the bank's gap-robust state-filtered LFSR expert, across ``P_d``;
* :func:`sensor_ablation` -- the same bank with the sensor-aware likelihood
  versus the naive clean-match likelihood, across ``P_d`` at a stressing
  false-alarm rate.
"""

from __future__ import annotations

import math

import numpy as np

from experiments import common as C
from fhew.channels import ChannelSet
from fhew.experts import HypothesisSpec, build_hypothesis
from fhew.likelihood import SensorAwareEmission
from fhew.recovery import recover_and_predict
from fhew.sensor import SensorModel, clean_subsequence
from fhew.sim import make_episode, run_bank

P_D_GRID = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)


def _bm_recovers(ep, b: int) -> bool:
    chans = [c for _, c in clean_subsequence(ep.observations)]
    if len(chans) < 8:
        return False
    return bool(recover_and_predict(chans, b=b, horizon=5, max_length=12).recovered)


def _state_filter_locks(ep, ch, window, sensor, length, threshold=0.9) -> bool:
    hyp = build_hypothesis(HypothesisSpec("lfsr", "lfsr", "deterministic",
                                          grid={"length": [length]}), ch, window)
    emit = SensorAwareEmission(sensor)
    for o in ep.observations:
        hyp.step(emit.log_emission(o))
        if float(hyp.candidate_posterior().max()) >= threshold:
            return True
    return False


def bm_vs_filtered(n_seeds: int = 20, *, length: int = 7, p_fa: float = 0.0,
                   window: int = C.DEFAULT_WINDOW) -> dict:
    """Recovery rate of plain BM vs the state-filtered expert on an LFSR (exp6)."""
    ch = ChannelSet.of_size(C.DEFAULT_M)
    b = int(round(math.log2(ch.size)))
    rows = []
    for pd in P_D_GRID:
        sensor = SensorModel(p_d=pd, p_fa=p_fa)
        bm, sf = [], []
        for s in C.seeds(n_seeds):
            ep = make_episode("lfsr", ch, sensor, window, np.random.default_rng(s),
                              gen_params={"length": length})
            bm.append(_bm_recovers(ep, b))
            sf.append(_state_filter_locks(ep, ch, window, sensor, length))
        rows.append({"p_d": pd, "bm": float(np.mean(bm)), "state_filtered": float(np.mean(sf))})
    return {"n_seeds": n_seeds, "length": length, "p_fa": p_fa, "rows": rows}


def sensor_ablation(n_seeds: int = 12, *, p_fa: float = 0.05,
                    generators=("lfsr", "weak_seed", "lcg"),
                    window: int = C.DEFAULT_WINDOW) -> dict:
    """Accuracy and hops of the sensor-aware vs naive likelihood (exp7)."""
    ch = ChannelSet.of_size(C.DEFAULT_M)
    specs = C.default_specs(C.DEFAULT_M)
    tgt = C.default_target_labels(specs)
    rows = []
    for pd in P_D_GRID:
        sensor = SensorModel(p_d=pd, p_fa=p_fa)
        row = {"p_d": pd}
        for emission in ("sensor_aware", "naive"):
            bank = C.make_bank(ch, window, sensor, specs, emission_kind=emission)
            res = [run_bank(bank, C.episode_for(g, ch, sensor, window, s, target_labels=tgt))
                   for g in generators for s in C.seeds(n_seeds)]
            hops = [r.tau_stop for r in res if r.correct and r.declared]
            row[emission] = {"accuracy": float(np.mean([r.correct for r in res])),
                             "mean_hops": float(np.mean(hops)) if hops else None}
        rows.append(row)
    return {"n_seeds": n_seeds, "p_fa": p_fa, "generators": list(generators), "rows": rows}
