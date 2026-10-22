"""Experiment 6: plain Berlekamp-Massey vs the gap-robust state-filtered bank.

Plain BM assumes clean consecutive outputs and fails under any missed detection.
The state-filtered variant (the LFSR hypothesis's particle filter over register
states) keeps a posterior through gaps. We sweep P_d for LFSR emitters and report
the recovery / lock rate of each, showing where plain BM breaks and the
gap-robust variant holds (paper Section 11, item 6, and the recovery
characterization in Section 10).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments import common as C
from experiments import style
from fhew.channels import ChannelSet
from fhew.experts import build_hypothesis, HypothesisSpec
from fhew.likelihood import SensorAwareEmission
from fhew.logging_utils import load_json, save_json
from fhew.recovery import recover_and_predict
from fhew.sensor import SensorModel, clean_subsequence
from fhew.sim import make_episode

NAME = "exp6_bm_vs_filtered"


def _bm_recovers(ep, b, horizon) -> bool:
    """Plain BM on the clean single-detection subsequence (brittle to gaps)."""
    clean = clean_subsequence(ep.observations)
    chans = [c for _, c in clean]
    if len(chans) < 8:
        return False
    res = recover_and_predict(chans, b=b, horizon=5, max_length=12)
    return bool(res.recovered)


def _state_filter_locks(ep, ch, window, sensor, length, threshold=0.9) -> tuple[bool, int | None]:
    """State-filtered BM = LFSR-hypothesis particle filter; lock = posterior peak."""
    spec = HypothesisSpec("lfsr", "lfsr", "deterministic", grid={"length": [length]})
    hyp = build_hypothesis(spec, ch, window)
    emit = SensorAwareEmission(sensor)
    for t, o in enumerate(ep.observations):
        res = hyp.step(emit.log_emission(o))
        peak = float(hyp.candidate_posterior().max())
        if peak >= threshold:
            return True, t + 1
    return False, None


def compute(*, M=C.DEFAULT_M, window=C.DEFAULT_WINDOW, n_seeds=40, length=7,
            p_d_grid=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5), p_fa=0.0) -> dict:
    import math
    ch = ChannelSet.of_size(M)
    b = int(round(math.log2(M)))
    out = {"meta": {"M": M, "window": window, "n_seeds": n_seeds, "length": length,
                    "p_fa": p_fa, "p_d_grid": list(p_d_grid)},
           "bm": [], "state_filtered": []}
    for pd in p_d_grid:
        sensor = SensorModel(p_d=pd, p_fa=p_fa)
        bm_hits, sf_hits, sf_taus = [], [], []
        for s in C.seeds(n_seeds):
            rng = np.random.default_rng(s)
            ep = make_episode("lfsr", ch, sensor, window, rng, gen_params={"length": length})
            bm_hits.append(_bm_recovers(ep, b, window))
            locked, tau = _state_filter_locks(ep, ch, window, sensor, length)
            sf_hits.append(locked)
            if locked:
                sf_taus.append(tau)
        out["bm"].append({"p_d": pd, "recovery_rate": float(np.mean(bm_hits))})
        out["state_filtered"].append({
            "p_d": pd, "recovery_rate": float(np.mean(sf_hits)),
            "mean_lock_slot": float(np.mean(sf_taus)) if sf_taus else float("nan")})
    return out


def plot(results: dict, path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    pd = [r["p_d"] for r in results["bm"]]
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.plot(pd, [r["recovery_rate"] for r in results["bm"]], "-o",
            color=style.PALETTE["vermillion"], label="plain Berlekamp-Massey")
    ax.plot(pd, [r["recovery_rate"] for r in results["state_filtered"]], "-o",
            color=style.PALETTE["blue"], label="state-filtered (gap-robust)")
    ax.set_xlabel("detection probability $P_d$"); ax.invert_xaxis()
    ax.set_ylabel("recovery / lock rate"); ax.set_ylim(-0.02, 1.05)
    ax.set_title(f"LFSR recovery under gaps (L={results['meta']['length']}, "
                 f"$P_{{fa}}={results['meta']['p_fa']}$)")
    ax.legend(loc="lower left")
    return style.save(fig, path)


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n-seeds", type=int, default=40)
    p.add_argument("--plot-only", action="store_true")
    args = p.parse_args(argv)
    jpath = C.RESULTS_DIR / f"{NAME}.json"
    results = load_json(jpath) if args.plot_only else compute(n_seeds=args.n_seeds)
    if not args.plot_only:
        save_json(results, jpath)
    fig = plot(results, C.FIG_DIR / f"{NAME}.png")
    print(f"[{NAME}] wrote {jpath} and {fig}")
    for bm, sf in zip(results["bm"], results["state_filtered"]):
        print(f"  Pd={bm['p_d']}: BM={bm['recovery_rate']:.2f} "
              f"state_filtered={sf['recovery_rate']:.2f}")


if __name__ == "__main__":
    main()
