"""Experiment 5: the stale-bank test.

The reconnaissance bank is deliberately mismatched: the true generator's secret
lies *outside* the searched ranges (an LCG seed beyond the search keyspace, a
weak-seed keyspace larger than searched, an LFSR longer than any hypothesis).
The claim is that the bank degrades to *blind* -- it routes to H_0 (unknown)
rather than confidently naming the wrong generator. We report that the
false-identification rate stays low under staleness, unlike a plain SPRT.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments import common as C
from experiments import style
from fhew.baselines import PairwiseSPRT
from fhew.channels import ChannelSet
from fhew.experts import build_hypothesis
from fhew.likelihood import SensorAwareEmission
from fhew.logging_utils import load_json, save_json
from fhew.sensor import SensorModel
from fhew.sim import run_bank, run_sprt

NAME = "exp5_stale_bank"

# Each case: a true generator whose hidden parameter is OUTSIDE the bank's range.
STALE_CASES = {
    "lcg_seed_out_of_range": ("lcg", {"seed": 50000}),       # bank searches seeds < 1024
    "weak_seed_large_keyspace": ("weak_seed", {"keyspace": 65536}),  # bank searches 256
    "lfsr_too_long": ("lfsr", {"length": 10}),               # bank searches lengths 5..7
}


def compute(*, M=C.DEFAULT_M, window=C.DEFAULT_WINDOW, n_seeds=40, p_d=0.85, p_fa=0.02) -> dict:
    ch = ChannelSet.of_size(M)
    specs = C.default_specs(M)
    # Under staleness the correct target is 'unknown' (no in-range hypothesis matches).
    tgt = {"unknown"} | {s.label for s in specs}
    sensor = SensorModel(p_d=p_d, p_fa=p_fa)
    emit = SensorAwareEmission(sensor)
    bank = C.make_bank(ch, window, sensor, specs)
    sprt = PairwiseSPRT([build_hypothesis(s, ch, window) for s in specs], emit, M)

    out = {"meta": {"M": M, "window": window, "n_seeds": n_seeds, "p_d": p_d, "p_fa": p_fa},
           "cases": {}}
    for case, (gen, params) in STALE_CASES.items():
        bank_fid, bank_hard, sprt_fid, sprt_hard, routed = [], [], [], [], []
        for s in C.seeds(n_seeds):
            # force the out-of-range secret while keeping the family
            ep = C.episode_for(gen, ch, sensor, window, s, gen_params=params)
            ep.true_label = gen
            ep.target_label = "unknown"
            br = run_bank(bank, ep)
            sr = run_sprt(sprt, ep)
            bank_fid.append(br.false_id)
            bank_hard.append(C.is_hard_false_id(br))
            routed.append(br.decision_label == "unknown")
            sprt_fid.append(sr.false_id)
            sprt_hard.append(C.is_hard_false_id(sr))
        out["cases"][case] = {
            "true_family": gen, "stale_param": params,
            "bank_false_id": float(np.mean(bank_fid)),
            "bank_hard_false_id": float(np.mean(bank_hard)),
            "bank_routed_to_unknown": float(np.mean(routed)),
            "sprt_false_id": float(np.mean(sprt_fid)),
            "sprt_hard_false_id": float(np.mean(sprt_hard)),
        }
    return out


def plot(results: dict, path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    cases = list(results["cases"].keys())
    x = np.arange(len(cases))
    w = 0.27
    bank_hard = [results["cases"][c]["bank_hard_false_id"] for c in cases]
    bank_any = [results["cases"][c]["bank_false_id"] for c in cases]
    sprt = [results["cases"][c]["sprt_false_id"] for c in cases]
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.bar(x - w, bank_hard, w, label="bank: claims a deterministic break",
           color=style.PALETTE["blue"])
    ax.bar(x, bank_any, w, label="bank: any wrong name (incl. markov)",
           color=style.PALETTE["sky"])
    ax.bar(x + w, sprt, w, label="pairwise SPRT (no H_0)", color=style.PALETTE["vermillion"])
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in cases], fontsize=9)
    ax.set_ylabel("false-identification rate"); ax.set_ylim(0, 1.12)
    ax.set_title("Stale bank: degrade to blind, not to a false algebraic break")
    ax.legend(fontsize=8)
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
    for c, d in results["cases"].items():
        print(f"  {c:28s} bank_hard={d['bank_hard_false_id']:.2f} "
              f"bank_any={d['bank_false_id']:.2f} "
              f"routed_unknown={d['bank_routed_to_unknown']:.2f} "
              f"sprt={d['sprt_false_id']:.2f}")


if __name__ == "__main__":
    main()
