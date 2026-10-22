"""Experiment 13: large-scale per-type identification across the whole library.

The comprehensive sweep requested as a huge, varied, all-types dataset: every
generator family (LFSR, filtered-LFSR, LCG, full-state LCG, weak-seed, logistic
/ tent / Henon chaos, Markov, and the keyed CSPRNG) is run through the
one-expert-per-family identification bank across a grid of detection and
false-alarm probabilities with many seeds each. It reports per-type accuracy,
hops-to-identify and false-ID, plus the overall dataset size.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments import common as C
from experiments import style
from fhew.channels import ChannelSet
from fhew.config import GENERATOR_TARGET_FAMILY
from fhew.logging_utils import load_json, save_json
from fhew.metrics import summarize
from fhew.sensor import SensorModel
from fhew.sim import run_bank

NAME = "exp13_coverage"

# Declaring one of these families when it is not the truth is a *dangerous* false
# identification (a claimed algebraic break). Declaring 'markov'/'chaos'/'unknown'
# on a boundary emitter is a graceful fallback, tracked as plain false-ID only.
_ALGEBRAIC_FAMILIES = frozenset({"lfsr", "filtered_lfsr", "lcg", "lcg_full", "weak_seed"})


def _hard_false_id(results) -> float:
    return float(np.mean([
        r.decision_label in _ALGEBRAIC_FAMILIES and r.decision_label != r.target_label
        for r in results])) if results else float("nan")


def compute(*, M=C.DEFAULT_M, window=160, n_seeds=50,
            p_d_grid=(1.0, 0.9, 0.8, 0.7, 0.6),
            p_fa_grid=(0.0, 0.02, 0.05),
            generators=C.ALL_GENERATORS) -> dict:
    ch = ChannelSet.of_size(M)
    specs = C.id_specs(M)
    # one bank per (p_d, p_fa) condition, reused across seeds and generators
    out = {"meta": {"M": M, "window": window, "n_seeds": n_seeds,
                    "p_d_grid": list(p_d_grid), "p_fa_grid": list(p_fa_grid),
                    "generators": list(generators),
                    "experts": [s.label for s in specs]},
           "by_type": {g: [] for g in generators}, "n_episodes": 0}
    n_ep = 0
    for pfa in p_fa_grid:
        for pd in p_d_grid:
            sensor = SensorModel(p_d=pd, p_fa=pfa)
            bank = C.make_bank(ch, window, sensor, specs)
            for gen in generators:
                results = []
                target = GENERATOR_TARGET_FAMILY.get(gen, "unknown")
                for s in C.seeds(n_seeds):
                    ep = C.episode_for(gen, ch, sensor, window, s)
                    ep.target_label = target
                    bank.reset()
                    results.append(run_bank(bank, ep))
                n_ep += len(results)
                summ = summarize(results).as_dict()
                summ["hard_false_id_rate"] = _hard_false_id(results)
                out["by_type"][gen].append({"p_d": pd, "p_fa": pfa, **summ})
    out["n_episodes"] = n_ep
    # aggregate across all conditions per type
    agg = {}
    for gen, rows in out["by_type"].items():
        accs = [r["accuracy"] for r in rows]
        fids = [r["false_id_rate"] for r in rows]
        hard = [r["hard_false_id_rate"] for r in rows]
        hops = [r["mean_hops_to_identify"] for r in rows if not np.isnan(r["mean_hops_to_identify"])]
        agg[gen] = {"accuracy": float(np.mean(accs)), "false_id_rate": float(np.mean(fids)),
                    "hard_false_id_rate": float(np.mean(hard)),
                    "mean_hops": float(np.mean(hops)) if hops else float("nan")}
    out["aggregate_by_type"] = agg
    return out


def plot(results: dict, path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    gens = results["meta"]["generators"]
    # left: accuracy vs P_d per type (averaged over P_fa); right: aggregate bars
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(12.5, 5.0),
                                   gridspec_kw={"width_ratios": [1.3, 1]})
    pd_grid = results["meta"]["p_d_grid"]
    for i, gen in enumerate(gens):
        rows = results["by_type"][gen]
        acc_by_pd = []
        for pd in pd_grid:
            vals = [r["accuracy"] for r in rows if r["p_d"] == pd]
            acc_by_pd.append(np.mean(vals) if vals else np.nan)
        color = style.color_for(gen, style.GENERATOR_COLORS, i)
        ls = "--" if gen in ("csprng",) else "-"
        axl.plot(pd_grid, acc_by_pd, ls + "o", color=color, label=gen)
    axl.set_xlabel("detection probability $P_d$"); axl.invert_xaxis()
    axl.set_ylabel("identification accuracy"); axl.set_ylim(0, 1.03)
    axl.set_title("Accuracy vs $P_d$ (avg over $P_{fa}$), per family")
    axl.legend(fontsize=8, ncol=2, loc="lower left")

    agg = results["aggregate_by_type"]
    y = np.arange(len(gens))
    accs = [agg[g]["accuracy"] for g in gens]
    fids = [agg[g]["false_id_rate"] for g in gens]
    axr.barh(y - 0.2, accs, 0.4, color=style.PALETTE["blue"], label="accuracy")
    axr.barh(y + 0.2, fids, 0.4, color=style.PALETTE["vermillion"], label="false-ID")
    axr.set_yticks(y); axr.set_yticklabels(gens, fontsize=9); axr.invert_yaxis()
    axr.set_xlim(0, 1.05); axr.set_xlabel("rate (avg over all conditions)")
    axr.set_title("Aggregate per family")
    axr.legend(fontsize=9, loc="lower right")
    fig.suptitle(f"Large-scale coverage: {results['n_episodes']} episodes "
                 f"({len(gens)} families x {len(results['meta']['p_d_grid'])} $P_d$ x "
                 f"{len(results['meta']['p_fa_grid'])} $P_{{fa}}$ x "
                 f"{results['meta']['n_seeds']} seeds)", y=1.02)
    return style.save(fig, path)


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n-seeds", type=int, default=50)
    p.add_argument("--plot-only", action="store_true")
    args = p.parse_args(argv)
    jpath = C.RESULTS_DIR / f"{NAME}.json"
    results = load_json(jpath) if args.plot_only else compute(n_seeds=args.n_seeds)
    if not args.plot_only:
        save_json(results, jpath)
    fig = plot(results, C.FIG_DIR / f"{NAME}.png")
    print(f"[{NAME}] wrote {jpath} and {fig}  ({results['n_episodes']} episodes)")
    for g, a in results["aggregate_by_type"].items():
        hp = f"{a['mean_hops']:.1f}" if not np.isnan(a["mean_hops"]) else "-"
        print(f"  {g:14s} acc={a['accuracy']:.2f} false_id={a['false_id_rate']:.2f} "
              f"hard_false_id={a['hard_false_id_rate']:.2f} hops={hp}")


if __name__ == "__main__":
    main()
