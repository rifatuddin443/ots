"""Experiment 12: full-expert-bank coverage confusion matrix.

Runs the complete thesis expert bank (every expert, including redundant ones per
family -- plain vs state-filtered BM, LCG seed-search vs algebraic cryptanalysis,
chaos-ID vs phase-space, Markov vs context-mixing vs behaviour, plus the barrage
floor and H_0) against every generator family in the library, and records which
*expert* wins. The diagonal is correct coverage; off-diagonal cells expose
cross-talk between experts. This is the single figure that shows the whole bank
is implemented and each expert fires on its intended target.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments import common as C
from experiments import style
from fhew.channels import ChannelSet
from fhew.config import EXPERT_TARGET_FAMILY, GENERATOR_TARGET_FAMILY
from fhew.logging_utils import load_json, save_json
from fhew.sensor import SensorModel
from fhew.sim import run_bank

NAME = "exp12_full_bank_confusion"


def compute(*, M=C.DEFAULT_M, window=140, n_seeds=40, p_d=0.95, p_fa=0.01,
            generators=C.ALL_GENERATORS) -> dict:
    ch = ChannelSet.of_size(M)
    specs = C.full_specs(M)
    expert_labels = ["H0"] + [s.label for s in specs]
    sensor = SensorModel(p_d=p_d, p_fa=p_fa)
    bank = C.make_bank(ch, window, sensor, specs)

    # rows = true generator, cols = winning expert (argmax posterior at window end)
    counts = {g: {e: 0 for e in expert_labels} for g in generators}
    per_gen_acc = {}
    for gen in generators:
        correct = 0
        for s in C.seeds(n_seeds):
            ep = C.episode_for(gen, ch, sensor, window, s)
            bank.reset()
            run_bank(bank, ep, stop_early=False, eval_nexthop=False)
            out = bank.output()
            win = "H0" if out.top_index == 0 else out.labels[out.top_index]
            counts[gen][win] += 1
            if EXPERT_TARGET_FAMILY.get(win, "unknown") == GENERATOR_TARGET_FAMILY.get(gen, "unknown"):
                correct += 1
        per_gen_acc[gen] = correct / n_seeds
    return {"meta": {"M": M, "window": window, "n_seeds": n_seeds, "p_d": p_d, "p_fa": p_fa},
            "expert_labels": expert_labels, "generators": list(generators),
            "counts": counts, "per_generator_family_accuracy": per_gen_acc}


def plot(results: dict, path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    gens = results["generators"]
    experts = results["expert_labels"]
    n = results["meta"]["n_seeds"]
    mat = np.array([[results["counts"][g][e] / n for e in experts] for g in gens])
    fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(experts)), 0.55 * len(gens) + 2))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(experts)))
    ax.set_xticklabels(experts, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(gens)))
    ax.set_yticklabels([f"{g}  (acc {results['per_generator_family_accuracy'][g]:.2f})"
                        for g in gens], fontsize=9)
    for i in range(len(gens)):
        for j in range(len(experts)):
            v = mat[i, j]
            if v > 0.02:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v > 0.5 else "black", fontsize=7)
    ax.set_xlabel("winning expert (argmax posterior)")
    ax.set_ylabel("true generator")
    ax.set_title(f"Full expert-bank coverage  ($P_d={results['meta']['p_d']}$, "
                 f"$P_{{fa}}={results['meta']['p_fa']}$, {n} seeds)")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="fraction of episodes")
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
    for g, a in results["per_generator_family_accuracy"].items():
        print(f"  {g:14s} family-accuracy={a:.2f}")


if __name__ == "__main__":
    main()
