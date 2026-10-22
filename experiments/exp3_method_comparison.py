"""Experiment 3: method comparison on identical data.

Bayesian bank + MSPRT against pairwise SPRT, a fixed-length test, and the
learned classifier -- every method sees the same episodes and the same
sensor-aware likelihoods. The headline is that the bank matches the others on
in-bank generators while keeping the false-ID rate low on out-of-bank emitters
(which only the bank's H_0 and the learned 'unknown' class can handle). A paired
bootstrap and a McNemar test accompany the accuracy comparison.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments import common as C
from experiments import style
from fhew.baselines import FixedLengthTest, LearnedClassifier, PairwiseSPRT
from fhew.channels import ChannelSet
from fhew.likelihood.emission import SensorAwareEmission
from fhew.logging_utils import load_json, save_json
from fhew.metrics import mcnemar_test, paired_bootstrap_diff, summarize
from fhew.sensor import SensorModel
from fhew.sim.engine import run_bank, run_fixed_length, run_learned, run_sprt

NAME = "exp3_method_comparison"


def compute(*, M=C.DEFAULT_M, window=C.DEFAULT_WINDOW, n_seeds=30, p_d=0.8, p_fa=0.02,
            generators=("lfsr", "weak_seed", "lcg", "markov", "csprng", "filtered_lfsr"),
            classifier_path="data/learned_classifier.json") -> dict:
    ch = ChannelSet.of_size(M)
    specs = C.default_specs(M)
    tgt = C.default_target_labels(specs)
    sensor = SensorModel(p_d=p_d, p_fa=p_fa)
    emit = SensorAwareEmission(sensor)

    bank = C.make_bank(ch, window, sensor, specs)

    # Build fresh hypothesis objects per method (stateful, not shared with the bank).
    from fhew.experts.hypothesis import build_hypothesis
    sprt = PairwiseSPRT([build_hypothesis(s, ch, window) for s in specs], emit, M)
    fixedl = FixedLengthTest([build_hypothesis(s, ch, window) for s in specs], emit, M,
                             n_fixed=30)
    clf = None
    try:
        clf = LearnedClassifier.from_dict(load_json(classifier_path))
    except Exception:
        clf = None

    per_method: dict[str, list] = {"bank": [], "pairwise_sprt": [], "fixed_length": []}
    if clf is not None:
        per_method["learned"] = []

    for s in C.seeds(n_seeds):
        for gen in generators:
            ep = C.episode_for(gen, ch, sensor, window, s, target_labels=tgt)
            per_method["bank"].append(run_bank(bank, ep))
            per_method["pairwise_sprt"].append(run_sprt(sprt, ep))
            per_method["fixed_length"].append(run_fixed_length(fixedl, ep))
            if clf is not None:
                per_method["learned"].append(run_learned(clf, ep, threshold=0.9))

    summaries = {m: summarize(r).as_dict() for m, r in per_method.items()}
    # significance: bank vs each other method on correctness (paired, same episode order)
    sig = {}
    for m in per_method:
        if m == "bank":
            continue
        sig[m] = {
            "paired_bootstrap_accuracy": paired_bootstrap_diff(
                per_method["bank"], per_method[m], metric="correct"),
            "mcnemar": mcnemar_test(per_method["bank"], per_method[m]),
        }
    # per-generator accuracy table
    per_gen = {}
    for m, res in per_method.items():
        byg = {}
        for gen in generators:
            rr = [r for r in res if r.true_label == gen]
            byg[gen] = {"accuracy": float(np.mean([r.correct for r in rr])),
                        "false_id": float(np.mean([r.false_id for r in rr]))}
        per_gen[m] = byg
    return {"meta": {"M": M, "window": window, "n_seeds": n_seeds, "p_d": p_d,
                     "p_fa": p_fa, "generators": list(generators)},
            "summaries": summaries, "significance_vs_bank": sig, "per_generator": per_gen}


def plot(results: dict, path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    methods = list(results["summaries"].keys())
    acc = [results["summaries"][m]["accuracy"] for m in methods]
    fid = [results["summaries"][m]["false_id_rate"] for m in methods]
    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    w = 0.38
    ax.bar(x - w / 2, acc, w, label="accuracy", color=style.PALETTE["blue"])
    ax.bar(x + w / 2, fid, w, label="false-ID rate", color=style.PALETTE["vermillion"])
    for xi, a, f in zip(x, acc, fid):
        ax.text(xi - w / 2, a + 0.01, f"{a:.2f}", ha="center", fontsize=9)
        ax.text(xi + w / 2, f + 0.01, f"{f:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("rate")
    ax.set_title(f"Method comparison (mixed in/out-of-bank, "
                 f"$P_d={results['meta']['p_d']}$, $P_{{fa}}={results['meta']['p_fa']}$)")
    ax.legend(loc="upper right")
    return style.save(fig, path)


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n-seeds", type=int, default=30)
    p.add_argument("--plot-only", action="store_true")
    args = p.parse_args(argv)
    jpath = C.RESULTS_DIR / f"{NAME}.json"
    results = load_json(jpath) if args.plot_only else compute(n_seeds=args.n_seeds)
    if not args.plot_only:
        save_json(results, jpath)
    fig = plot(results, C.FIG_DIR / f"{NAME}.png")
    print(f"[{NAME}] wrote {jpath} and {fig}")
    for m, s in results["summaries"].items():
        print(f"  {m:14s} acc={s['accuracy']:.3f} false_id={s['false_id_rate']:.3f} "
              f"hops={s['mean_hops_to_identify']:.1f}")


if __name__ == "__main__":
    main()
