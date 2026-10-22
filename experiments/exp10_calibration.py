"""Experiment 10: posterior calibration (reliability diagram + ECE).

The bank reports a posterior, so a declared confidence of 0.9 should mean it is
right about 90% of the time. We collect (confidence, correct) pairs over a mixed
episode set -- confidence being the posterior mass on the decided label at the
decision slot -- and plot the reliability curve with the expected calibration
error (paper Section 11, "posterior calibration").
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments import common as C
from experiments import style
from fhew.channels import ChannelSet
from fhew.logging_utils import load_json, save_json
from fhew.metrics import ece, reliability_curve
from fhew.sensor import SensorModel
from fhew.sim import run_bank

NAME = "exp10_calibration"


def compute(*, M=C.DEFAULT_M, window=C.DEFAULT_WINDOW, n_seeds=80, p_d=0.75, p_fa=0.03,
            generators=("lfsr", "weak_seed", "lcg", "markov", "csprng", "filtered_lfsr"),
            n_bins=10) -> dict:
    ch = ChannelSet.of_size(M)
    specs = C.default_specs(M)
    tgt = C.default_target_labels(specs)
    sensor = SensorModel(p_d=p_d, p_fa=p_fa)
    bank = C.make_bank(ch, window, sensor, specs)
    conf, correct = [], []
    for gen in generators:
        for s in C.seeds(n_seeds):
            r = run_bank(bank, C.episode_for(gen, ch, sensor, window, s, target_labels=tgt))
            # confidence = posterior on the decided label at the decision slot
            lab = "H0" if r.decision_label == "unknown" else r.decision_label
            conf.append(float(r.posterior_final.get(lab, max(r.posterior_final.values()))))
            correct.append(bool(r.correct))
    centers, acc, avg_conf, count = reliability_curve(conf, correct, n_bins)
    return {"meta": {"M": M, "window": window, "n_seeds": n_seeds, "p_d": p_d, "p_fa": p_fa,
                     "n_samples": len(conf)},
            "ece": ece(conf, correct, n_bins),
            "bin_centers": centers.tolist(),
            "bin_accuracy": [None if np.isnan(a) else float(a) for a in acc],
            "bin_confidence": [None if np.isnan(a) else float(a) for a in avg_conf],
            "bin_count": count.tolist()}


def plot(results: dict, path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    centers = np.asarray(results["bin_centers"])
    acc = np.array([np.nan if a is None else a for a in results["bin_accuracy"]])
    count = np.asarray(results["bin_count"])
    fig, ax = plt.subplots(figsize=(6.0, 5.4))
    ax.plot([0, 1], [0, 1], ls="--", color=style.PALETTE["grey"], label="perfect calibration")
    mask = ~np.isnan(acc)
    ax.plot(centers[mask], acc[mask], "-o", color=style.PALETTE["blue"], label="bank")
    # bar of bin counts (secondary axis)
    ax2 = ax.twinx()
    ax2.bar(centers, count, width=0.08, alpha=0.15, color=style.PALETTE["blue"])
    ax2.set_ylabel("samples per bin"); ax2.set_ylim(0, max(count.max(), 1) * 3)
    ax.set_xlabel("declared confidence"); ax.set_ylabel("empirical accuracy")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(f"Posterior calibration  (ECE = {results['ece']:.3f}, "
                 f"n = {results['meta']['n_samples']})")
    ax.legend(loc="upper left")
    return style.save(fig, path)


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n-seeds", type=int, default=80)
    p.add_argument("--plot-only", action="store_true")
    args = p.parse_args(argv)
    jpath = C.RESULTS_DIR / f"{NAME}.json"
    results = load_json(jpath) if args.plot_only else compute(n_seeds=args.n_seeds)
    if not args.plot_only:
        save_json(results, jpath)
    fig = plot(results, C.FIG_DIR / f"{NAME}.png")
    print(f"[{NAME}] wrote {jpath} and {fig}  ECE={results['ece']:.3f}")


if __name__ == "__main__":
    main()
