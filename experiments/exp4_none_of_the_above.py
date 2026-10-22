"""Experiment 4: the none-of-the-above test.

Out-of-bank generators and a keyed CSPRNG: the bank's H_0 should win and no
wrong generator should be declared. A plain pairwise SPRT, lacking H_0, is
forced to name a generator and therefore false-identifies. This is the
experiment that shows H_0 earns its place (paper Section 11, item 4).
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

NAME = "exp4_none_of_the_above"


def compute(*, M=C.DEFAULT_M, window=C.DEFAULT_WINDOW, n_seeds=40, p_d=0.85, p_fa=0.02,
            generators=("csprng", "filtered_lfsr")) -> dict:
    ch = ChannelSet.of_size(M)
    specs = C.default_specs(M)
    tgt = C.default_target_labels(specs)
    sensor = SensorModel(p_d=p_d, p_fa=p_fa)
    emit = SensorAwareEmission(sensor)
    bank = C.make_bank(ch, window, sensor, specs)
    sprt = PairwiseSPRT([build_hypothesis(s, ch, window) for s in specs], emit, M)

    out = {"meta": {"M": M, "window": window, "n_seeds": n_seeds, "p_d": p_d, "p_fa": p_fa},
           "generators": {}}
    for gen in generators:
        bank_res, sprt_res = [], []
        for s in C.seeds(n_seeds):
            ep = C.episode_for(gen, ch, sensor, window, s, target_labels=tgt)
            bank_res.append(run_bank(bank, ep))
            sprt_res.append(run_sprt(sprt, ep))
        out["generators"][gen] = {
            "bank": _breakdown(bank_res),
            "pairwise_sprt": _breakdown(sprt_res),
        }
    return out


def _breakdown(results) -> dict:
    labels = [r.decision_label for r in results]
    uniq = sorted(set(labels))
    dist = {u: float(np.mean([l == u for l in labels])) for u in uniq}
    return {"routed_to_unknown": float(np.mean([r.decision_label == "unknown" for r in results])),
            "false_id_rate": float(np.mean([r.false_id for r in results])),
            "hard_false_id_rate": float(np.mean([C.is_hard_false_id(r) for r in results])),
            "accuracy": float(np.mean([r.correct for r in results])),
            "decision_distribution": dist}


def plot(results: dict, path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    gens = list(results["generators"].keys())
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    x = np.arange(len(gens))
    w = 0.27
    bars = {
        "bank: claims a deterministic break": (
            [results["generators"][g]["bank"]["hard_false_id_rate"] for g in gens],
            style.PALETTE["blue"], -1),
        "bank: any wrong name (incl. markov fallback)": (
            [results["generators"][g]["bank"]["false_id_rate"] for g in gens],
            style.PALETTE["sky"], 0),
        "pairwise SPRT (no H_0)": (
            [results["generators"][g]["pairwise_sprt"]["false_id_rate"] for g in gens],
            style.PALETTE["vermillion"], 1),
    }
    for label, (vals, color, off) in bars.items():
        bb = ax.bar(x + off * w, vals, w, label=label, color=color)
        for b, v in zip(bb, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                    ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f"{g}\n(out-of-bank)" for g in gens])
    ax.set_ylabel("false-identification rate"); ax.set_ylim(0, 1.18)
    ax.set_title("None-of-the-above: the bank never falsely claims an algebraic break")
    ax.legend(fontsize=8, loc="upper left")
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
    for g, d in results["generators"].items():
        print(f"  {g}: bank unknown={d['bank']['routed_to_unknown']:.2f} "
              f"hard_false_id={d['bank']['hard_false_id_rate']:.2f} "
              f"any_false_id={d['bank']['false_id_rate']:.2f} | "
              f"sprt false_id={d['pairwise_sprt']['false_id_rate']:.2f}")


if __name__ == "__main__":
    main()
