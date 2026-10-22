"""Experiment 1 (headline): hops-to-identify vs P_d, per generator type.

The main figure of the paper. For each in-bank generator and each detection
probability, we measure the mean number of hops the bank needs to correctly
declare the generator (MSPRT at delta=0.05), with bootstrap CIs over seeds.
Lower P_d thins the evidence, so hops-to-identify should rise as P_d falls.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments import common as C
from experiments import style
from fhew.channels import ChannelSet
from fhew.logging_utils import load_json, save_json
from fhew.metrics import summarize
from fhew.sensor import SensorModel
from fhew.sim import run_bank

NAME = "exp1_hops_vs_pd"


def compute(*, M=C.DEFAULT_M, window=C.DEFAULT_WINDOW, n_seeds=30,
            p_d_grid=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5), p_fa=0.02,
            generators=C.IN_BANK_GENERATORS) -> dict:
    ch = ChannelSet.of_size(M)
    specs = C.default_specs(M)
    tgt = C.default_target_labels(specs)
    out = {"meta": {"M": M, "window": window, "n_seeds": n_seeds, "p_fa": p_fa,
                    "p_d_grid": list(p_d_grid), "delta": 0.05}, "series": {}}
    for pd in p_d_grid:
        sensor = SensorModel(p_d=pd, p_fa=p_fa)
        bank = C.make_bank(ch, window, sensor, specs)
        for gen in generators:
            res = [run_bank(bank, C.episode_for(gen, ch, sensor, window, s,
                                                target_labels=tgt))
                   for s in C.seeds(n_seeds)]
            summ = summarize(res).as_dict()
            out["series"].setdefault(gen, []).append({"p_d": pd, **summ})
    return out


def plot(results: dict, path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    fig, (axh, axa) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for i, (gen, rows) in enumerate(results["series"].items()):
        color = style.color_for(gen, style.GENERATOR_COLORS, i)
        pd = [r["p_d"] for r in rows]
        hops = [r["mean_hops_to_identify"] for r in rows]
        lo = [r["hops_ci"][0] for r in rows]
        hi = [r["hops_ci"][1] for r in rows]
        axh.plot(pd, hops, "-o", color=color, label=gen)
        style.band(axh, pd, lo, hi, color)
        acc = [r["accuracy"] for r in rows]
        axa.plot(pd, acc, "-o", color=color, label=gen)
    axh.set_xlabel("detection probability $P_d$")
    axh.set_ylabel("mean hops-to-identify")
    axh.set_title("Hops-to-identify vs $P_d$")
    axh.invert_xaxis()
    axa.set_xlabel("detection probability $P_d$")
    axa.set_ylabel("identification accuracy")
    axa.set_title("Accuracy vs $P_d$")
    axa.set_ylim(0, 1.03)
    axa.invert_xaxis()
    axa.legend(title="generator", loc="lower left")
    fig.suptitle(f"Sensor-aware bank, $P_{{fa}}={results['meta']['p_fa']}$, "
                 f"{results['meta']['n_seeds']} seeds", y=1.02)
    return style.save(fig, path)


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n-seeds", type=int, default=30)
    p.add_argument("--plot-only", action="store_true")
    args = p.parse_args(argv)
    jpath = C.RESULTS_DIR / f"{NAME}.json"
    if args.plot_only:
        results = load_json(jpath)
    else:
        results = compute(n_seeds=args.n_seeds)
        save_json(results, jpath)
    fig = plot(results, C.FIG_DIR / f"{NAME}.png")
    print(f"[{NAME}] wrote {jpath} and {fig}")


if __name__ == "__main__":
    main()
