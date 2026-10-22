"""Experiment 2: accuracy and false-identification rate vs P_fa, per class.

False alarms inject phantom hops. A sensor-aware bank should hold its accuracy
and keep the false-ID rate low as P_fa grows, including on out-of-bank emitters
(which H_0 must catch). We sweep P_fa at a fixed P_d and report both metrics per
generator, in-bank and out-of-bank.
"""

from __future__ import annotations

from pathlib import Path

from experiments import common as C
from experiments import style
from fhew.channels import ChannelSet
from fhew.logging_utils import load_json, save_json
from fhew.metrics import summarize
from fhew.sensor import SensorModel
from fhew.sim.engine import run_bank

NAME = "exp2_accuracy_vs_pfa"


def compute(*, M=C.DEFAULT_M, window=C.DEFAULT_WINDOW, n_seeds=30, p_d=0.8,
            p_fa_grid=(0.0, 0.01, 0.02, 0.05, 0.1),
            generators=("lfsr", "weak_seed", "lcg", "csprng", "filtered_lfsr")) -> dict:
    ch = ChannelSet.of_size(M)
    specs = C.default_specs(M)
    tgt = C.default_target_labels(specs)
    out = {"meta": {"M": M, "window": window, "n_seeds": n_seeds, "p_d": p_d,
                    "p_fa_grid": list(p_fa_grid)}, "series": {}}
    for pfa in p_fa_grid:
        sensor = SensorModel(p_d=p_d, p_fa=pfa)
        bank = C.make_bank(ch, window, sensor, specs)
        for gen in generators:
            res = [run_bank(bank, C.episode_for(gen, ch, sensor, window, s,
                                                target_labels=tgt))
                   for s in C.seeds(n_seeds)]
            summ = summarize(res).as_dict()
            out["series"].setdefault(gen, []).append({"p_fa": pfa, **summ})
    return out


def plot(results: dict, path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    fig, (axa, axf) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for i, (gen, rows) in enumerate(results["series"].items()):
        color = style.color_for(gen, style.GENERATOR_COLORS, i)
        pfa = [r["p_fa"] for r in rows]
        acc = [r["accuracy"] for r in rows]
        lo = [r["accuracy_ci"][0] for r in rows]
        hi = [r["accuracy_ci"][1] for r in rows]
        axa.plot(pfa, acc, "-o", color=color, label=gen)
        style.band(axa, pfa, lo, hi, color)
        axf.plot(pfa, [r["false_id_rate"] for r in rows], "-o", color=color, label=gen)
    axa.set_xlabel("false-alarm rate $P_{fa}$"); axa.set_ylabel("accuracy (correct vs target)")
    axa.set_title("Accuracy vs $P_{fa}$"); axa.set_ylim(0, 1.03)
    axf.set_xlabel("false-alarm rate $P_{fa}$"); axf.set_ylabel("false-identification rate")
    axf.set_title("False-ID vs $P_{fa}$"); axf.set_ylim(-0.02, 1.03)
    axf.legend(title="generator", loc="upper left", fontsize=9)
    fig.suptitle(f"$P_d={results['meta']['p_d']}$, {results['meta']['n_seeds']} seeds "
                 f"(csprng/filtered_lfsr are out-of-bank: target = unknown)", y=1.02, fontsize=10)
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


if __name__ == "__main__":
    main()
