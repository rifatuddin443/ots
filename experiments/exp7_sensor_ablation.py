"""Experiment 7: the sensor-awareness ablation.

The same bank with the naive likelihood (ignores P_d/P_fa, scores clean matches)
versus the sensor-aware likelihood (the HMM forward pass). The difference,
measured as hops-to-identify and accuracy across P_d at a stressing false-alarm
rate, is the ablation that justifies the whole approach (paper Section 11,
item 7). It is one switch in the bank, not a second codebase.
"""

from __future__ import annotations

from pathlib import Path

from experiments import common as C
from experiments import style
from fhew.channels import ChannelSet
from fhew.logging_utils import load_json, save_json
from fhew.metrics import summarize
from fhew.sensor import SensorModel
from fhew.sim import run_bank

NAME = "exp7_sensor_ablation"


def compute(*, M=C.DEFAULT_M, window=C.DEFAULT_WINDOW, n_seeds=30,
            p_d_grid=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5), p_fa=0.05,
            generators=("lfsr", "weak_seed", "lcg")) -> dict:
    ch = ChannelSet.of_size(M)
    specs = C.default_specs(M)
    tgt = C.default_target_labels(specs)
    out = {"meta": {"M": M, "window": window, "n_seeds": n_seeds, "p_fa": p_fa,
                    "p_d_grid": list(p_d_grid), "generators": list(generators)},
           "sensor_aware": [], "naive": []}
    for emission in ("sensor_aware", "naive"):
        for pd in p_d_grid:
            sensor = SensorModel(p_d=pd, p_fa=p_fa)
            bank = C.make_bank(ch, window, sensor, specs, emission_kind=emission)
            res = []
            for gen in generators:
                res += [run_bank(bank, C.episode_for(gen, ch, sensor, window, s,
                                                     target_labels=tgt))
                        for s in C.seeds(n_seeds)]
            summ = summarize(res).as_dict()
            out[emission].append({"p_d": pd, **summ})
    return out


def plot(results: dict, path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    fig, (axh, axa) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for emission, color, mk in [("sensor_aware", style.PALETTE["blue"], "-o"),
                                ("naive", style.PALETTE["vermillion"], "--s")]:
        rows = results[emission]
        pd = [r["p_d"] for r in rows]
        axh.plot(pd, [r["mean_hops_to_identify"] for r in rows], mk, color=color,
                 label=emission)
        axa.plot(pd, [r["accuracy"] for r in rows], mk, color=color, label=emission)
    axh.set_xlabel("detection probability $P_d$"); axh.invert_xaxis()
    axh.set_ylabel("mean hops-to-identify"); axh.set_title("Hops-to-identify")
    axh.legend()
    axa.set_xlabel("detection probability $P_d$"); axa.invert_xaxis()
    axa.set_ylabel("accuracy"); axa.set_ylim(0, 1.03); axa.set_title("Accuracy")
    axa.legend()
    fig.suptitle(f"Sensor-aware vs naive likelihood ($P_{{fa}}={results['meta']['p_fa']}$, "
                 f"{results['meta']['n_seeds']} seeds/gen)", y=1.02)
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
    for sa, nv in zip(results["sensor_aware"], results["naive"]):
        print(f"  Pd={sa['p_d']}: sensor-aware acc={sa['accuracy']:.2f} hops={sa['mean_hops_to_identify']:.1f}"
              f" | naive acc={nv['accuracy']:.2f} hops={nv['mean_hops_to_identify']:.1f}")


if __name__ == "__main__":
    main()
