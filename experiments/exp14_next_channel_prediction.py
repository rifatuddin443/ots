"""Experiment 14: next-channel prediction — the headline achievement metric.

This is the experiment that scores the system on *predicting the next channel*,
not on identifying the generator. Its primary number is **top-1 next-channel
accuracy**: the fraction of slots where the single most probable channel is the
one the emitter actually hops to.

It makes two points the identification experiments cannot:

1. The Fixed-Share predictor (`proposed`, carrying the zero-shot universal
   expert) predicts the next channel for a **new algorithm the bank has no
   expert for** — the additive `lagged_fib`, the nonlinear `quadratic_cg`, the
   unseen chaotic `sine_map` — far above chance and far above an identification-
   only bank, which collapses toward uniform when no hypothesis matches.
2. On known generators it loses nothing (both reach ≈1.0), and on a keyed CSPRNG
   both correctly fall to chance — the predictability boundary, reported.

Methods compared, all on the same lossy stream and the same sensor-aware
likelihoods: `proposed` (Fixed-Share + universal), `id_only` (Fixed-Share over
the identification bank, no universal fallback), online first-order `markov`,
and the `uniform` floor (`1/M`).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from experiments import common as C
from experiments import style
from fhew.channels import ChannelSet
from fhew.config import (
    NEW_ALGORITHM_GENERATORS,
    identification_bank_specs,
    prediction_expert_specs,
)
from fhew.logging_utils import load_json, save_json
from fhew.predict import ChannelPredictor, predict_over_episode
from fhew.sensor import SensorModel

NAME = "exp14_next_channel_prediction"

# Enumeration ranges kept modest so the two full predictors run quickly; the
# episode secrets are drawn inside the same ranges (information parity).
LCG_SEED_MAX = 256
WEAK_KEYSPACE = 256

IN_BANK = ("lfsr", "weak_seed", "lcg", "logistic", "markov")
CONTROL = ("csprng",)
GENERATORS = IN_BANK + NEW_ALGORITHM_GENERATORS + CONTROL


def _pred_specs(M):
    return prediction_expert_specs(M, lcg_seed_max=LCG_SEED_MAX, weak_keyspace=WEAK_KEYSPACE)


def _id_specs(M):
    return identification_bank_specs(M, lcg_seed_max=LCG_SEED_MAX, weak_keyspace=WEAK_KEYSPACE)


def _markov_top1(true, observations, M):
    """Online first-order Markov next-channel top-1 (fit on the clean stream)."""
    counts = np.ones((M, M))
    last = None
    hits = []
    for t, o in enumerate(observations):
        if t >= len(true) - 1:
            break
        p = counts[last] / counts[last].sum() if last is not None else np.full(M, 1.0 / M)
        hits.append(int(int(np.argmax(p)) == int(true[t + 1])))
        if len(o.detections) == 1:
            c = o.detections[0].channel
            if last is not None:
                counts[last, c] += 1.0
            last = c
    return float(np.mean(hits)) if hits else float("nan")


def _episode(gen, ch, sensor, window, seed, tgt):
    inb = gen in IN_BANK
    params = {"keyspace": WEAK_KEYSPACE} if gen == "weak_seed" else None
    return C.episode_for(gen, ch, sensor, window, seed, gen_params=params,
                         target_labels=tgt if inb else None, lcg_keyspace=LCG_SEED_MAX)


def compute(*, M=C.DEFAULT_M, window=120, n_seeds=12, p_d=0.95, p_fa=0.0,
            pd_sweep=(1.0, 0.95, 0.9, 0.8, 0.7), sweep_seeds=10) -> dict:
    ch = ChannelSet.of_size(M)
    chance = 1.0 / M

    # --- operating-point comparison (the main per-generator table) ---
    sensor = SensorModel(p_d=p_d, p_fa=p_fa)
    proposed = ChannelPredictor(ch, sensor, specs=_pred_specs(M))
    id_only = ChannelPredictor(ch, sensor, specs=_id_specs(M))
    tgt = C.default_target_labels(_id_specs(M))

    per_gen = {}
    for gen in GENERATORS:
        acc = {k: [] for k in ("proposed_top1", "proposed_steady", "proposed_top3",
                               "proposed_logloss", "id_only_steady", "markov_top1")}
        for s in C.seeds(n_seeds):
            ep = _episode(gen, ch, sensor, window, s, tgt)
            pr = predict_over_episode(proposed, ep.true_channels, ep.observations)
            idr = predict_over_episode(id_only, ep.true_channels, ep.observations)
            acc["proposed_top1"].append(pr.top1)
            acc["proposed_steady"].append(pr.top1_steady)
            acc["proposed_top3"].append(pr.top3)
            acc["proposed_logloss"].append(pr.logloss)
            acc["id_only_steady"].append(idr.top1_steady)
            acc["markov_top1"].append(_markov_top1(ep.true_channels, ep.observations, M))
        per_gen[gen] = {k: float(np.nanmean(v)) for k, v in acc.items()}
        per_gen[gen]["uniform_top1"] = chance
        per_gen[gen]["group"] = ("in_bank" if gen in IN_BANK
                                 else "new_algorithm" if gen in NEW_ALGORITHM_GENERATORS
                                 else "control")

    # --- detection-probability robustness sweep for the new-algorithm group ---
    sweep = {"p_d": list(pd_sweep), "proposed": {}, "id_only": {}}
    for gen in NEW_ALGORITHM_GENERATORS:
        prop_curve, id_curve = [], []
        for pd in pd_sweep:
            sens = SensorModel(p_d=pd, p_fa=p_fa)
            pp = ChannelPredictor(ch, sens, specs=_pred_specs(M))
            ii = ChannelPredictor(ch, sens, specs=_id_specs(M))
            pr, idr = [], []
            for s in C.seeds(sweep_seeds):
                ep = _episode(gen, ch, sens, window, s, tgt)
                pr.append(predict_over_episode(pp, ep.true_channels, ep.observations).top1_steady)
                idr.append(predict_over_episode(ii, ep.true_channels, ep.observations).top1_steady)
            prop_curve.append(float(np.mean(pr)))
            id_curve.append(float(np.mean(idr)))
        sweep["proposed"][gen] = prop_curve
        sweep["id_only"][gen] = id_curve

    return {
        "meta": {"M": M, "window": window, "n_seeds": n_seeds, "p_d": p_d, "p_fa": p_fa,
                 "chance": chance, "metric": "next-channel top-1 accuracy"},
        "operating_point": per_gen,
        "pd_sweep": sweep,
    }


def plot(results: dict, path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    per = results["operating_point"]
    gens = [g for g in GENERATORS if g in per]
    x = np.arange(len(gens))
    chance = results["meta"]["chance"]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 4.8),
                                   gridspec_kw={"width_ratios": [1.6, 1.0]})

    # Panel A: steady-state top-1 by generator, four methods.
    w = 0.21
    proposed = [per[g]["proposed_steady"] for g in gens]
    idonly = [per[g]["id_only_steady"] for g in gens]
    markov = [per[g]["markov_top1"] for g in gens]
    axA.bar(x - 1.5 * w, proposed, w, label="proposed (+ universal)",
            color=style.PALETTE["blue"])
    axA.bar(x - 0.5 * w, idonly, w, label="bank only", color=style.PALETTE["orange"])
    axA.bar(x + 0.5 * w, markov, w, label="online Markov", color=style.PALETTE["green"])
    axA.bar(x + 1.5 * w, [chance] * len(gens), w, label=f"uniform ({chance:.3f})",
            color=style.PALETTE["grey"])
    axA.axhline(chance, color=style.PALETTE["grey"], ls=":", lw=1)
    # shade the new-algorithm block
    new_idx = [i for i, g in enumerate(gens) if g in NEW_ALGORITHM_GENERATORS]
    if new_idx:
        axA.axvspan(min(new_idx) - 0.5, max(new_idx) + 0.5, color=style.PALETTE["yellow"],
                    alpha=0.12, zorder=0)
        axA.text(np.mean(new_idx), 1.02, "new algorithms\n(no dedicated expert)",
                 ha="center", va="bottom", fontsize=9, color="#555")
    axA.set_xticks(x)
    axA.set_xticklabels(gens, rotation=30, ha="right")
    axA.set_ylabel("next-channel top-1 accuracy (steady state)")
    axA.set_ylim(0, 1.18)
    axA.set_title(f"Next-channel prediction  ($P_d={results['meta']['p_d']}$, "
                  f"$P_{{fa}}={results['meta']['p_fa']}$)")
    axA.legend(loc="upper right", fontsize=8, ncol=1, frameon=True,
               facecolor="white", framealpha=0.9)

    # Panel B: detection-probability robustness for new algorithms.
    sw = results["pd_sweep"]
    pds = sw["p_d"]
    for i, gen in enumerate(NEW_ALGORITHM_GENERATORS):
        col = style.SERIES_COLORS[i % len(style.SERIES_COLORS)]
        axB.plot(pds, sw["proposed"][gen], "-o", color=col, label=f"{gen} (proposed)")
        axB.plot(pds, sw["id_only"][gen], "--x", color=col, alpha=0.7,
                 label=f"{gen} (bank only)")
    axB.axhline(chance, color=style.PALETTE["grey"], ls=":", lw=1, label=f"chance {chance:.3f}")
    axB.set_xlabel("detection probability $P_d$")
    axB.set_ylabel("steady-state top-1")
    axB.set_title("New-algorithm prediction vs sensor loss")
    axB.set_ylim(0, 1.0)
    axB.legend(fontsize=7.5)

    fig.tight_layout()
    return style.save(fig, path)


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n-seeds", type=int, default=12)
    p.add_argument("--plot-only", action="store_true")
    args = p.parse_args(argv)
    jpath = C.RESULTS_DIR / f"{NAME}.json"
    results = load_json(jpath) if args.plot_only else compute(n_seeds=args.n_seeds)
    if not args.plot_only:
        save_json(results, jpath)
    fig = plot(results, C.FIG_DIR / f"{NAME}.png")
    print(f"[{NAME}] wrote {jpath} and {fig}")
    per = results["operating_point"]
    print(f"  {'generator':14s} {'proposed':>9s} {'id_only':>8s} {'markov':>7s}  "
          f"(chance={results['meta']['chance']:.3f})")
    for g, d in per.items():
        tag = "  <- NEW" if d["group"] == "new_algorithm" else ""
        print(f"  {g:14s} {d['proposed_steady']:9.3f} {d['id_only_steady']:8.3f} "
              f"{d['markov_top1']:7.3f}{tag}")


if __name__ == "__main__":
    main()
