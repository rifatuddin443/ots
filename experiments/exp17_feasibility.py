"""Experiment 16: the feasibility map with the reactive-jamming region (Section 10).

The feasibility map names which strategy is achievable in which regime, so "what
if the enemy comes fast or hops unpredictably?" has a principled answer. Two
panels:

Panel A -- **dwell vs reaction latency, on an unpredictable (keyed) hopper.**
Prediction is hopeless here (kappa pinned at 0), yet a whole region is still
jammable: wherever the dwell exceeds the reaction latency, reactive / follower
jamming works. Above that line only the barrage floor remains. This is the
region Revision 3 adds, and it is why "unpredictable" is not "un-jammable".

Panel B -- **window vs hop rate, the classic map.** With a breakable generator
the plane partitions into exact-recovery (enough clean consecutive hops),
short-horizon prediction, and barrage.

The map is analytic (``fhew.feasibility``), so this experiment is a fast render,
not a simulation; the thresholds it draws match the mechanisms the other
experiments measure.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments import common as C
from experiments import style
from fhew.feasibility import (
    REGION_COLORS,
    REGIONS,
    RegimeParams,
    dwell_latency_grid,
    window_hoprate_grid,
)
from fhew.logging_utils import load_json, save_json

NAME = "exp17_feasibility"


def compute(*, n_seeds=0, **_) -> dict:
    # Analytic map; n_seeds is ignored (kept for the run_all interface).
    dwells = list(range(1, 13))
    latencies = [1, 2, 3, 4, 5, 6]
    windows = [20, 40, 60, 80, 100, 140, 180, 240]
    hop_rates = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]

    # Panel A base: an unpredictable keyed hopper (kappa=0, key_strength>0) so the
    # only feasible non-barrage strategy is reactive follow.
    baseA = RegimeParams(window=120, kappa=0.0, key_strength=1.0)
    gridA = dwell_latency_grid(dwells, latencies, base=baseA)

    # Panel B base: a breakable generator. Exploitable predictability grows with
    # evidence (hops = window x hop-rate), and exact recovery needs more clean
    # data than short-horizon prediction, so the plane splits into the classic
    # three bands (barrage / short-horizon / exact-recovery).
    baseB = RegimeParams(window=100, kappa=0.9, register_length=14, p_d=0.55, dwell=1.0,
                         reaction_latency=1.0, key_strength=0.0)
    gridB = window_hoprate_grid(windows, hop_rates, base=baseB, evidence_ref=200.0)

    return {
        "meta": {"regions": list(REGIONS)},
        "panelA": {"dwells": dwells, "latencies": latencies, "grid": gridA.tolist(),
                   "base": baseA.__dict__},
        "panelB": {"windows": windows, "hop_rates": hop_rates, "grid": gridB.tolist(),
                   "base": baseB.__dict__},
    }


def _heatmap(ax, grid, xlabels, ylabels, xlab, ylab, title):
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import matplotlib.patches as mpatches

    cmap = ListedColormap([REGION_COLORS[r] for r in REGIONS])
    norm = BoundaryNorm(np.arange(-0.5, len(REGIONS) + 0.5), cmap.N)
    g = np.asarray(grid)
    ax.imshow(g, aspect="auto", origin="lower", cmap=cmap, norm=norm)
    ax.set_xticks(range(len(xlabels))); ax.set_xticklabels(xlabels)
    ax.set_yticks(range(len(ylabels))); ax.set_yticklabels(ylabels)
    ax.set_xlabel(xlab); ax.set_ylabel(ylab); ax.set_title(title)
    present = sorted(set(int(v) for row in grid for v in row))
    handles = [mpatches.Patch(color=REGION_COLORS[REGIONS[i]],
                              label=REGIONS[i].replace("_", " ")) for i in present]
    ax.legend(handles=handles, fontsize=8, loc="upper right", framealpha=0.9)


def plot(results: dict, path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.0, 5.0))
    A = results["panelA"]
    _heatmap(axA, A["grid"], A["latencies"], A["dwells"],
             "reaction latency (slots)", "dwell (slots per channel)",
             "A. Unpredictable hopper ($\\kappa{=}0$): reactive region is a floor")
    B = results["panelB"]
    _heatmap(axB, B["grid"], B["hop_rates"], B["windows"],
             "hop rate (hops/slot)", "engagement window $T$ (slots)",
             "B. Breakable generator: recovery / prediction / barrage")
    return style.save(fig, path)


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--n-seeds", type=int, default=0)
    args = p.parse_args(argv)
    jpath = C.RESULTS_DIR / f"{NAME}.json"
    results = load_json(jpath) if args.plot_only else compute()
    if not args.plot_only:
        save_json(results, jpath)
    fig = plot(results, C.FIG_DIR / f"{NAME}.png")
    print(f"[{NAME}] wrote {jpath} and {fig}")
    gA = np.asarray(results["panelA"]["grid"])
    frac = {REGIONS[i]: float((gA == i).mean()) for i in sorted(set(gA.ravel().tolist()))}
    print(f"  panel A region fractions (unpredictable hopper): {frac}")


if __name__ == "__main__":
    main()
