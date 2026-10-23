"""Experiment 15: multi-frequency power allocation (thesis Sections 6.3 / 7).

A single next-channel prediction is not enough to act on, so the action is a
power *vector* over channels (a multi-tone jam). This experiment shows the two
levers that vector gives, measured as power-on-target reward (the realised
jamming effect ``g`` on the channel the adversary actually used, Section 7.5):

Panel A -- **spot vs barrage falls out of the predictive + knapsack.** Across
adversaries of increasing predictability, the greedy threshold-aware knapsack
tracks the best allocation: it concentrates (spot) when the prediction is sharp
and spreads (barrage) when it is flat, beating plain / weighted water-filling
and equal power. No operator-selected mode.

Panel B -- **reactive / follower jamming is the second weapon.** Against an
*unpredictable* keyed hopper (CSPRNG) the predictive allocators collapse to the
barrage floor, but detect-and-follow jams it whenever the dwell exceeds the
reaction latency -- so reward rises with dwell independently of kappa. The
entropy-maximising hopper at dwell 1 defeats both: that corner is the thesis'
true predictability boundary.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments import common as C
from experiments import style
from fhew.allocator import AllocationController, CDTController, EffectModel
from fhew.channels import ChannelSet
from fhew.dwell import DwellTracker
from fhew.logging_utils import load_json, save_json
from fhew.metrics import bootstrap_ci
from fhew.mixture import build_default_mixture
from fhew.scenarios import make_dwell_episode
from fhew.sensor import SensorModel

NAME = "exp16_allocation"

# Adversaries ordered from unpredictable (left) to predictable (right); each is
# labelled by a nominal predictability for the x-axis of panel A.
_PREDICTABILITY_SOURCES = [
    ("csprng", {}, 0.0),
    ("markov", {"alpha": 0.8}, 0.2),
    ("markov", {"alpha": 0.2}, 0.5),
    ("weak_seed", {"keyspace": 256}, 0.8),
    ("lfsr", {"length": 7}, 1.0),
]
_ALLOCATORS = ("greedy_knapsack", "weighted_water_filling", "water_filling", "equal")


def _run_allocation(gen, gen_params, allocator, ch, sensor, window, budget, effect,
                    reaction_latency, dwell, seed, *, enable_reactive=True):
    rng = np.random.default_rng(seed)
    ep = make_dwell_episode(gen, ch, sensor, window, rng, dwell=dwell,
                            gen_params=dict(gen_params))
    mix = build_default_mixture(ch, window, sensor)
    tracker = DwellTracker()
    ctrl = AllocationController(effect, budget, reaction_latency=reaction_latency,
                               cdt=CDTController(alpha=0.2, eta=0.05), allocator=allocator)
    rewards = []
    for t, o in enumerate(ep.observations):
        mix.update(o)
        tracker.update(o)
        if t >= window - 1:
            continue
        obs_ch = (o.detections[0].channel if len(o.detections) == 1
                  else tracker.current_channel)
        dwell_est = tracker.estimate() if enable_reactive else 0.0
        step = ctrl.step(mix.predict(), int(ep.true_channels[t + 1]),
                         observed_channel=obs_ch, dwell_estimate=dwell_est,
                         current_run=tracker.current_run)
        rewards.append(step.effect_on_true)
    return float(np.mean(rewards))


def compute(*, M=C.DEFAULT_M, window=160, n_seeds=30, p_d=0.95, p_fa=0.01,
            budget=3.0, tau=1.0, reaction_latency=1,
            dwell_grid=(1, 2, 3, 5, 8)) -> dict:
    ch = ChannelSet.of_size(M)
    sensor = SensorModel(p_d=p_d, p_fa=p_fa)
    effect = EffectModel(ch, tau=tau, sharpness=10.0)
    out = {"meta": {"M": M, "window": window, "n_seeds": n_seeds, "p_d": p_d, "p_fa": p_fa,
                    "budget": budget, "tau": tau, "reaction_latency": reaction_latency,
                    "channels_denyable": int(budget / effect.costs().mean())},
           "panelA": {"predictability": [], "allocators": {a: [] for a in _ALLOCATORS}},
           "panelB": {"dwell_grid": list(dwell_grid), "series": {}}}

    # Panel A: spot<->barrage across predictability (no dwell, no reactive mode).
    for gen, params, kappa_nom in _PREDICTABILITY_SOURCES:
        out["panelA"]["predictability"].append(kappa_nom)
        for alloc in _ALLOCATORS:
            vals = [_run_allocation(gen, params, alloc, ch, sensor, window, budget,
                                    effect, reaction_latency, dwell=1, seed=s,
                                    enable_reactive=False)
                    for s in C.seeds(n_seeds)]
            arr = np.asarray(vals)
            out["panelA"]["allocators"][alloc].append(
                {"mean": float(arr.mean()), "ci": bootstrap_ci(arr)})

    # Panel B: reactive follower vs dwell, on an unpredictable keyed hopper and on
    # the entropy-maximising hopper (both uncrackable by prediction).
    for gen, params in (("csprng", {}), ("entropy_max", {})):
        key = gen
        out["panelB"]["series"][key] = {"knapsack_reactive": [], "knapsack_predict_only": []}
        for dwell in dwell_grid:
            rv = [_run_allocation(gen, params, "greedy_knapsack", ch, sensor, window,
                                  budget, effect, reaction_latency, dwell=dwell, seed=s,
                                  enable_reactive=True) for s in C.seeds(n_seeds)]
            pv = [_run_allocation(gen, params, "greedy_knapsack", ch, sensor, window,
                                  budget, effect, reaction_latency, dwell=dwell, seed=s,
                                  enable_reactive=False) for s in C.seeds(n_seeds)]
            out["panelB"]["series"][key]["knapsack_reactive"].append(
                {"dwell": dwell, "mean": float(np.mean(rv)), "ci": bootstrap_ci(np.asarray(rv))})
            out["panelB"]["series"][key]["knapsack_predict_only"].append(
                {"dwell": dwell, "mean": float(np.mean(pv)), "ci": bootstrap_ci(np.asarray(pv))})
    return out


def plot(results: dict, path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.6, 4.8))

    # Panel A
    x = results["panelA"]["predictability"]
    alloc_style = {
        "greedy_knapsack": ("greedy knapsack (main)", style.PALETTE["blue"], "-o"),
        "weighted_water_filling": ("weighted water-filling", style.PALETTE["orange"], "--s"),
        "water_filling": ("water-filling", style.PALETTE["green"], "--^"),
        "equal": ("equal power", style.PALETTE["grey"], ":d"),
    }
    for alloc, (lab, col, ls) in alloc_style.items():
        rows = results["panelA"]["allocators"][alloc]
        y = [r["mean"] for r in rows]
        lo = [r["ci"][0] for r in rows]; hi = [r["ci"][1] for r in rows]
        axA.plot(x, y, ls, color=col, label=lab)
        style.band(axA, x, lo, hi, col)
    axA.set_xlabel("adversary predictability (nominal)")
    axA.set_ylabel("power-on-target reward (mean $g$)")
    axA.set_title("A. Spot↔barrage emerges from predictive + knapsack")
    axA.legend(fontsize=9)

    # Panel B
    dwell = results["panelB"]["dwell_grid"]
    colors = {"csprng": style.PALETTE["vermillion"], "entropy_max": style.PALETTE["purple"]}
    for gen, series in results["panelB"]["series"].items():
        col = colors.get(gen, style.PALETTE["black"])
        yr = [r["mean"] for r in series["knapsack_reactive"]]
        yp = [r["mean"] for r in series["knapsack_predict_only"]]
        axB.plot(dwell, yr, "-o", color=col, label=f"{gen}: +reactive follow")
        axB.plot(dwell, yp, "--x", color=col, alpha=0.7, label=f"{gen}: predict only")
    axB.axvline(results["meta"]["reaction_latency"], color=style.PALETTE["grey"], ls=":",
                label="reaction latency")
    axB.set_xlabel("dwell (slots per channel)")
    axB.set_ylabel("power-on-target reward (mean $g$)")
    axB.set_title("B. Reactive jamming beats an unpredictable but slow hopper")
    axB.legend(fontsize=9)
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
    km = results["panelA"]["allocators"]["greedy_knapsack"]
    wf = results["panelA"]["allocators"]["water_filling"]
    print("  panel A (knapsack vs water-filling reward):")
    for kv, wv, kap in zip(km, wf, results["panelA"]["predictability"]):
        print(f"    kappa~{kap}: knapsack={kv['mean']:.2f}  water-fill={wv['mean']:.2f}")
    print("  panel B (reactive vs predict-only on csprng):")
    s = results["panelB"]["series"]["csprng"]
    for rr, pp in zip(s["knapsack_reactive"], s["knapsack_predict_only"]):
        print(f"    dwell={rr['dwell']}: reactive={rr['mean']:.2f}  predict-only={pp['mean']:.2f}")


if __name__ == "__main__":
    main()
