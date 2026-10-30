"""Experiment 18: following a transmitter that switches algorithm at will.

The operator of the control room presses one hop algorithm after another; each
press re-keys the radio with a fresh secret from the recce ranges. This
experiment replays that at scale: for each sensor quality (including a perfect
sensor) it draws random switch plans over the whole library -- all 13 algorithms,
counter-based PRNGs, chaotic maps, the "new algorithm" families and the keyed
CSPRNG -- with regimes of 50-90 hops, and runs :class:`fhew.tracker.SwitchTracker`
on the sensor stream.

Per switch it records whether and how fast the switch was detected and how well
it was dated; per regime, the identification at the regime's end (exact, a
graceful non-algebraic "closest family" for an algorithm the bank has no expert
for, or wrong), whether a correct lock was reached and how many hops it took, and
any *dangerous* lock -- an algebraic family that is not the truth -- made after
the switch was detected. False alarms are switches declared inside a regime.

Panel A: detection rate and delay per sensor. Panel B: final identification and
locks per sensor.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from experiments import common as C
from experiments import style
from fhew.channels import ChannelSet
from fhew.config import GENERATOR_TARGET_FAMILY
from fhew.generators import make_generator
from fhew.logging_utils import load_json, save_json
from fhew.sensor import SensorModel
from fhew.tracker import ExpertLibrary, SwitchTracker

NAME = "exp18_switch_tracking"

SENSORS = {"perfect": (1.0, 0.0), "strong": (0.95, 0.01), "typical": (0.8, 0.02),
           "contested": (0.6, 0.05)}
ALGORITHMS = ("lfsr", "filtered_lfsr", "lcg", "lcg_full", "weak_seed", "logistic", "tent",
              "henon", "markov", "lagged_fib", "quadratic_cg", "sine_map", "csprng")
ALGEBRAIC = frozenset({"lfsr", "filtered_lfsr", "lcg", "lcg_full", "weak_seed"})
WINDOW = 400
#: each plan is ~400 tracked hops (seconds each), so seeds are capped for run_all
MAX_SEEDS = 8


def switch_plan(rng: np.random.Generator, window: int) -> list[tuple[int, str]]:
    """Random regimes of 50-90 hops, never the same algorithm twice in a row."""
    t, plan, prev = 0, [], None
    while t < window - 30:
        algo = str(rng.choice([a for a in ALGORITHMS if a != prev]))
        plan.append((t, algo))
        prev = algo
        t += int(rng.integers(50, 91))
    return plan


def run_plan(library: ExpertLibrary, sensor: SensorModel, seed: int, window: int = WINDOW):
    ch = library.channels
    rng = np.random.default_rng(seed)
    plan = switch_plan(rng, window)
    tracker = SwitchTracker(ch, sensor, library=library)
    starts, gen, steps = dict(plan), None, []
    t0 = time.time()
    for t in range(window):
        if t in starts:
            algo = starts[t]
            gen = make_generator(algo, ch.size, rng=rng, **C.draw_episode_params(algo, rng))
        obs = sensor.observe([gen.next_channel()], ch, rng, start_slot=t)[0]
        steps.append(tracker.observe(obs))
    return plan, tracker.switches, steps, (time.time() - t0) / window


def score(plan, switches, steps, window: int = WINDOW) -> list[dict]:
    """One row per regime of a run."""
    rows = []
    for i, (s, algo) in enumerate(plan):
        e = plan[i + 1][0] if i + 1 < len(plan) else window
        target = GENERATOR_TARGET_FAMILY[algo]
        dets = [(d, b) for d, b in switches if s <= d < e]
        first = dets[0] if (i > 0 and dets) else None
        # identification after the switch was noticed (from the start for regime 0)
        t_from = s if i == 0 else (first[0] + 1 if first else e)
        lock, danger = None, False
        for t in range(t_from, e):
            idn = steps[t].ident
            lab = "unknown" if idn.label == "H0" else idn.label
            if idn.locked and lab in ALGEBRAIC and lab != target:
                danger = True
            if idn.locked and lab == target and lock is None:
                lock = t - s + 1
        end = steps[e - 1].ident
        lab = "unknown" if end.label == "H0" else end.label
        rows.append({
            "algo": algo, "is_switch": i > 0, "detected": first is not None,
            "delay": first[0] - s if first else None,
            "start_error": first[1] - s if first else None,
            "false_alarms": len(dets) - (1 if first else 0),
            "final": "exact" if lab == target else
                     ("soft" if target == "unknown" and lab not in ALGEBRAIC else "wrong"),
            "lock_hops": lock, "dangerous_lock": danger,
        })
    return rows


def summarise(rows: list[dict], hops: int, ms: float) -> dict:
    sw = [r for r in rows if r["is_switch"]]
    det = [r for r in sw if r["detected"]]
    delays = [r["delay"] for r in det]
    locks = [r["lock_hops"] for r in rows if r["lock_hops"] is not None]
    n = len(rows)
    return {
        "n_switches": len(sw), "n_regimes": n, "ms_per_hop": round(ms, 1),
        "detect_rate": round(len(det) / max(1, len(sw)), 3),
        "delay_median": float(np.median(delays)) if delays else None,
        "delay_p90": float(np.percentile(delays, 90)) if delays else None,
        "dated_within_1": round(float(np.mean([abs(r["start_error"]) <= 1 for r in det])), 3) if det else None,
        "false_alarms_per_100_hops": round(100 * sum(r["false_alarms"] for r in rows) / hops, 3),
        "final_exact": round(sum(r["final"] == "exact" for r in rows) / n, 3),
        "final_soft": round(sum(r["final"] == "soft" for r in rows) / n, 3),
        "final_wrong": round(sum(r["final"] == "wrong" for r in rows) / n, 3),
        "locked_correct": round(len(locks) / n, 3),
        "lock_hops_median": float(np.median(locks)) if locks else None,
        "dangerous_locks_per_regime": round(sum(r["dangerous_lock"] for r in rows) / n, 3),
    }


def compute(*, n_seeds=6, **_) -> dict:
    n = max(1, min(int(n_seeds), MAX_SEEDS))
    ch = ChannelSet.of_size(C.DEFAULT_M)
    library = ExpertLibrary.build(ch, WINDOW + 2)
    out = {}
    for name, (p_d, p_fa) in SENSORS.items():
        sensor = SensorModel(p_d, p_fa)
        rows, ms = [], []
        for seed in C.seeds(n, base=1800):
            plan, switches, steps, t = run_plan(library, sensor, seed)
            rows += score(plan, switches, steps)
            ms.append(t * 1000)
        out[name] = {"p_d": p_d, "p_fa": p_fa, **summarise(rows, n * WINDOW, float(np.mean(ms))),
                     "rows": rows}
    return {"meta": {"M": C.DEFAULT_M, "window": WINDOW, "n_plans": n, "regime_hops": [50, 90]},
            "sensors": out}


def plot(results: dict, path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    S = results["sensors"]
    names = list(S)
    x = np.arange(len(names))
    ticks = [f"{k}\n$P_d$={S[k]['p_d']:g}, $P_{{fa}}$={S[k]['p_fa']:g}" for k in names]
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw={"wspace": 0.42})
    # A: detection rate (bars) + delay median / p90 (points on a twin axis)
    axA.bar(x, [S[k]["detect_rate"] for k in names], color=style.PALETTE["sky"], width=0.55,
            label="switches detected (left axis)")
    for i, k in enumerate(names):
        axA.text(i, S[k]["detect_rate"] + 0.02, f"{S[k]['detect_rate']:.0%}", ha="center", fontsize=9)
    axA.set_ylim(0, 1.12)
    axA.set_ylabel("share of switches detected")
    axA.set_xticks(x, ticks, fontsize=9)
    ax2 = axA.twinx()
    ax2.plot(x, [S[k]["delay_median"] for k in names], "o-", color=style.PALETTE["blue"], label="median delay (right axis)")
    ax2.plot(x, [S[k]["delay_p90"] for k in names], "s--", color=style.PALETTE["vermillion"], label="90th pct delay (right axis)")
    ax2.set_ylabel("hops from switch to detection")
    ax2.set_ylim(0, max(5, max(S[k]["delay_p90"] or 0 for k in names) * 1.12))
    ax2.grid(False)
    h1, l1 = axA.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axA.legend(h1 + h2, l1 + l2, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3)
    axA.set_title("A. Detecting the switch", loc="left")
    # B: identification at the end of each regime (stacked) + dangerous locks
    bottom = np.zeros(len(names))
    for key, color, label in (("final_exact", style.PALETTE["green"], "named exactly"),
                              ("final_soft", style.PALETTE["yellow"], "closest family (no expert for it)"),
                              ("final_wrong", style.PALETTE["vermillion"], "wrong / not yet")):
        vals = np.array([S[k][key] for k in names])
        axB.bar(x, vals, bottom=bottom, color=color, width=0.55, label=label)
        bottom += vals
    for i, k in enumerate(names):
        axB.text(i, 1.03, f"dangerous locks\n{S[k]['dangerous_locks_per_regime']:.0%} of regimes",
                 ha="center", va="bottom", fontsize=8.5)
    axB.set_ylim(0, 1.22)
    axB.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axB.set_xticks(x, ticks, fontsize=9)
    axB.set_ylabel("share of regimes")
    axB.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3)
    axB.set_title("B. Naming the new algorithm by the end of its regime", loc="left")
    fig.suptitle(f"Switch tracking over random switch plans ({results['meta']['n_plans']} plans x "
                 f"{results['meta']['window']} hops per sensor, M={results['meta']['M']})", fontsize=11)
    return style.save(fig, path)


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--n-seeds", type=int, default=6)
    args = p.parse_args(argv)
    jpath = C.RESULTS_DIR / f"{NAME}.json"
    results = load_json(jpath) if args.plot_only else compute(n_seeds=args.n_seeds)
    if not args.plot_only:
        save_json(results, jpath)
    fig = plot(results, C.FIG_DIR / f"{NAME}.png")
    print(f"[{NAME}] wrote {jpath} and {fig}")
    for k, v in results["sensors"].items():
        print(f"  {k:9s} detect {v['detect_rate']:.0%} (median {v['delay_median']} hops, "
              f"p90 {v['delay_p90']}), dated +-1 {v['dated_within_1']}, false alarms/100 hops "
              f"{v['false_alarms_per_100_hops']}, final exact {v['final_exact']:.0%} + soft "
              f"{v['final_soft']:.0%}, dangerous locks {v['dangerous_locks_per_regime']:.0%}")


if __name__ == "__main__":
    main()
