"""Experiment 14: regime-switch recovery (thesis Sections 4.2 / 6.4).

The adversary changes its generator family mid-engagement (the "regime-switcher"
red-team adversary). This experiment measures the *recovery time* -- hops after
the switch until the system's top-1 next-hop prediction comes back -- for three
Layer-2/3 configurations:

    * ``bank_alone``       -- the Bayesian hypothesis bank only (weights by
      *cumulative* evidence, so it stays stuck on the old winner);
    * ``mix_weights_only`` -- Fixed-Share mixture whose regime monitor resets the
      mixture weights on an alarm but does **not** flush the context model (the
      merged-architecture behaviour the doc described);
    * ``mix_flush``        -- Fixed-Share mixture whose monitor resets the weights
      **and** forgets the universal predictor's context (Revision 3).

The sources are near-deterministic Markov hoppers of varying determinism
(Dirichlet concentration ``alpha``; smaller = more predictable), switching to a
fresh matrix at the midpoint. The headline: the context flush recovers fastest,
the cumulative bank slowest, and the gap is exactly the cost of leaving the
context model full of the old regime's patterns.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments import common as C
from experiments import style
from fhew.channels import ChannelSet
from fhew.config import build_bank, default_bank_specs
from fhew.logging_utils import load_json, save_json
from fhew.metrics import bootstrap_ci
from fhew.mixture import build_default_mixture
from fhew.regime import RegimeController, make_detector, recovery_time
from fhew.scenarios import make_switch_episode
from fhew.sensor import SensorModel

NAME = "exp15_regime_switch"

_DETECTOR = {"kind": "cusum", "k": 0.5, "h": 5.5, "warmup": 18, "ewma": 0.03}


def _switch_episode(ch, sensor, window, switch_at, alpha, seed):
    rng = np.random.default_rng(seed)
    return make_switch_episode(
        [("markov", {"alpha": alpha}), ("markov", {"alpha": alpha})],
        ch, sensor, window, rng, switch_at=switch_at)


def _top1_series(predict_each, ep, window):
    """Run a per-slot callback that consumes an obs and returns the current
    top-1 next-hop index; collect whether it matched the true next channel."""
    out = []
    for t, o in enumerate(ep.observations):
        top1 = predict_each(o)
        if t < window - 1:
            out.append(int(top1) == int(ep.true_channels[t + 1]))
    return out


def _run_mixture(ep, ch, sensor, window, switch_at, *, forget_context):
    mix = build_default_mixture(ch, window, sensor)
    ctrl = RegimeController(mix, make_detector(**_DETECTOR),
                            forget_context=forget_context, retain=0.0,
                            reset_weights=True, cooldown=8)
    def step(o):
        ctrl.update(o)
        return int(np.argmax(mix.predict()))
    t1 = _top1_series(step, ep, window)
    post = [a for a in ctrl.alarms if a >= switch_at]
    delay = (post[0] - switch_at) if post else None
    fp = len([a for a in ctrl.alarms if a < switch_at])
    return recovery_time(t1, switch_at, window=6, thresh=0.6), delay, fp


def _run_bank(ep, ch, sensor, window, switch_at):
    bank = build_bank(ch, window, sensor, default_bank_specs(ch.size))
    def step(o):
        bank.update(o)
        return int(np.argmax(bank.predict()))
    t1 = _top1_series(step, ep, window)
    return recovery_time(t1, switch_at, window=6, thresh=0.6)


def compute(*, M=C.DEFAULT_M, window=240, switch_at=120, n_seeds=30,
            p_d=0.97, p_fa=0.01, alpha_grid=(0.02, 0.04, 0.06, 0.08)) -> dict:
    ch = ChannelSet.of_size(M)
    sensor = SensorModel(p_d=p_d, p_fa=p_fa)
    out = {"meta": {"M": M, "window": window, "switch_at": switch_at, "n_seeds": n_seeds,
                    "p_d": p_d, "p_fa": p_fa, "detector": _DETECTOR},
           "alpha_grid": list(alpha_grid), "series": {}, "detection": {}}
    methods = ("mix_flush", "mix_weights_only", "bank_alone")
    for m in methods:
        out["series"][m] = []
    delays, fps, fired = [], [], 0
    for alpha in alpha_grid:
        rec = {m: [] for m in methods}
        for s in C.seeds(n_seeds):
            ep = _switch_episode(ch, sensor, window, switch_at, alpha, s)
            r_on, d, fp = _run_mixture(ep, ch, sensor, window, switch_at, forget_context=True)
            r_off, _, _ = _run_mixture(ep, ch, sensor, window, switch_at, forget_context=False)
            r_bk = _run_bank(ep, ch, sensor, window, switch_at)
            rec["mix_flush"].append(r_on)
            rec["mix_weights_only"].append(r_off)
            rec["bank_alone"].append(r_bk)
            if d is not None:
                delays.append(d); fired += 1
            fps.append(fp)
        for m in methods:
            arr = np.asarray(rec[m], dtype=np.float64)
            finite = arr[np.isfinite(arr)]
            out["series"][m].append({
                "alpha": alpha,
                "mean_recovery": float(finite.mean()) if finite.size else float("nan"),
                "ci": bootstrap_ci(finite) if finite.size else (float("nan"),) * 2,
                "recovered_frac": float(np.isfinite(arr).mean()),
            })
    n_tot = len(alpha_grid) * n_seeds
    out["detection"] = {
        "mean_post_switch_delay": float(np.mean(delays)) if delays else float("nan"),
        "fired_frac": fired / n_tot,
        "pre_switch_false_alarms_per_episode": float(np.mean(fps)) if fps else 0.0,
    }
    return out


def plot(results: dict, path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    alphas = results["alpha_grid"]
    labels = {"mix_flush": "mixture + context flush (Rev 3)",
              "mix_weights_only": "mixture, weights-only reset",
              "bank_alone": "Bayesian bank alone (cumulative)"}
    colors = {"mix_flush": style.PALETTE["green"],
              "mix_weights_only": style.PALETTE["orange"],
              "bank_alone": style.PALETTE["vermillion"]}
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for m, lab in labels.items():
        rows = results["series"][m]
        y = [r["mean_recovery"] for r in rows]
        lo = [r["ci"][0] for r in rows]
        hi = [r["ci"][1] for r in rows]
        ax.plot(alphas, y, "-o", color=colors[m], label=lab)
        style.band(ax, alphas, lo, hi, colors[m])
    ax.set_xlabel(r"source determinism (Markov $\alpha$; smaller = more predictable)")
    ax.set_ylabel("recovery time after switch (hops, lower is better)")
    det = results["detection"]
    ax.set_title("Regime-switch recovery: context flush re-points the mixture faster\n"
                 f"(monitor fired on {det['fired_frac']*100:.0f}% of switches, "
                 f"mean delay {det['mean_post_switch_delay']:.1f} hops, "
                 f"{det['pre_switch_false_alarms_per_episode']:.2f} false alarms/episode)")
    ax.legend()
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
    for m, rows in results["series"].items():
        txt = "  ".join(f"a={r['alpha']}:{r['mean_recovery']:.1f}" for r in rows)
        print(f"  {m:20s} {txt}")
    print(f"  detection: {results['detection']}")


if __name__ == "__main__":
    main()
