"""Experiment 24: the multi-tone response -- fraction of hops denied vs tone budget.

Experiment 20 shows the conformal next-channel set is calibrated; this one
measures the end quantity the paper claims it transfers to. The conformal set
*is* the multi-tone plan: a tone is funded on every channel of the set, and a
hop is **denied** when its true channel is funded.

Protocol (Experiment 20's, unchanged): the proposed predictor wrapped in
:class:`fhew.conformal.ConformalChannelPredictor` (ACI, target ``0.9``), with
the erasure feedback a passive node really has -- the set is recalibrated only
when the next slot is clean (exactly one detection). A transmitter with a
budget of ``B`` tones funds the ``min(|S_t|, B)`` most probable channels of the
set.

Reported per generator and sensor:

* fraction of hops denied versus the tone budget ``B``, scored against the
  hidden true channel on *every* slot, not only the clean ones;
* the tones actually transmitted (mean set size) with an unlimited budget;
* the same for a fixed top-``k`` shortlist, which spends ``k`` tones on every
  hop whatever the emitter.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments import common as C
from experiments import exp20_conformal_sets as E20
from experiments import style
from fhew.channels import ChannelSet
from fhew.conformal import ConformalChannelPredictor
from fhew.logging_utils import load_json, save_json
from fhew.predict import ChannelPredictor
from fhew.sensor import SensorModel

NAME = "exp24_denial"

SENSORS = {"strong": (0.95, 0.01), "typical": (0.8, 0.02), "contested": (0.6, 0.05)}
GENERATORS = ("lfsr", "lcg", "markov", "lagged_fib", "sine_map", "csprng")
WARMUP = 5


def run_episode(cpred: ConformalChannelPredictor, ep):
    """One pass; per scored slot, the rank of the true next channel in the
    forecast (0 = top call) and the conformal set size chosen for that slot."""
    cpred.reset()
    true = np.asarray(ep.true_channels, dtype=np.int64)
    obs = ep.observations
    ranks, sizes = [], []
    for t in range(len(obs) - 1):
        cpred.observe(obs[t])
        S = cpred.prediction_set()                      # most probable first
        nxt = int(true[t + 1])
        if t >= WARMUP:
            order = np.argsort(cpred.predict_next())[::-1]
            ranks.append(int(np.nonzero(order == nxt)[0][0]))
            sizes.append(len(S))
        if len(obs[t + 1].detections) == 1:             # erasure feedback only
            cpred.calibrate(nxt in S)
    return np.asarray(ranks), np.asarray(sizes)


def compute(*, M=C.DEFAULT_M, window=140, n_seeds=30, alpha=0.1, gamma=0.03,
            sensors=SENSORS, **_) -> dict:
    ch = ChannelSet.of_size(M)
    budgets = list(range(1, M + 1))
    out = {"meta": {"M": M, "window": window, "n_seeds": n_seeds, "budgets": budgets,
                    "target": 1.0 - alpha, "aci_gamma": gamma, "warmup": WARMUP,
                    "generators": list(GENERATORS),
                    "n_episodes": n_seeds * len(GENERATORS) * len(sensors)},
           "sensors": {}}
    for name, (p_d, p_fa) in sensors.items():
        sensor = SensorModel(p_d=p_d, p_fa=p_fa)
        pred = ChannelPredictor(ch, sensor, specs=E20._pred_specs(M))
        cpred = ConformalChannelPredictor(pred, alpha=alpha, gamma=gamma)
        per_gen = {}
        for gen in GENERATORS:
            runs = [run_episode(cpred, E20._episode(gen, ch, sensor, window, s))
                    for s in C.seeds(n_seeds)]
            ranks = np.concatenate([r for r, _ in runs])
            sizes = np.concatenate([z for _, z in runs])
            seed_den = [float((r < z).mean()) for r, z in runs]
            per_gen[gen] = {
                "denied_vs_budget": [float((ranks < np.minimum(sizes, b)).mean())
                                     for b in budgets],
                "tones_vs_budget": [float(np.minimum(sizes, b).mean()) for b in budgets],
                "fixed_topk_denied": [float((ranks < b).mean()) for b in budgets],
                "denied_unlimited": float(np.mean(seed_den)),
                "denied_unlimited_sd": float(np.std(seed_den)),
                "mean_tones": float(sizes.mean()),
            }
        out["sensors"][name] = {"p_d": p_d, "p_fa": p_fa, "generators": per_gen}
    return out


GEN_LABELS = {"lfsr": "LFSR", "lcg": "LCG", "markov": "Markov",
              "lagged_fib": "lagged-Fib. (held out)", "sine_map": "sine map (held out)",
              "csprng": "CSPRNG"}


def plot(results: dict, path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    names = list(results["sensors"])
    B = results["meta"]["budgets"]
    target = results["meta"]["target"]
    fig, axes = plt.subplots(1, len(names), figsize=(5.4 * len(names), 4.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        S = results["sensors"][name]
        for i, g in enumerate(results["meta"]["generators"]):
            d = S["generators"][g]
            ax.plot(B, d["denied_vs_budget"], "-o", ms=3.5,
                    color=style.SERIES_COLORS[i % len(style.SERIES_COLORS)],
                    label=f"{GEN_LABELS[g]} ({d['mean_tones']:.1f} tones)")
        ax.axhline(target, color="#555", ls=":", lw=1)
        ax.set_xlabel("tone budget $B$ (channels funded per hop, at most)")
        ax.set_xticks([1, 2, 4, 6, 8, 10, 12, 14, 16])
        ax.set_ylim(0, 1.04)
        ax.set_title(f"{name} ($P_d$={S['p_d']:g}, $P_{{fa}}$={S['p_fa']:g})")
        ax.legend(fontsize=7.5, loc="lower right")
    axes[0].set_ylabel("fraction of hops denied")
    fig.tight_layout()
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
    print(f"[{NAME}] wrote {jpath} and {fig}  ({results['meta']['n_episodes']} episodes)")
    for name, S in results["sensors"].items():
        print(f"  {name} (target {results['meta']['target']:g}):")
        for g, d in S["generators"].items():
            b = d["denied_vs_budget"]
            print(f"    {g:11s} denied={d['denied_unlimited']:.3f} tones={d['mean_tones']:5.2f}  "
                  f"B=1:{b[0]:.2f} B=2:{b[1]:.2f} B=4:{b[3]:.2f} B=8:{b[7]:.2f}  "
                  f"fixed top-3:{d['fixed_topk_denied'][2]:.2f}")


if __name__ == "__main__":
    main()
