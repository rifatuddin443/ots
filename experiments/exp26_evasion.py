"""Experiment 26: closed-loop jammer evasion -- the reactive adversary.

Every other experiment scores the OTS against a fixed hop tape. This one closes the
loop (:mod:`fhew.duel`): the emitter senses the jamming it receives and adapts --
deflecting off jammed channels, spreading its hops to kill the predictor's
``kappa``, and re-keying under sustained denial (:mod:`fhew.evasion`). We measure
how much a reactive transmitter can claw back from a jammer that was, on a fixed
tape, denying it completely.

Two panels:

* **A -- evasion vs base family.** Fraction of hops denied, evasion OFF vs ON, for a
  predictable in-bank source (LFSR), a predictable held-out source (lagged-Fibonacci
  the bank has no expert for), a stochastic hopper (Markov) and an uncrackable
  CSPRNG. The gap is the evasion dividend; the CSPRNG is the control -- already at
  the barrage floor, it has nothing to evade.
* **B -- the game curve.** Denial vs evasion aggressiveness for the predictable
  sources: as the radio reacts harder it trades stealth for escape, driving denial
  down toward the floor and the OTS predictability ``kappa`` with it.

Reported per condition: fraction denied, mean jamming effect, the OTS's next-hop
top-1 and ``kappa``, and the number of re-keys the adversary was forced into. Both
sides move from history only -- the emitter never sees the jam plan it is about to
face (see :func:`fhew.duel.run_duel`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments import common as C
from experiments import style
from fhew.allocator import EffectModel
from fhew.channels import ChannelSet
from fhew.duel import OTSJammer, run_duel
from fhew.evasion import NO_EVASION_CFG, AdaptiveEvasionGenerator, EvasionConfig
from fhew.logging_utils import load_json, save_json
from fhew.sensor import SensorModel

NAME = "exp26_evasion"

SENSORS = {"strong": (0.95, 0.01), "typical": (0.8, 0.02), "contested": (0.6, 0.05)}
# base_params carry NO seed/state, so each re-key draws a fresh key (a real re-key).
FAMILIES = {
    "lfsr": {"length": 7},
    "lagged_fib": {},
    "markov": {"alpha": 0.2},
    "csprng": {},
}
GEN_LABELS = {"lfsr": "LFSR (in-bank)", "lagged_fib": "lagged-Fib. (held out)",
              "markov": "Markov", "csprng": "CSPRNG (control)"}

#: a transmitter that never evades (shared with the duel; the fixed-tape baseline)
NO_EVASION = NO_EVASION_CFG


def evasion_config(aggr: float, feedback_mode: str = "spectrum") -> EvasionConfig:
    """Map an aggressiveness scalar in ``[0, 1]`` to an :class:`EvasionConfig`.

    ``aggr = 0`` is a timid radio (deflects only off a very hot channel, never
    spreads or re-keys in practice); ``aggr = 1`` is aggressive (deflects early,
    spreads under mild pressure, re-keys readily).
    """
    a = float(np.clip(aggr, 0.0, 1.0))
    lerp = lambda lo, hi: lo + (hi - lo) * a  # noqa: E731
    return EvasionConfig(
        avoid_threshold=lerp(0.6, 0.15),
        entropy_pressure=lerp(0.85, 0.25),
        rekey_pressure=lerp(0.97, 0.5),
        feedback_mode=feedback_mode,
    )


def _duel(family: str, sensor: SensorModel, ch: ChannelSet, cfg: EvasionConfig,
          seed: int, *, jammer: OTSJammer, window: int) -> dict:
    """One duel; ``jammer`` is reused across duels (``run_duel`` resets it)."""
    rng = np.random.default_rng(seed)
    emitter = AdaptiveEvasionGenerator.from_family(
        family, ch, rng, base_params=FAMILIES[family], config=cfg)
    return run_duel(emitter, jammer, sensor, ch, rng, window=window).aggregate()


def _mean(dicts: list[dict], key: str) -> float:
    vals = [d[key] for d in dicts if key in d and not np.isnan(d.get(key, np.nan))]
    return float(np.mean(vals)) if vals else float("nan")


def compute(*, M=C.DEFAULT_M, window=160, n_seeds=20, budget=3.0, max_tones=4,
            plan="knapsack", sensor="strong", aggr_grid=(0.0, 0.25, 0.5, 0.75, 1.0),
            **_) -> dict:
    ch = ChannelSet.of_size(M)
    p_d, p_fa = SENSORS[sensor]
    sens = SensorModel(p_d=p_d, p_fa=p_fa)
    seeds = C.seeds(n_seeds)
    # One jammer (one expensive predictor), reused and reset across every duel.
    jammer = OTSJammer(ch, sens, EffectModel(ch, tau=1.0), budget=budget, plan=plan,
                       max_tones=max_tones)

    # Panel A: evasion OFF vs ON per family (aggressive evader), one sensor.
    on_cfg = evasion_config(1.0)
    panelA = {}
    for fam in FAMILIES:
        off = [_duel(fam, sens, ch, NO_EVASION, s, jammer=jammer, window=window) for s in seeds]
        on = [_duel(fam, sens, ch, on_cfg, s, jammer=jammer, window=window) for s in seeds]
        panelA[fam] = {
            "off": {k: _mean(off, k) for k in ("denied_frac", "mean_effect", "nexthop_top1", "kappa_mean", "mean_tones")},
            "on": {k: _mean(on, k) for k in ("denied_frac", "mean_effect", "nexthop_top1", "kappa_mean", "mean_tones")},
            "rekeys_on": _mean(on, "rekeys"),
            "denial_drop": _mean(off, "denied_frac") - _mean(on, "denied_frac"),
        }

    # Panel B: the game curve -- denial vs aggressiveness for the predictable sources.
    panelB = {"aggr": list(aggr_grid), "series": {}}
    for fam in ("lfsr", "lagged_fib"):
        den, kap = [], []
        for a in aggr_grid:
            cfg = evasion_config(a)
            runs = [_duel(fam, sens, ch, cfg, s, jammer=jammer, window=window) for s in seeds]
            den.append(_mean(runs, "denied_frac"))
            kap.append(_mean(runs, "kappa_mean"))
        panelB["series"][fam] = {"denied_frac": den, "kappa_mean": kap}

    return {"meta": {"M": M, "window": window, "n_seeds": n_seeds, "budget": budget,
                     "max_tones": max_tones, "plan": plan, "sensor": sensor,
                     "p_d": p_d, "p_fa": p_fa,
                     "n_duels": n_seeds * (len(FAMILIES) * 2 + 2 * len(aggr_grid))},
            "panelA": panelA, "panelB": panelB}


def plot(results: dict, path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.4, 4.6))

    # Panel A: OFF vs ON denial per family.
    fams = list(results["panelA"].keys())
    x = np.arange(len(fams))
    w = 0.38
    off = [results["panelA"][f]["off"]["denied_frac"] for f in fams]
    on = [results["panelA"][f]["on"]["denied_frac"] for f in fams]
    b1 = axA.bar(x - w / 2, off, w, label="emitter oblivious (fixed tape)", color=style.PALETTE["blue"])
    b2 = axA.bar(x + w / 2, on, w, label="emitter evades (closed loop)", color=style.PALETTE["vermillion"])
    for bars, vals, fkey in ((b1, off, "off"), (b2, on, "on")):
        for b, v, f in zip(bars, vals, fams):
            k = results["panelA"][f][fkey]["kappa_mean"]
            axA.text(b.get_x() + b.get_width() / 2, v + 0.015,
                     f"{v:.2f}\n$\\kappa${k:.2f}", ha="center", fontsize=7.5)
    axA.set_xticks(x); axA.set_xticklabels([GEN_LABELS[f] for f in fams], fontsize=8, rotation=12)
    axA.set_ylabel("fraction of hops denied"); axA.set_ylim(0, 1.18)
    axA.set_title(f"A. Evasion dividend ({results['meta']['sensor']} sensor, "
                  f"{results['meta']['plan']} jammer)")
    axA.legend(fontsize=8, loc="upper right")

    # Panel B: the game curve.
    aggr = results["panelB"]["aggr"]
    for i, (fam, d) in enumerate(results["panelB"]["series"].items()):
        col = style.SERIES_COLORS[i % len(style.SERIES_COLORS)]
        axB.plot(aggr, d["denied_frac"], "-o", color=col, label=f"{GEN_LABELS[fam]} denial")
        axB.plot(aggr, d["kappa_mean"], "--s", color=col, ms=4, alpha=0.6,
                 label=f"{GEN_LABELS[fam]} $\\kappa$")
    axB.set_xlabel("evasion aggressiveness"); axB.set_ylabel("fraction denied / $\\kappa$")
    axB.set_ylim(0, 1.04)
    axB.set_title("B. The game: harder evasion → less denial, lower $\\kappa$")
    axB.legend(fontsize=7.5, loc="upper right")

    fig.tight_layout()
    return style.save(fig, path)


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n-seeds", type=int, default=20)
    p.add_argument("--window", type=int, default=160)
    p.add_argument("--sensor", default="strong", choices=list(SENSORS))
    p.add_argument("--plan", default="knapsack", choices=["knapsack", "conformal", "reactive"])
    p.add_argument("--plot-only", action="store_true")
    args = p.parse_args(argv)
    jpath = C.RESULTS_DIR / f"{NAME}.json"
    results = (load_json(jpath) if args.plot_only
               else compute(n_seeds=args.n_seeds, window=args.window, sensor=args.sensor, plan=args.plan))
    if not args.plot_only:
        save_json(results, jpath)
    fig = plot(results, C.FIG_DIR / f"{NAME}.png")
    print(f"[{NAME}] wrote {jpath} and {fig}  ({results['meta']['n_duels']} duels)")
    print(f"  Panel A ({results['meta']['sensor']} sensor, {results['meta']['plan']} jammer):")
    for fam, d in results["panelA"].items():
        print(f"    {fam:11s} denied {d['off']['denied_frac']:.2f} -> {d['on']['denied_frac']:.2f} "
              f"(drop {d['denial_drop']:+.2f})  kappa {d['off']['kappa_mean']:.2f} -> {d['on']['kappa_mean']:.2f}  "
              f"rekeys={d['rekeys_on']:.1f}")


if __name__ == "__main__":
    main()
