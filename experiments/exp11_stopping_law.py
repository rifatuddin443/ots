"""Experiment 11: the stopping-time law (paper Section 10).

The predicted law is

    E[tau_stop] ~ log(1/delta) / (P_d * min_{k != k*} KL(H_k* || H_k)),

a standard sequential-test result whose contribution here is the P_d thinning
and the match to measurement. We verify the two tractable dependencies on one
clean generator: E[tau] is linear in log(1/delta) (panel A) and linear in 1/P_d
(panel B, the detection-probability thinning). Linear fits with R^2 are reported.
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
from fhew.sim.engine import run_bank

NAME = "exp11_stopping_law"


def _mean_tau(ch, specs, tgt, sensor, window, gen, delta, n_seeds):
    bank = C.make_bank(ch, window, sensor, specs, stopping={"kind": "msprt", "delta": delta})
    res = [run_bank(bank, C.episode_for(gen, ch, sensor, window, s, target_labels=tgt))
           for s in C.seeds(n_seeds)]
    s = summarize(res)
    return s.mean_hops_to_identify, s.accuracy


def _linfit(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    A = np.vstack([x, np.ones_like(x)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), float(r2)


def compute(*, M=C.DEFAULT_M, window=150, n_seeds=40, gen="markov",
            delta_grid=(0.2, 0.1, 0.05, 0.02, 0.01, 0.005),
            p_d_grid=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5)) -> dict:
    # A finite-per-slot-KL generator (Markov) is used so both the log(1/delta)
    # scaling and the 1/P_d thinning are observable; a deterministic generator
    # saturates (its wrong hypotheses get ~zero likelihood almost immediately, so
    # delta stops mattering), which would hide the law rather than test it.
    ch = ChannelSet.of_size(M)
    specs = C.default_specs(M)
    tgt = C.default_target_labels(specs)

    # Panel A: vary delta at fixed P_d.
    sensorA = SensorModel(p_d=0.9, p_fa=0.0)
    A = [{"delta": d, "log_inv_delta": float(np.log(1.0 / d)),
          "mean_tau": _mean_tau(ch, specs, tgt, sensorA, window, gen, d, n_seeds)[0]}
         for d in delta_grid]
    sa, ia, r2a = _linfit([r["log_inv_delta"] for r in A], [r["mean_tau"] for r in A])

    # Panel B: vary P_d at fixed delta.
    B = []
    for pd in p_d_grid:
        tau, _ = _mean_tau(ch, specs, tgt, SensorModel(p_d=pd, p_fa=0.0), window, gen, 0.05, n_seeds)
        B.append({"p_d": pd, "inv_p_d": 1.0 / pd, "mean_tau": tau})
    sb, ib, r2b = _linfit([r["inv_p_d"] for r in B], [r["mean_tau"] for r in B])

    return {"meta": {"M": M, "window": window, "n_seeds": n_seeds, "generator": gen},
            "panelA_delta": {"rows": A, "slope": sa, "intercept": ia, "r2": r2a},
            "panelB_pd": {"rows": B, "slope": sb, "intercept": ib, "r2": r2b}}


def plot(results: dict, path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    A = results["panelA_delta"]
    x = np.array([r["log_inv_delta"] for r in A["rows"]])
    y = np.array([r["mean_tau"] for r in A["rows"]])
    axA.plot(x, y, "o", color=style.PALETTE["blue"])
    xs = np.linspace(x.min(), x.max(), 50)
    axA.plot(xs, A["slope"] * xs + A["intercept"], "-", color=style.PALETTE["blue"],
             label=f"fit $R^2$={A['r2']:.3f}")
    axA.set_xlabel(r"$\log(1/\delta)$"); axA.set_ylabel("mean hops-to-identify")
    axA.set_title("E[$\\tau$] vs $\\log(1/\\delta)$"); axA.legend()

    B = results["panelB_pd"]
    xb = np.array([r["inv_p_d"] for r in B["rows"]])
    yb = np.array([r["mean_tau"] for r in B["rows"]])
    axB.plot(xb, yb, "o", color=style.PALETTE["vermillion"])
    xbs = np.linspace(xb.min(), xb.max(), 50)
    axB.plot(xbs, B["slope"] * xbs + B["intercept"], "-", color=style.PALETTE["vermillion"],
             label=f"fit $R^2$={B['r2']:.3f}")
    axB.set_xlabel(r"$1/P_d$"); axB.set_ylabel("mean hops-to-identify")
    axB.set_title("E[$\\tau$] vs $1/P_d$ (detection thinning)"); axB.legend()
    fig.suptitle(f"Stopping-time law, generator = {results['meta']['generator']}", y=1.02)
    return style.save(fig, path)


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n-seeds", type=int, default=40)
    p.add_argument("--plot-only", action="store_true")
    args = p.parse_args(argv)
    jpath = C.RESULTS_DIR / f"{NAME}.json"
    results = load_json(jpath) if args.plot_only else compute(n_seeds=args.n_seeds)
    if not args.plot_only:
        save_json(results, jpath)
    fig = plot(results, C.FIG_DIR / f"{NAME}.png")
    print(f"[{NAME}] wrote {jpath} and {fig}")
    print(f"  E[tau] ~ log(1/delta): slope={results['panelA_delta']['slope']:.2f} "
          f"R2={results['panelA_delta']['r2']:.3f}")
    print(f"  E[tau] ~ 1/P_d:        slope={results['panelB_pd']['slope']:.2f} "
          f"R2={results['panelB_pd']['r2']:.3f}")


if __name__ == "__main__":
    main()
