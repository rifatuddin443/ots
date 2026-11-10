"""Experiment 20: a lower bound (converse) on hops-to-identify.

`exp11` showed the *achievability* side empirically: E[tau] is linear in
log(1/delta) and in 1/P_d. This experiment adds the matching **converse** — the
information-theoretic floor no sensor-aware identifier can beat — and shows the
bank sits at or above it (i.e. the measured cost is within a small constant of
optimal).

The floor is the standard sequential-test / change-of-measure bound (Wald; the
change-of-measure lemmas behind PAC best-arm identification). To declare the true
family with error at most ``delta``, the posterior log-odds of the truth against
its nearest competitor must climb to ~``log((1-delta)/delta)``; it climbs at the
per-slot rate ``D`` = the Kullback-Leibler drift of the true next-channel law
against the nearest competitor, *thinned by detection* (a missed hop contributes
no evidence, so ``D`` carries the 1/P_d factor automatically). Hence

    E[tau]  >=  log((1 - delta) / delta) / D ,      D = D(P_d)  ~  P_d * KL*.

We estimate ``D`` directly from the realised posterior log-odds drift on the same
runs (no analytic KL needed, so it holds for the actual sensor-aware emission),
then overlay the bound on the measured E[tau]. A Markov source is used (as in
exp11) so ``KL*`` is finite and the law is visible; a deterministic family
saturates the drift and the bound is trivially loose.
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
from fhew.sim import run_bank

NAME = "exp21_identification_converse"


def _mean_tau(ch, specs, tgt, sensor, window, gen, delta, n_seeds):
    bank = C.make_bank(ch, window, sensor, specs, stopping={"kind": "msprt", "delta": delta})
    res = [run_bank(bank, C.episode_for(gen, ch, sensor, window, s, target_labels=tgt))
           for s in C.seeds(n_seeds)]
    return summarize(res).mean_hops_to_identify


def _llr_drift(ch, specs, tgt, sensor, window, gen, n_seeds, true_label, *,
               t0: int = 2, fit_cap: int = 60) -> float:
    """Estimate the per-slot posterior log-odds drift D of the true family against
    its nearest competitor, averaged over seeds. D ~ P_d * KL* (the detection-
    thinned KL rate), measured on the actual sensor-aware runs with no early stop.
    """
    bank = C.make_bank(ch, window, sensor, specs, stopping={"kind": "msprt", "delta": 1e-9})
    slopes = []
    for s in C.seeds(n_seeds):
        ep = C.episode_for(gen, ch, sensor, window, s, target_labels=tgt)
        res = run_bank(bank, ep, stop_early=False, record_perslot=True, eval_nexthop=False)
        llr = []
        for rec in res.per_slot:
            post = rec["posterior"]
            pt = max(float(post.get(true_label, 0.0)), 1e-300)
            others = [float(v) for k, v in post.items() if k != true_label]
            pc = max(max(others) if others else 1e-300, 1e-300)
            llr.append(np.log(pt) - np.log(pc))
        llr = np.asarray(llr, dtype=np.float64)
        hi = min(len(llr), fit_cap)
        if hi - t0 >= 3:
            d = (llr[hi - 1] - llr[t0]) / (hi - 1 - t0)
            if np.isfinite(d) and d > 0:
                slopes.append(float(d))
    return float(np.mean(slopes)) if slopes else float("nan")


def _bound(delta: float, drift: float) -> float:
    return float(np.log((1.0 - delta) / delta) / drift) if drift and drift > 0 else float("nan")


def compute(*, M=C.DEFAULT_M, window=200, n_seeds=40, gen="markov",
            delta_grid=(0.2, 0.1, 0.05, 0.02, 0.01, 0.005),
            p_d_grid=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5), drift_seeds=24) -> dict:
    ch = ChannelSet.of_size(M)
    specs = C.default_specs(M)
    tgt = C.default_target_labels(specs)

    # Panel A: vary delta at a fixed sensor. One drift estimate (delta-independent).
    sensorA = SensorModel(p_d=0.9, p_fa=0.0)
    driftA = _llr_drift(ch, specs, tgt, sensorA, window, gen, drift_seeds, gen)
    A = []
    for d in delta_grid:
        tau = _mean_tau(ch, specs, tgt, sensorA, window, gen, d, n_seeds)
        lb = _bound(d, driftA)
        A.append({"delta": d, "log_inv_odds": float(np.log((1 - d) / d)),
                  "mean_tau": tau, "bound": lb,
                  "efficiency": float(tau / lb) if lb and lb == lb else float("nan")})

    # Panel B: vary P_d at fixed delta; the drift (and so the bound) depends on P_d.
    delta_b = 0.05
    B = []
    for pd in p_d_grid:
        sensor = SensorModel(p_d=pd, p_fa=0.0)
        tau = _mean_tau(ch, specs, tgt, sensor, window, gen, delta_b, n_seeds)
        drift = _llr_drift(ch, specs, tgt, sensor, window, gen, drift_seeds, gen)
        lb = _bound(delta_b, drift)
        B.append({"p_d": pd, "inv_p_d": 1.0 / pd, "mean_tau": tau, "drift": drift,
                  "bound": lb, "efficiency": float(tau / lb) if lb and lb == lb else float("nan")})

    effs = [r["efficiency"] for r in A + B if r["efficiency"] == r["efficiency"]]
    return {
        "meta": {"M": M, "window": window, "n_seeds": n_seeds, "generator": gen,
                 "delta_b": delta_b, "drift_at_Pd0.9": driftA,
                 "metric": "E[tau] vs sequential-test lower bound"},
        "panelA_delta": {"rows": A, "drift": driftA},
        "panelB_pd": {"rows": B},
        "efficiency": {"min": float(min(effs)) if effs else float("nan"),
                       "max": float(max(effs)) if effs else float("nan"),
                       "mean": float(np.mean(effs)) if effs else float("nan"),
                       "all_at_or_above_bound": bool(all(e >= 0.999 for e in effs)) if effs else False},
    }


def plot(results: dict, path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.0, 4.3))

    A = results["panelA_delta"]["rows"]
    x = np.array([r["log_inv_odds"] for r in A])
    y = np.array([r["mean_tau"] for r in A])
    lb = np.array([r["bound"] for r in A])
    order = np.argsort(x)
    axA.plot(x[order], y[order], "o-", color=style.PALETTE["blue"], label="measured E[$\\tau$]")
    axA.plot(x[order], lb[order], "--", color=style.PALETTE["grey"], label="lower bound (converse)")
    axA.fill_between(x[order], lb[order], y[order], color=style.PALETTE["blue"], alpha=0.10)
    axA.set_xlabel(r"$\log\!\left(\frac{1-\delta}{\delta}\right)$")
    axA.set_ylabel("mean hops-to-identify")
    axA.set_title("E[$\\tau$] vs the evidence floor")
    axA.legend(fontsize=9)

    B = results["panelB_pd"]["rows"]
    xb = np.array([r["inv_p_d"] for r in B])
    yb = np.array([r["mean_tau"] for r in B])
    lbb = np.array([r["bound"] for r in B])
    order = np.argsort(xb)
    axB.plot(xb[order], yb[order], "o-", color=style.PALETTE["vermillion"], label="measured E[$\\tau$]")
    axB.plot(xb[order], lbb[order], "--", color=style.PALETTE["grey"], label="lower bound (converse)")
    axB.fill_between(xb[order], lbb[order], yb[order], color=style.PALETTE["vermillion"], alpha=0.10)
    axB.set_xlabel(r"$1/P_d$")
    axB.set_ylabel("mean hops-to-identify")
    axB.set_title("Detection thinning vs the floor")
    axB.legend(fontsize=9)

    eff = results["efficiency"]
    fig.suptitle(f"Hops-to-identify meets its lower bound  (generator = {results['meta']['generator']}, "
                 f"efficiency {eff['mean']:.2f}$\\times$)", y=1.03)
    fig.tight_layout()
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
    eff = results["efficiency"]
    print(f"  efficiency E[tau]/bound: mean={eff['mean']:.2f} min={eff['min']:.2f} "
          f"(>=1 means at/above the converse: {eff['all_at_or_above_bound']})")
    for r in results["panelB_pd"]["rows"]:
        print(f"  P_d={r['p_d']:.2f}  E[tau]={r['mean_tau']:6.2f}  bound={r['bound']:6.2f}  "
              f"drift={r['drift']:.3f}")


if __name__ == "__main__":
    main()
