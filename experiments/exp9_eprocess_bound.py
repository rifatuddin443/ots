"""Experiment 9: the anytime-valid false-identification bound (paper Section 10).

With the e-process stopping rule, Ville's inequality bounds the probability of
ever declaring the wrong generator by the nominal beta, *whenever* the system
stops. That guarantee is against the mixture-of-competitors null, i.e. it holds
when the true generator is itself in the bank. So the bound is tested exactly
there: in-bank deterministic truths, and a false-ID is declaring a *different*
in-bank generator. We sweep beta and plot the empirical false-ID against the
nominal diagonal; it should stay on or under the line.

The out-of-bank case is a different question (the truth is outside every
hypothesis, so this martingale says nothing); H_0 is the protection there, and
that is what Experiment 4 measures. Reporting the bound only where its
assumption holds is the honest-guarantee discipline of Section 11.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments import common as C
from experiments import style
from fhew.channels import ChannelSet
from fhew.config import in_bank_labels
from fhew.experts.hypothesis import HypothesisSpec
from fhew.logging_utils import load_json, save_json
from fhew.sensor import SensorModel
from fhew.sim.engine import run_bank

NAME = "exp9_eprocess_bound"

# Deterministic in-bank families only: the true generator is always in the bank,
# so the e-process null (truth among the competitors) holds and the bound applies.
_SPECS = [
    HypothesisSpec("lfsr", "lfsr", "deterministic", grid={"length": [5, 6, 7]}),
    HypothesisSpec("weak_seed", "weak_seed", "deterministic", grid={"keyspace": 256}),
    HypothesisSpec("lcg", "lcg", "deterministic", grid={"seed_max": 1024}),
]


def compute(*, M=C.DEFAULT_M, window=C.DEFAULT_WINDOW, n_seeds=60, p_d=0.7, p_fa=0.05,
            beta_grid=(0.5, 0.2, 0.1, 0.05, 0.02, 0.01),
            generators=("lfsr", "weak_seed", "lcg")) -> dict:
    # The bound is a (high-probability) rate statement, so the smallest beta
    # needs enough episodes that 1/N is well below it; floor the resolution here.
    n_seeds = max(n_seeds, 100)
    ch = ChannelSet.of_size(M)
    tgt = in_bank_labels(_SPECS)
    sensor = SensorModel(p_d=p_d, p_fa=p_fa)
    out = {"meta": {"M": M, "window": window, "n_seeds": n_seeds, "p_d": p_d, "p_fa": p_fa,
                    "beta_grid": list(beta_grid), "generators": list(generators)}, "rows": []}
    for beta in beta_grid:
        bank = C.make_bank(ch, window, sensor, _SPECS,
                           stopping={"kind": "eprocess", "beta": beta})
        res = []
        for gen in generators:
            res += [run_bank(bank, C.episode_for(gen, ch, sensor, window, s,
                                                 target_labels=tgt))
                    for s in C.seeds(n_seeds)]
        # false-ID = crossed the e-threshold for a *different* in-bank generator
        wrong = [r for r in res if r.declared and r.decision_label != r.target_label
                 and r.decision_label != "unknown"]
        out["rows"].append({
            "beta": beta, "n": len(res),
            "empirical_false_id": len(wrong) / len(res),
            "declaration_rate": float(np.mean([r.declared for r in res])),
        })
    return out


def plot(results: dict, path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    rows = results["rows"]
    beta = [r["beta"] for r in rows]
    fid = [r["empirical_false_id"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    lim = max(beta) * 1.1
    ax.fill_between([0, lim], [0, lim], lim, color=style.PALETTE["vermillion"], alpha=0.07)
    ax.plot([0, lim], [0, lim], color=style.PALETTE["grey"], ls="--",
            label="nominal bound (Ville): $y = \\beta$")
    ax.plot(beta, fid, "-o", color=style.PALETTE["blue"],
            label="empirical false-ID")
    for b, f in zip(beta, fid):
        ax.annotate(f"{f:.3f}", (b, f), textcoords="offset points", xytext=(6, -2),
                    fontsize=8)
    ax.set_xlabel(r"nominal $\beta$"); ax.set_ylabel("empirical false-identification rate")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_title(f"e-process false-ID stays under the bound\n"
                 f"(in-bank truths, $P_d={results['meta']['p_d']}$, "
                 f"$P_{{fa}}={results['meta']['p_fa']}$)")
    ax.legend(loc="upper left", fontsize=9)
    return style.save(fig, path)


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n-seeds", type=int, default=60)
    p.add_argument("--plot-only", action="store_true")
    args = p.parse_args(argv)
    jpath = C.RESULTS_DIR / f"{NAME}.json"
    results = load_json(jpath) if args.plot_only else compute(n_seeds=args.n_seeds)
    if not args.plot_only:
        save_json(results, jpath)
    fig = plot(results, C.FIG_DIR / f"{NAME}.png")
    print(f"[{NAME}] wrote {jpath} and {fig}")
    for r in results["rows"]:
        flag = "OK" if r["empirical_false_id"] <= r["beta"] else "OVER"
        print(f"  beta={r['beta']:.2f}: empirical_false_id={r['empirical_false_id']:.3f} "
              f"decl_rate={r['declaration_rate']:.2f}  [{flag}]")


if __name__ == "__main__":
    main()
