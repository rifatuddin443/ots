"""Extra, larger experiments run for the IEEE conference paper (docs/paper/ieee).

These reuse the existing experiment code and the same information-parity rule
(experts see only observations) and only change *how much* is run and *where*:

  confusion   family-level confusion matrices (true generator -> declared family)
              of the identification bank at four sensors, 13 generators
              (10 library families + 3 held-out), 100 seeds each.
  mismatch    mis-specified sensor: data generated at a true P_d, scored with an
              assumed P_d (and the same for P_fa).
  predict     next-channel prediction (exp14 protocol) at strong / typical /
              contested sensors, 30 seeds.
  conformal   conformal next-channel sets (exp20 protocol) at the same sensors.
  switch      switch tracking (exp18 protocol) with 24 random plans per sensor.

Usage:  python -m experiments.paper_extras confusion mismatch predict conformal switch
JSON goes to docs/paper/ieee/data/ (committed, so the paper figures re-render
without re-simulating).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from experiments import common as C
from experiments import exp14_next_channel_prediction as E14
from experiments import exp18_switch_tracking as E18
from experiments import exp20_conformal_sets as E20
from fhew.channels import ChannelSet
from fhew.config import GENERATOR_TARGET_FAMILY, NEW_ALGORITHM_GENERATORS
from fhew.logging_utils import save_json
from fhew.sensor import SensorModel
from fhew.sim import run_bank
from fhew.tracker import ExpertLibrary

OUT = Path("docs/paper/ieee/data")

SENSORS = {"perfect": (1.0, 0.0), "strong": (0.95, 0.01),
           "typical": (0.8, 0.02), "contested": (0.6, 0.05)}
ALGEBRAIC = frozenset({"lfsr", "filtered_lfsr", "lcg", "lcg_full", "weak_seed"})
FAMILIES = ("lfsr", "filtered_lfsr", "lcg", "lcg_full", "weak_seed", "chaos", "markov", "unknown")


def confusion(n_seeds: int = 100, window: int = 160, M: int = C.DEFAULT_M) -> dict:
    """True generator -> declared family, identification bank + MSPRT (delta 0.05)."""
    ch = ChannelSet.of_size(M)
    specs = C.id_specs(M)
    gens = tuple(C.ALL_GENERATORS) + tuple(NEW_ALGORITHM_GENERATORS)
    out = {"meta": {"M": M, "window": window, "n_seeds": n_seeds, "families": list(FAMILIES),
                    "generators": list(gens), "delta": 0.05}, "sensors": {}}
    for name, (p_d, p_fa) in SENSORS.items():
        sensor = SensorModel(p_d=p_d, p_fa=p_fa)
        bank = C.make_bank(ch, window, sensor, specs)
        counts = {g: {f: 0 for f in FAMILIES} for g in gens}
        hops = {g: [] for g in gens}
        for g in gens:
            target = GENERATOR_TARGET_FAMILY.get(g, "unknown")
            for s in C.seeds(n_seeds, base=4000):
                ep = C.episode_for(g, ch, sensor, window, s)
                ep.target_label = target
                bank.reset()
                r = run_bank(bank, ep, eval_nexthop=False)
                counts[g][r.decision_label] += 1
                if r.correct and r.declared:
                    hops[g].append(r.tau_stop)
        per = {}
        for g in gens:
            target = GENERATOR_TARGET_FAMILY.get(g, "unknown")
            hard = sum(v for f, v in counts[g].items() if f in ALGEBRAIC and f != target)
            per[g] = {"target": target, "accuracy": counts[g][target] / n_seeds,
                      "hard_false_id": hard / n_seeds, "n_hard": hard,
                      "mean_hops": float(np.mean(hops[g])) if hops[g] else None}
        out["sensors"][name] = {"p_d": p_d, "p_fa": p_fa, "counts": counts, "per_generator": per}
        print(f"[confusion] {name} done", flush=True)
    return out


def mismatch(n_seeds: int = 40, window: int = 100, M: int = C.DEFAULT_M) -> dict:
    """Bank scores with an *assumed* sensor while the data come from the true one."""
    ch = ChannelSet.of_size(M)
    specs = C.default_specs(M)
    targets = C.default_target_labels(specs)
    gens = C.IN_BANK_GENERATORS
    grid = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)

    def cell(true_s: SensorModel, assumed_s: SensorModel) -> dict:
        bank = C.make_bank(ch, window, assumed_s, specs)
        res = []
        for g in gens:
            for s in C.seeds(n_seeds, base=6000):
                ep = C.episode_for(g, ch, true_s, window, s, target_labels=targets)
                bank.reset()
                res.append(run_bank(bank, ep, eval_nexthop=False))
        hops = [r.tau_stop for r in res if r.correct and r.declared]
        return {"accuracy": float(np.mean([r.correct for r in res])),
                "hard_false_id": float(np.mean([C.is_hard_false_id(r) for r in res])),
                "mean_hops": float(np.mean(hops)) if hops else None, "n": len(res)}

    out = {"meta": {"M": M, "window": window, "n_seeds": n_seeds, "generators": list(gens),
                    "p_d_grid": list(grid), "p_fa_fixed": 0.02, "p_d_fixed": 0.8},
           "pd": [], "pfa": []}
    for pd_true in grid:
        row = []
        for pd_assumed in grid:
            # an assumed P_d of exactly 1 makes a miss impossible under the model
            a = min(pd_assumed, 0.99)
            row.append({"p_d_true": pd_true, "p_d_assumed": pd_assumed,
                        **cell(SensorModel(pd_true, 0.02), SensorModel(a, 0.02))})
        out["pd"].append(row)
        print(f"[mismatch] true P_d={pd_true} done", flush=True)
    pfa_grid = (0.005, 0.02, 0.05, 0.1)
    out["meta"]["p_fa_grid"] = list(pfa_grid)
    for fa_true in pfa_grid:
        row = []
        for fa_assumed in pfa_grid:
            row.append({"p_fa_true": fa_true, "p_fa_assumed": fa_assumed,
                        **cell(SensorModel(0.8, fa_true), SensorModel(0.8, fa_assumed))})
        out["pfa"].append(row)
    return out


def predict(n_seeds: int = 30) -> dict:
    out = {}
    for name in ("strong", "typical", "contested"):
        p_d, p_fa = SENSORS[name]
        r = E14.compute(n_seeds=n_seeds, p_d=p_d, p_fa=p_fa, pd_sweep=(), sweep_seeds=1)
        out[name] = {"meta": r["meta"], "operating_point": r["operating_point"]}
        print(f"[predict] {name} done", flush=True)
    return {"sensors": out}


def conformal(n_seeds: int = 30) -> dict:
    out = {}
    for name in ("strong", "typical", "contested"):
        p_d, p_fa = SENSORS[name]
        out[name] = E20.compute(n_seeds=n_seeds, p_d=p_d, p_fa=p_fa)
        print(f"[conformal] {name} done", flush=True)
    return {"sensors": out}


def switch(n_plans: int = 24) -> dict:
    ch = ChannelSet.of_size(C.DEFAULT_M)
    library = ExpertLibrary.build(ch, E18.WINDOW + 2)
    out = {}
    for name, (p_d, p_fa) in SENSORS.items():
        sensor = SensorModel(p_d, p_fa)
        rows, ms = [], []
        for seed in C.seeds(n_plans, base=1800):
            plan, switches, steps, hits, t = E18.run_plan(library, sensor, seed)
            rows += E18.score(plan, switches, steps, hits)
            ms.append(t * 1000)
        summ = E18.summarise(rows, n_plans * E18.WINDOW, float(np.mean(ms)))
        summ["n_dangerous"] = int(sum(r["dangerous_lock"] for r in rows))
        out[name] = {"p_d": p_d, "p_fa": p_fa, **summ, "rows": rows}
        print(f"[switch] {name} done", flush=True)
    return {"meta": {"M": C.DEFAULT_M, "window": E18.WINDOW, "n_plans": n_plans,
                     "regime_hops": [50, 90]}, "sensors": out}


TASKS = {"confusion": confusion, "mismatch": mismatch, "predict": predict,
         "conformal": conformal, "switch": switch}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("tasks", nargs="+", choices=list(TASKS))
    args = p.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    for name in args.tasks:
        t0 = time.time()
        save_json(TASKS[name](), OUT / f"extra_{name}.json")
        print(f"[{name}] wrote {OUT / f'extra_{name}.json'} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
