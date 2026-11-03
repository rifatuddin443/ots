"""Experiment 19: sequence-type coverage — accuracy and hops-to-lock, before vs after.

Scores the next-channel predictor on every structured sequence family and on
secrets far outside the recce ranges, with and without the structured experts
(`rule_search`, `seed_beam_*`, `bm_wide`). Two numbers per type:

* **steady top-1** — next-channel accuracy over the final 40 slots;
* **hops to lock** — the median number of observed slots after which the
  next five predictions are all correct (``nan`` = never locked).

Groups: `structured` (arithmetic progression, polynomial, linear / nonlinear
recurrence, lookup table, the two hybrids, the noisy rule), `beyond_range`
(weak-seed keyspace 2^18, an LCG seed anywhere in its modulus, a 16-bit LFSR)
and the CSPRNG `control`, which nothing should predict.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments import common as C
from experiments import style
from fhew.channels import ChannelSet
from fhew.config import STRUCTURED_GENERATORS, prediction_expert_specs
from fhew.logging_utils import load_json, save_json
from fhew.predict import ChannelPredictor
from fhew.sensor import SensorModel
from fhew.sim import make_episode

NAME = "exp19_sequence_coverage"

#: (case label, generator, params) for secrets the narrow recce ranges miss
BEYOND_RANGE = (
    ("weak_seed_2^18", "weak_seed", {"keyspace": 2 ** 18}),
    ("lcg_any_seed", "lcg", {}),
    ("lfsr_16", "lfsr", {"length": 16}),
)
CASES = ([(g, g, {}) for g in STRUCTURED_GENERATORS] + list(BEYOND_RANGE)
         + [("csprng", "csprng", {})])
LOCK_RUN = 5


def _group(label: str) -> str:
    if label in STRUCTURED_GENERATORS:
        return "structured"
    return "control" if label == "csprng" else "beyond_range"


def _run(pred: ChannelPredictor, ep, steady: int = 40) -> tuple[float, float]:
    """Steady-state top-1 and hops-to-lock for one episode."""
    pred.reset()
    hits = []
    for t, o in enumerate(ep.observations[:-1]):
        pred.observe(o)
        hits.append(int(pred.predict_top1() == int(ep.true_channels[t + 1])))
    lock = next((t + 1 for t in range(len(hits) - LOCK_RUN + 1) if all(hits[t:t + LOCK_RUN])),
                float("nan"))
    return float(np.mean(hits[-steady:])), float(lock)


def compute(*, M=C.DEFAULT_M, window=120, n_seeds=8,
            sensors=((0.95, 0.0), (0.85, 0.02))) -> dict:
    ch = ChannelSet.of_size(M)
    out = {"meta": {"M": M, "window": window, "n_seeds": n_seeds, "chance": 1.0 / M,
                    "lock_run": LOCK_RUN, "sensors": [list(s) for s in sensors]},
           "sensors": {}}
    for p_d, p_fa in sensors:
        sensor = SensorModel(p_d=p_d, p_fa=p_fa)
        before = ChannelPredictor(ch, sensor, specs=prediction_expert_specs(M, structured=False))
        after = ChannelPredictor(ch, sensor, specs=prediction_expert_specs(M))
        rows = {}
        for label, gen, params in CASES:
            acc = {"before_top1": [], "before_lock": [], "after_top1": [], "after_lock": []}
            for s in C.seeds(n_seeds):
                ep = make_episode(gen, ch, sensor, window, np.random.default_rng(s),
                                  gen_params=dict(params), seed=s)
                for name, pred in (("before", before), ("after", after)):
                    top1, lock = _run(pred, ep)
                    acc[f"{name}_top1"].append(top1)
                    acc[f"{name}_lock"].append(lock)
            rows[label] = {
                "group": _group(label),
                "before_top1": float(np.mean(acc["before_top1"])),
                "after_top1": float(np.mean(acc["after_top1"])),
                "before_lock": float(np.nanmedian(acc["before_lock"]))
                if not np.all(np.isnan(acc["before_lock"])) else float("nan"),
                "after_lock": float(np.nanmedian(acc["after_lock"]))
                if not np.all(np.isnan(acc["after_lock"])) else float("nan"),
                "after_locked_frac": float(np.mean(~np.isnan(acc["after_lock"]))),
            }
        out["sensors"][f"pd{p_d}_pfa{p_fa}"] = rows
    return out


def plot(results: dict, path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    keys = list(results["sensors"])
    fig, axes = plt.subplots(len(keys), 2, figsize=(14, 4.4 * len(keys)),
                             gridspec_kw={"width_ratios": [1.5, 1.0]}, squeeze=False)
    chance = results["meta"]["chance"]
    for row, key in zip(axes, keys):
        per = results["sensors"][key]
        labels = list(per)
        x = np.arange(len(labels))
        axA, axB = row
        w = 0.38
        axA.bar(x - w / 2, [per[g]["before_top1"] for g in labels], w,
                label="before (no structured experts)", color=style.PALETTE["orange"])
        axA.bar(x + w / 2, [per[g]["after_top1"] for g in labels], w,
                label="after (+ rule search, seed beam, wide BM)", color=style.PALETTE["blue"])
        axA.axhline(chance, color=style.PALETTE["grey"], ls=":", lw=1)
        axA.set_xticks(x)
        axA.set_xticklabels(labels, rotation=35, ha="right")
        axA.set_ylim(0, 1.1)
        axA.set_ylabel("steady-state top-1")
        axA.set_title(f"Next-channel accuracy by sequence type ({key})")
        axA.legend(fontsize=8, loc="lower left")

        lock_b = [per[g]["before_lock"] for g in labels]
        lock_a = [per[g]["after_lock"] for g in labels]
        cap = results["meta"]["window"]
        axB.barh(x + w / 2, [cap if np.isnan(v) else v for v in lock_b], w,
                 color=style.PALETTE["orange"], alpha=0.85, label="before")
        axB.barh(x - w / 2, [cap if np.isnan(v) else v for v in lock_a], w,
                 color=style.PALETTE["blue"], label="after")
        axB.set_yticks(x)
        axB.set_yticklabels(labels)
        axB.invert_yaxis()
        axB.set_xlabel(f"hops to lock (median; full bar = never in {cap})")
        axB.set_title("How fast it locks")
        axB.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return style.save(fig, path)


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n-seeds", type=int, default=8)
    p.add_argument("--plot-only", action="store_true")
    args = p.parse_args(argv)
    jpath = C.RESULTS_DIR / f"{NAME}.json"
    results = load_json(jpath) if args.plot_only else compute(n_seeds=args.n_seeds)
    if not args.plot_only:
        save_json(results, jpath)
    fig = plot(results, C.FIG_DIR / f"{NAME}.png")
    print(f"[{NAME}] wrote {jpath} and {fig}")
    for key, per in results["sensors"].items():
        print(f"  {key}")
        print(f"    {'type':16s} {'before':>7s} {'after':>7s} {'lock_b':>7s} {'lock_a':>7s}")
        for g, d in per.items():
            print(f"    {g:16s} {d['before_top1']:7.2f} {d['after_top1']:7.2f} "
                  f"{d['before_lock']:7.1f} {d['after_lock']:7.1f}")


if __name__ == "__main__":
    main()
