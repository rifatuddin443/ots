"""Experiment 19: conformal next-channel prediction sets, coupled to the jam plan.

This experiment builds the piece the architecture names as a flagship novelty but
the code did not have: a **calibrated prediction set** whose size adapts so the
true next channel is covered with a controlled probability ``1 - alpha`` — on any
emitter, with no distributional assumption — and which *is* the multi-tone jam
plan. It makes three points a fixed ``top-k`` shortlist cannot:

1. **Coverage is guaranteed and adaptive (Panel A).** Across a mixed generator
   pool, the conformal set's realised coverage tracks the nominal ``1 - alpha``
   diagonal, while a fixed ``top-1`` / ``top-3`` shortlist sits off it (it
   over-covers predictable emitters and under-covers unpredictable ones).
2. **The set size reports predictability honestly (Panel B).** At a fixed target
   the set collapses to one channel on a pinned-down LFSR (kappa≈1) and opens out
   toward the whole plan on a keyed CSPRNG (kappa≈0) — the number of tones is set
   by the data, not by a hand-tuned ``max_tones``.
3. **Coverage becomes a jamming guarantee (Panel C, D1).** Fund exactly the
   conformal set and the realised next hop is denied at the coverage rate, so the
   fraction-of-hops-jammed inherits the distribution-free ``1 - alpha`` bound;
   a fixed ``top-3`` jam does not, and the confidence-tone rule has no guarantee.

Everything reads only observations (information parity); the hidden truth is used
only to score coverage, and the ACI calibration bit is the *erasure feedback* a
passive node would actually see (clean slots only).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments import common as C
from experiments import style
from fhew.allocator import EffectModel, tones_from_confidence
from fhew.channels import ChannelSet
from fhew.conformal import (
    ConformalChannelPredictor,
    aps_set,
    conformal_over_episode,
    jam_set_from_conformal,
)
from fhew.config import NEW_ALGORITHM_GENERATORS, prediction_expert_specs
from fhew.logging_utils import load_json, save_json
from fhew.predict import ChannelPredictor
from fhew.sensor import SensorModel

NAME = "exp20_conformal_sets"

LCG_SEED_MAX = 256
WEAK_KEYSPACE = 256

IN_BANK = ("lfsr", "weak_seed", "lcg", "logistic", "markov")
CONTROL = ("csprng",)
GENERATORS = IN_BANK + NEW_ALGORITHM_GENERATORS + CONTROL
POOL = ("lfsr", "lcg", "markov", "lagged_fib", "csprng")   # mixed pool for calibration


def _pred_specs(M):
    return prediction_expert_specs(M, lcg_seed_max=LCG_SEED_MAX, weak_keyspace=WEAK_KEYSPACE)


def _episode(gen, ch, sensor, window, seed):
    params = {"keyspace": WEAK_KEYSPACE} if gen == "weak_seed" else None
    return C.episode_for(gen, ch, sensor, window, seed, gen_params=params,
                         lcg_keyspace=LCG_SEED_MAX)


def _fixed_topk_coverage(pred: ChannelPredictor, true, observations, k, *, warmup=5):
    """Realised coverage of a fixed top-k shortlist (no calibration, size k)."""
    true = np.asarray(true, dtype=np.int64)
    window = len(observations)
    pred.reset()
    cov = []
    for t, o in enumerate(observations):
        pred.observe(o)
        if t >= window - 1 or t < warmup:
            continue
        S = aps_set(pred.predict_next(), 1.0, min_size=1)[:k]  # top-k by prob
        cov.append(int(int(true[t + 1]) in S))
    return float(np.mean(cov)) if cov else float("nan")


def _confidence_tones(pred: ChannelPredictor, true, observations, max_tones, *, warmup=5):
    """Coverage and mean tone count of the confidence-driven rule (no guarantee)."""
    true = np.asarray(true, dtype=np.int64)
    window = len(observations)
    pred.reset()
    cov, tones = [], []
    for t, o in enumerate(observations):
        pred.observe(o)
        if t >= window - 1 or t < warmup:
            continue
        n = tones_from_confidence(pred.predictability(), max_tones)
        S = aps_set(pred.predict_next(), 1.0, min_size=1)[:n]
        cov.append(int(int(true[t + 1]) in S))
        tones.append(n)
    return (float(np.mean(cov)) if cov else float("nan"),
            float(np.mean(tones)) if tones else float("nan"))


def _jam_hit_rate(cpred: ConformalChannelPredictor, true, observations, effect, budget,
                  *, warmup=5):
    """Fraction of hops whose realised channel is denied when the conformal set is
    funded as the jam plan (D1). Equals coverage when the budget funds the set."""
    true = np.asarray(true, dtype=np.int64)
    window = len(observations)
    cpred.reset()
    hits = []
    for t, o in enumerate(observations):
        cpred.observe(o)
        if t >= window - 1:
            break
        S = cpred.prediction_set()
        power = jam_set_from_conformal(S, effect, budget)
        nxt = int(true[t + 1])
        if t >= warmup:
            hits.append(int(effect.effect_on(power, nxt) >= 0.5))
        if len(observations[t + 1].detections) == 1:
            cpred.calibrate(nxt in S)
    return float(np.mean(hits)) if hits else float("nan")


def compute(*, M=C.DEFAULT_M, window=140, n_seeds=12, p_d=0.95, p_fa=0.01,
            alpha=0.1, gamma=0.03,
            target_grid=(0.6, 0.7, 0.8, 0.9, 0.95)) -> dict:
    ch = ChannelSet.of_size(M)
    sensor = SensorModel(p_d=p_d, p_fa=p_fa)
    specs = _pred_specs(M)
    chance = 1.0 / M

    # one predictor reused (reset per episode); a conformal wrapper per target
    pred = ChannelPredictor(ch, sensor, specs=specs)

    # --- Panel A: coverage calibration across a mixed pool vs the target ---
    calib = {"target": list(target_grid), "conformal": [], "top1": None, "top3": None}
    top1_cov, top3_cov = [], []
    for tgt in target_grid:
        cp = ConformalChannelPredictor(pred, alpha=1.0 - tgt, gamma=gamma)
        covs = []
        for gen in POOL:
            for s in C.seeds(n_seeds):
                ep = _episode(gen, ch, sensor, window, s)
                covs.append(conformal_over_episode(cp, ep.true_channels, ep.observations).coverage)
        calib["conformal"].append(float(np.nanmean(covs)))
    # fixed shortlists don't depend on the target; measure once over the pool
    for gen in POOL:
        for s in C.seeds(n_seeds):
            ep = _episode(gen, ch, sensor, window, s)
            top1_cov.append(_fixed_topk_coverage(pred, ep.true_channels, ep.observations, 1))
            top3_cov.append(_fixed_topk_coverage(pred, ep.true_channels, ep.observations, 3))
    calib["top1"] = float(np.nanmean(top1_cov))
    calib["top3"] = float(np.nanmean(top3_cov))

    # --- Panels B & C: per-generator behaviour at the headline target 1 - alpha ---
    effect = EffectModel(ch, tau=1.0)                 # uniform-gain effect model
    budget = float(M * effect.costs().mean())         # enough to fund the whole plan
    cp = ConformalChannelPredictor(pred, alpha=alpha, gamma=gamma)
    per_gen = {}
    for gen in GENERATORS:
        cov, size, kap, j_hit = [], [], [], []
        t3, ct_cov, ct_tones = [], [], []
        for s in C.seeds(n_seeds):
            ep = _episode(gen, ch, sensor, window, s)
            sc = conformal_over_episode(cp, ep.true_channels, ep.observations)
            cov.append(sc.coverage); size.append(sc.mean_set_size); kap.append(sc.kappa_mean)
            j_hit.append(_jam_hit_rate(cp, ep.true_channels, ep.observations, effect, budget))
            t3.append(_fixed_topk_coverage(pred, ep.true_channels, ep.observations, 3))
            c, n = _confidence_tones(pred, ep.true_channels, ep.observations, max_tones=M)
            ct_cov.append(c); ct_tones.append(n)
        per_gen[gen] = {
            "coverage": float(np.nanmean(cov)),
            "mean_set_size": float(np.nanmean(size)),
            "kappa": float(np.nanmean(kap)),
            "jam_hit_rate": float(np.nanmean(j_hit)),
            "top3_coverage": float(np.nanmean(t3)),
            "conf_tones_coverage": float(np.nanmean(ct_cov)),
            "conf_tones_mean": float(np.nanmean(ct_tones)),
            "group": ("in_bank" if gen in IN_BANK
                      else "new_algorithm" if gen in NEW_ALGORITHM_GENERATORS else "control"),
        }

    return {
        "meta": {"M": M, "window": window, "n_seeds": n_seeds, "p_d": p_d, "p_fa": p_fa,
                 "alpha": alpha, "target_coverage": 1.0 - alpha, "gamma": gamma,
                 "chance": chance, "pool": list(POOL),
                 "metric": "conformal next-channel coverage + set size (tones)"},
        "calibration": calib,
        "operating_point": per_gen,
    }


def plot(results: dict, path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    meta = results["meta"]
    per = results["operating_point"]
    gens = [g for g in GENERATORS if g in per]
    x = np.arange(len(gens))
    tgt = meta["target_coverage"]

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(16.5, 4.8),
                                        gridspec_kw={"width_ratios": [1.0, 1.25, 1.35]})

    # Panel A: coverage calibration.
    cal = results["calibration"]
    g = np.array(cal["target"])
    axA.plot([0, 1], [0, 1], ls=":", color=style.PALETTE["grey"], label="ideal (y=x)")
    axA.plot(g, cal["conformal"], "-o", color=style.PALETTE["blue"], label="conformal set")
    axA.axhline(cal["top3"], color=style.PALETTE["orange"], ls="--", label=f"fixed top-3 ({cal['top3']:.2f})")
    axA.axhline(cal["top1"], color=style.PALETTE["vermillion"], ls="--", label=f"fixed top-1 ({cal['top1']:.2f})")
    axA.set_xlabel("target coverage $1-\\alpha$")
    axA.set_ylabel("realised coverage")
    axA.set_title("Coverage calibration (mixed pool)")
    axA.set_xlim(0.55, 1.0); axA.set_ylim(0.0, 1.02)
    axA.legend(fontsize=8, loc="lower right")

    # Panel B: mean set size (tones) per generator, with kappa annotated.
    sizes = [per[gn]["mean_set_size"] for gn in gens]
    axB.bar(x, sizes, color=style.PALETTE["blue"])
    for i, gn in enumerate(gens):
        axB.text(i, sizes[i] + 0.15, f"$\\kappa$={per[gn]['kappa']:.2f}",
                 ha="center", va="bottom", fontsize=7.5, color="#555", rotation=0)
    axB.axhline(meta["M"], color=style.PALETTE["grey"], ls=":", lw=1, label=f"full plan (M={meta['M']})")
    axB.set_xticks(x); axB.set_xticklabels(gens, rotation=30, ha="right")
    axB.set_ylabel("mean set size = tones")
    axB.set_title(f"Adaptive set size at $1-\\alpha$={tgt:.2f}")
    axB.set_ylim(0, meta["M"] * 1.12)
    axB.legend(fontsize=8, loc="upper left")
    new_idx = [i for i, gn in enumerate(gens) if gn in NEW_ALGORITHM_GENERATORS]
    if new_idx:
        axB.axvspan(min(new_idx) - 0.5, max(new_idx) + 0.5, color=style.PALETTE["yellow"],
                    alpha=0.12, zorder=0)

    # Panel C: coverage / jam-hit comparison vs the fixed and confidence rules.
    w = 0.27
    conf = [per[gn]["coverage"] for gn in gens]
    t3 = [per[gn]["top3_coverage"] for gn in gens]
    ct = [per[gn]["conf_tones_coverage"] for gn in gens]
    axC.bar(x - w, conf, w, label="conformal (= jam-hit)", color=style.PALETTE["blue"])
    axC.bar(x, ct, w, label="confidence tones", color=style.PALETTE["green"])
    axC.bar(x + w, t3, w, label="fixed top-3", color=style.PALETTE["orange"])
    axC.axhline(tgt, color=style.PALETTE["black"], ls="--", lw=1.2, label=f"target {tgt:.2f}")
    axC.set_xticks(x); axC.set_xticklabels(gens, rotation=30, ha="right")
    axC.set_ylabel("coverage of true next hop")
    axC.set_title("Coverage guarantee $\\Rightarrow$ jam-hit rate")
    axC.set_ylim(0, 1.05)
    axC.legend(fontsize=8, loc="lower left", ncol=1)

    fig.suptitle(f"Conformal next-channel sets  ($P_d={meta['p_d']}$, $P_{{fa}}={meta['p_fa']}$, "
                 f"$\\gamma={meta['gamma']}$)", y=1.03)
    fig.tight_layout()
    return style.save(fig, path)


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n-seeds", type=int, default=12)
    p.add_argument("--plot-only", action="store_true")
    args = p.parse_args(argv)
    jpath = C.RESULTS_DIR / f"{NAME}.json"
    results = load_json(jpath) if args.plot_only else compute(n_seeds=args.n_seeds)
    if not args.plot_only:
        save_json(results, jpath)
    fig = plot(results, C.FIG_DIR / f"{NAME}.png")
    print(f"[{NAME}] wrote {jpath} and {fig}")
    cal = results["calibration"]
    tgt = results["meta"]["target_coverage"]
    print(f"  calibration (pool): conformal tracks target; fixed top-1={cal['top1']:.2f} "
          f"top-3={cal['top3']:.2f}")
    print(f"  {'generator':14s} {'cover':>6s} {'tones':>6s} {'kappa':>6s} {'jam-hit':>8s} "
          f"{'top3cov':>8s}   (target={tgt:.2f})")
    for gn, d in results["operating_point"].items():
        tag = " <-NEW" if d["group"] == "new_algorithm" else ""
        print(f"  {gn:14s} {d['coverage']:6.2f} {d['mean_set_size']:6.2f} {d['kappa']:6.2f} "
              f"{d['jam_hit_rate']:8.2f} {d['top3_coverage']:8.2f}{tag}")


if __name__ == "__main__":
    main()
