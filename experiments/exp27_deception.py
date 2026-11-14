"""Experiment 27: counter-reconnaissance deception -- fooling the OTS on purpose.

``exp4`` showed the mandatory none-of-the-above class ``H_0`` keeps the bank from
naming a generator when the emitter is merely *out of bank* (a passive failure of
the attacker's). This experiment makes the failure *active*: the adversary plants
detections that are time-coherent with an in-bank generator (:mod:`fhew.deception`),
so a single-emitter collector is actively pushed toward the wrong answer and toward
spending jam power on channels no one is using.

Two panels:

* **A -- counter-identification.** Real emitter is a keyed CSPRNG (truth: unknown);
  the adversary overlays an LFSR decoy whose *persistence* ``p_present`` we sweep.
  We plot the induced hard false-identification rate for the bank **with** ``H_0``
  against a pairwise SPRT **without** ``H_0`` (exactly ``exp4``'s contrast, now under
  attack). ``H_0`` buys a graceful "unknown" that defeats an intermittent decoy; the
  no-``H_0`` test is forced to name the decoy whatever its persistence. The honest
  limit is visible too: a near-always-on, perfectly coherent decoy embeds a genuine
  LFSR and defeats even ``H_0`` -- the motivation for emitter-ID sorting (journal).

* **B -- counter-jam.** In the closed loop (:mod:`fhew.duel`) against a predictable
  emitter the OTS would otherwise deny completely, we sweep the number of coherent
  decoys and measure the fraction of hops still denied and the tones spent. Each
  decoy splits the predictive jammer's belief; a handful drive its denial down toward
  the blind barrage floor while its tone count climbs -- power wasted on ghosts.

The hidden truth is used only to score; the OTS sees only (corrupted) observations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments import common as C
from experiments import style
from fhew.allocator import EffectModel
from fhew.baselines import PairwiseSPRT
from fhew.channels import ChannelSet
from fhew.deception import DeceptionConfig, DeceptionEmitter, make_deception_episode
from fhew.duel import OTSJammer, run_duel
from fhew.evasion import NO_EVASION_CFG, AdaptiveEvasionGenerator
from fhew.experts import build_hypothesis
from fhew.generators import make_generator
from fhew.likelihood import SensorAwareEmission
from fhew.logging_utils import load_json, save_json
from fhew.sensor import SensorModel
from fhew.sim import run_bank, run_sprt

NAME = "exp27_deception"

DECOY_FAMILY = "lfsr"
DECOY_PARAMS = {"length": 7}


def counter_id(ch, sensor, *, window, n_seeds, persistence) -> dict:
    """Induced false-ID vs decoy persistence: bank (H_0) vs pairwise SPRT (no H_0)."""
    specs = C.default_specs(ch.size)
    tgt = C.default_target_labels(specs)
    emit = SensorAwareEmission(sensor)
    rows = []
    for pp in persistence:
        bank = C.make_bank(ch, window, sensor, specs)
        sprt = PairwiseSPRT([build_hypothesis(s, ch, window) for s in specs], emit, ch.size)
        bank_hard, bank_unknown, sprt_false = [], [], []
        for s in C.seeds(n_seeds):
            base = C.episode_for("csprng", ch, sensor, window, s, target_labels=tgt)
            if pp is None:                                   # the no-deception control
                ep = base
            else:
                ep = make_deception_episode(
                    base, DECOY_FAMILY, decoy_params=DECOY_PARAMS,
                    config=DeceptionConfig(n_decoys=1, p_present=pp))
            rb, rs = run_bank(bank, ep), run_sprt(sprt, ep)
            bank_hard.append(C.is_hard_false_id(rb))
            bank_unknown.append(rb.decision_label == "unknown")
            sprt_false.append(rs.false_id)
        rows.append({"p_present": pp,
                     "bank_hard_false_id": float(np.mean(bank_hard)),
                     "bank_unknown": float(np.mean(bank_unknown)),
                     "sprt_false_id": float(np.mean(sprt_false))})
    return {"rows": rows}


def counter_jam(ch, sensor, *, window, n_seeds, n_decoy_grid, budget, max_tones,
                families) -> dict:
    """Denial + tones vs number of coherent decoys, in the closed loop."""
    effect = EffectModel(ch, tau=1.0)
    # One jammer (one expensive predictor), reused and reset across every duel.
    jammer = OTSJammer(ch, sensor, effect, budget=budget, plan="knapsack", max_tones=max_tones)
    series = {}
    for fam, fam_params in families.items():
        den, tones, top1 = [], [], []
        for nd in n_decoy_grid:
            agg = []
            for s in C.seeds(n_seeds):
                rng = np.random.default_rng(s)
                emitter = AdaptiveEvasionGenerator.from_family(
                    fam, ch, rng, base_params=fam_params, config=NO_EVASION_CFG)
                dec = None
                if nd > 0:
                    dg = make_generator(DECOY_FAMILY, ch.size, rng=rng, **DECOY_PARAMS)
                    dec = DeceptionEmitter(dg, ch, DeceptionConfig(n_decoys=nd),
                                           decoy_family=DECOY_FAMILY, decoy_params=DECOY_PARAMS,
                                           rng=rng)
                agg.append(run_duel(emitter, jammer, sensor, ch, rng,
                                    window=window, deception=dec).aggregate())
            den.append(float(np.mean([a["denied_frac"] for a in agg])))
            tones.append(float(np.mean([a["mean_tones"] for a in agg])))
            top1.append(float(np.mean([a["nexthop_top1"] for a in agg])))
        series[fam] = {"denied_frac": den, "mean_tones": tones, "nexthop_top1": top1}
    return {"n_decoys": list(n_decoy_grid), "series": series}


def compute(*, M=C.DEFAULT_M, window=120, n_seeds=25, p_d=0.85, p_fa=0.02,
            persistence=(None, 1.0, 0.8, 0.6, 0.5, 0.35, 0.2),
            n_decoy_grid=(0, 1, 2, 3, 4), budget=3.0, max_tones=5,
            jam_families=None, **_) -> dict:
    ch = ChannelSet.of_size(M)
    sensor = SensorModel(p_d=p_d, p_fa=p_fa)
    jam_families = jam_families or {"lfsr": {"length": 7}, "lagged_fib": {}}
    cid = counter_id(ch, sensor, window=window, n_seeds=n_seeds, persistence=persistence)
    cjam = counter_jam(ch, sensor, window=window, n_seeds=n_seeds, n_decoy_grid=n_decoy_grid,
                       budget=budget, max_tones=max_tones, families=jam_families)
    return {"meta": {"M": M, "window": window, "n_seeds": n_seeds, "p_d": p_d, "p_fa": p_fa,
                     "decoy_family": DECOY_FAMILY, "budget": budget, "max_tones": max_tones},
            "counter_id": cid, "counter_jam": cjam}


GEN_LABELS = {"lfsr": "LFSR (in-bank)", "lagged_fib": "lagged-Fib. (held out)"}


def plot(results: dict, path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.6, 4.6))

    # Panel A: counter-ID vs persistence.
    rows = results["counter_id"]["rows"]
    xs = [r["p_present"] for r in rows]
    # the control (None) is drawn as a left-hand reference tick at x = -0.1
    def xval(pp):  # noqa: E306
        return -0.12 if pp is None else pp
    xp = [xval(pp) for pp in xs]
    axA.plot(xp, [r["bank_hard_false_id"] for r in rows], "-o",
             color=style.PALETTE["blue"], label="bank with $H_0$: claims a break")
    axA.plot(xp, [r["bank_unknown"] for r in rows], "-^",
             color=style.PALETTE["green"], label="bank with $H_0$: answers unknown")
    axA.plot(xp, [r["sprt_false_id"] for r in rows], "-s",
             color=style.PALETTE["vermillion"], label="pairwise SPRT (no $H_0$): false-ID")
    axA.axvline(0.0, color="#bbb", lw=0.8, ls=":")
    axA.text(-0.12, 1.07, "no\ndecoy", ha="center", fontsize=7, color="#555")
    axA.set_xlabel("decoy persistence $p_{\\mathrm{present}}$")
    axA.set_ylabel("rate over episodes"); axA.set_ylim(0, 1.15)
    axA.set_title("A. Counter-ID: $H_0$ routes intermittent deception to unknown")
    axA.legend(fontsize=7.5, loc="center left")

    # Panel B: counter-jam vs number of decoys.
    nd = results["counter_jam"]["n_decoys"]
    for i, (fam, d) in enumerate(results["counter_jam"]["series"].items()):
        col = style.SERIES_COLORS[i % len(style.SERIES_COLORS)]
        axB.plot(nd, d["denied_frac"], "-o", color=col, label=f"{GEN_LABELS.get(fam, fam)} denied")
        axB.plot(nd, d["mean_tones"], "--s", color=col, ms=4, alpha=0.6,
                 label=f"{GEN_LABELS.get(fam, fam)} tones")
    floor = 1.0 / results["meta"]["M"]
    axB.axhline(floor, color="#555", ls=":", lw=1, label=f"blind 1-tone floor (1/M={floor:.2f})")
    axB.set_xlabel("number of coherent decoys")
    axB.set_ylabel("fraction denied  /  tones lit")
    axB.set_title("B. Counter-jam: decoys destroy the prediction advantage")
    axB.legend(fontsize=7.5, loc="center right")

    fig.tight_layout()
    return style.save(fig, path)


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n-seeds", type=int, default=25)
    p.add_argument("--window", type=int, default=120)
    p.add_argument("--plot-only", action="store_true")
    args = p.parse_args(argv)
    jpath = C.RESULTS_DIR / f"{NAME}.json"
    results = load_json(jpath) if args.plot_only else compute(n_seeds=args.n_seeds, window=args.window)
    if not args.plot_only:
        save_json(results, jpath)
    fig = plot(results, C.FIG_DIR / f"{NAME}.png")
    print(f"[{NAME}] wrote {jpath} and {fig}")
    print("  Counter-ID (real=csprng, decoy=LFSR):")
    for r in results["counter_id"]["rows"]:
        tag = "no-decoy" if r["p_present"] is None else f"p={r['p_present']:.2f}"
        print(f"    {tag:9s}: bank hard-false-ID={r['bank_hard_false_id']:.2f} "
              f"unknown={r['bank_unknown']:.2f} | SPRT false-ID={r['sprt_false_id']:.2f}")
    print("  Counter-jam (denied / tones vs #decoys):")
    for fam, d in results["counter_jam"]["series"].items():
        den = " ".join(f"{v:.2f}" for v in d["denied_frac"])
        ton = " ".join(f"{v:.1f}" for v in d["mean_tones"])
        print(f"    {fam:11s} n={results['counter_jam']['n_decoys']}  denied=[{den}]  tones=[{ton}]")


if __name__ == "__main__":
    main()
