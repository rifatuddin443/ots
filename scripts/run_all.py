"""Run every conference experiment end to end and write a summary.

Usage:
    python -m scripts.run_all                 # all experiments, default seeds
    python -m scripts.run_all --quick         # fewer seeds, for a fast smoke run
    python -m scripts.run_all --only exp1 exp3

Figures land in ``results/figures/``, per-experiment JSON in ``results/``, and a
consolidated ``results/summary.json`` with the headline numbers. Every figure
regenerates from its JSON via ``--plot-only`` on the individual script, so a
re-render never re-simulates.
"""

from __future__ import annotations

import argparse
import time

from experiments import (
    exp1_hops_vs_pd,
    exp2_accuracy_vs_pfa,
    exp3_method_comparison,
    exp4_none_of_the_above,
    exp5_stale_bank,
    exp6_bm_vs_filtered,
    exp7_sensor_ablation,
    exp8_nexthop,
    exp9_eprocess_bound,
    exp10_calibration,
    exp11_stopping_law,
    exp12_full_bank_confusion,
    exp13_coverage,
    exp14_next_channel_prediction,
)
from experiments import common as C
from fhew.channels import ChannelSet
from fhew.datagen import grounding_report, train_classifier
from fhew.logging_utils import load_json, save_json

EXPERIMENTS = {
    "exp1": exp1_hops_vs_pd,
    "exp2": exp2_accuracy_vs_pfa,
    "exp3": exp3_method_comparison,
    "exp4": exp4_none_of_the_above,
    "exp5": exp5_stale_bank,
    "exp6": exp6_bm_vs_filtered,
    "exp7": exp7_sensor_ablation,
    "exp8": exp8_nexthop,
    "exp9": exp9_eprocess_bound,
    "exp10": exp10_calibration,
    "exp11": exp11_stopping_law,
    "exp12": exp12_full_bank_confusion,
    "exp13": exp13_coverage,
    "exp14": exp14_next_channel_prediction,
}

# The comprehensive full-bank experiments get their own (larger) seed floors so
# the "huge, varied, all-types" dataset stays big even on a default run.
_SEED_FLOOR = {"exp12": 40, "exp13": 50}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Run all FH-ID conference experiments.")
    p.add_argument("--quick", action="store_true", help="fewer seeds for a fast run")
    p.add_argument("--only", nargs="*", default=None, help="subset, e.g. exp1 exp3")
    p.add_argument("--n-seeds", type=int, default=None, help="override seed count")
    args = p.parse_args(argv)

    n_seeds = args.n_seeds if args.n_seeds is not None else (8 if args.quick else 30)
    chosen = args.only or list(EXPERIMENTS)

    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Train / refresh the learned classifier the method comparison needs.
    if "exp3" in chosen:
        ch = ChannelSet.of_size(C.DEFAULT_M)
        clf, info = train_classifier(ch, n_per_class=80 if args.quick else 250)
        save_json(clf.to_dict(), "data/learned_classifier.json")
        save_json(info, "data/learned_classifier_info.json")
        print(f"[classifier] train_acc={info['train_accuracy']:.3f} "
              f"test_acc={info.get('test_accuracy', float('nan')):.3f}")

    save_json(grounding_report(), C.RESULTS_DIR / "grounding.json")

    summary = {}
    for name in chosen:
        mod = EXPERIMENTS[name]
        t0 = time.time()
        seeds = n_seeds if args.quick else max(n_seeds, _SEED_FLOOR.get(name, 0))
        results = mod.compute(n_seeds=seeds)
        save_json(results, C.RESULTS_DIR / f"{mod.NAME}.json")
        mod.plot(results, C.FIG_DIR / f"{mod.NAME}.png")
        summary[mod.NAME] = _headline(name, results)
        print(f"[{mod.NAME}] done in {time.time() - t0:.1f}s")

    save_json(summary, C.RESULTS_DIR / "summary.json")
    print(f"\nWrote results/ and {len(chosen)} figures. Summary: results/summary.json")


def _headline(name: str, r: dict) -> dict:
    """Pull one or two headline numbers per experiment for the summary file."""
    try:
        if name == "exp1":
            return {gen: {"hops@Pd1.0": rows[0]["mean_hops_to_identify"],
                          "acc@Pd0.5": rows[-1]["accuracy"]}
                    for gen, rows in r["series"].items()}
        if name == "exp3":
            return {m: {"accuracy": s["accuracy"], "false_id": s["false_id_rate"]}
                    for m, s in r["summaries"].items()}
        if name == "exp4":
            return {g: {"bank_hard_false_id": d["bank"]["hard_false_id_rate"],
                        "sprt_false_id": d["pairwise_sprt"]["false_id_rate"]}
                    for g, d in r["generators"].items()}
        if name == "exp6":
            return {"bm@Pd0.9": r["bm"][1]["recovery_rate"],
                    "state_filtered@Pd0.9": r["state_filtered"][1]["recovery_rate"]}
        if name == "exp9":
            return {f"beta={row['beta']}": row["empirical_false_id"] for row in r["rows"]}
        if name == "exp10":
            return {"ece": r["ece"]}
        if name == "exp11":
            return {"r2_log_inv_delta": r["panelA_delta"]["r2"],
                    "r2_inv_pd": r["panelB_pd"]["r2"]}
        if name == "exp12":
            return r["per_generator_family_accuracy"]
        if name == "exp13":
            return {"n_episodes": r["n_episodes"],
                    **{g: {"acc": a["accuracy"], "hard_false_id": a["hard_false_id_rate"]}
                       for g, a in r["aggregate_by_type"].items()}}
        if name == "exp14":
            per = r["operating_point"]
            return {g: {"proposed_top1": d["proposed_steady"],
                        "id_only_top1": d["id_only_steady"]}
                    for g, d in per.items() if d["group"] == "new_algorithm"}
    except Exception as exc:  # pragma: no cover
        return {"note": f"headline extraction failed: {exc}"}
    return {}


if __name__ == "__main__":
    main()
