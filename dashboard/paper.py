"""Condense the conference paper's saved results into the dashboard's ``paper`` block.

The Proof and How-it-works pages show the paper's latest evaluation. Its numbers
are already on disk as JSON under ``docs/paper/ieee/data`` (written by
``experiments/paper_extras.py`` and ``experiments/exp22``-``exp25``), so the
dashboard reads them instead of re-simulating. This module only uses the
standard library: ``python -m dashboard.build --reuse`` can refresh the block
without the simulator's dependencies.

Three results have no JSON in the repository (their ``results/*.json`` is not
committed): the identification converse (exp21) and the closed-loop adversary
(exp26 evasion, exp27 deception). Their numbers are copied from the methodology
write-up into :data:`DOCUMENTED`, each with the section it comes from, and the
pages label them as quoted.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

PAPER_DATA = Path(__file__).resolve().parent.parent / "docs" / "paper" / "ieee" / "data"

#: the paper's four sensor operating points (detection, false-alarm probability)
SENSORS = {
    "perfect": {"name": "Perfect", "p_d": 1.0, "p_fa": 0.0},
    "strong": {"name": "Strong", "p_d": 0.95, "p_fa": 0.01},
    "typical": {"name": "Typical", "p_d": 0.8, "p_fa": 0.02},
    "contested": {"name": "Contested", "p_d": 0.6, "p_fa": 0.05},
}

#: Results whose JSON is not in the repository, quoted from the write-ups.
DOCUMENTED = {
    "episodes": {
        "source": "Paper, Section IV (Experimental setup)",
        "total": 15690,
        "parts": [
            {"label": "Identification: 13 generators × 4 sensors × 100 seeds", "n": 5200},
            {"label": "Large-scale coverage", "n": 7500},
            {"label": "Conventional identifiers, same episodes", "n": 2000},
            {"label": "External predictors, same episodes", "n": 450},
            {"label": "Multi-tone response (hops denied)", "n": 540},
        ],
    },
    "converse": {
        "source": "Methodology write-up §5.20 (exp21), Markov source, M = 16, δ = 0.05",
        "p_d": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
        "measured": [31.3, 36.4, 43.9, 57.6, 74.6, 94.3],
        "bound": [11.2, 14.9, 21.2, 30.2, 46.4, 69.8],
        "drift": [0.264, 0.197, 0.139, 0.098, 0.063, 0.042],
    },
    "evasion": {
        "source": "Methodology write-up §7.2 (exp26), strong sensor, knapsack jammer",
        "rows": [
            {"id": "lfsr", "name": "LFSR", "group": "in_bank",
             "denied_passive": 1.00, "denied_evading": 0.17, "kappa_passive": 0.95, "kappa_evading": 0.02},
            {"id": "lagged_fib", "name": "Lagged generator", "group": "new_algorithm",
             "denied_passive": 0.99, "denied_evading": 0.14, "kappa_passive": 0.93, "kappa_evading": 0.01},
            {"id": "markov", "name": "Markov hopper", "group": "in_bank",
             "denied_passive": 0.41, "denied_evading": 0.12, "kappa_passive": 0.06, "kappa_evading": 0.01},
            {"id": "csprng", "name": "Keyed CSPRNG", "group": "control",
             "denied_passive": 0.13, "denied_evading": 0.13, "kappa_passive": 0.01, "kappa_evading": 0.01},
        ],
    },
    "deception": {
        "source": "Methodology write-up §7.3 (exp27), keyed CSPRNG behind an LFSR decoy",
        "sprt_false_id": 1.00,
        "bank_false_id_low": 0.03,
        "bank_unknown_low": 0.9,
        "low_persistence": 0.35,
        "defeats_h0_from": 0.7,
        "counter_jam_denied_before": 1.0,
        "counter_jam_denied_after": 0.1,
    },
}


def _r(x, nd: int = 4):
    """Round floats (recursively) so the block stays small; leave ints and None alone."""
    if isinstance(x, float):
        return round(x, nd)
    if isinstance(x, list):
        return [_r(v, nd) for v in x]
    if isinstance(x, dict):
        return {k: _r(v, nd) for k, v in x.items()}
    return x


def _load(name: str) -> dict | None:
    path = PAPER_DATA / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(v):
    return v["mean"] if isinstance(v, dict) and "mean" in v else v


def confusion(d: dict) -> dict:
    """Family confusion counts, accuracy, hard false-ID and hops per generator x sensor."""
    meta = d["meta"]
    fams, gens = meta["families"], meta["generators"]
    out = {"window": meta["window"], "n_seeds": meta["n_seeds"], "delta": meta["delta"],
           "families": fams, "generators": gens, "by_sensor": {}}
    for sid, s in d["sensors"].items():
        pg = s["per_generator"]
        out["by_sensor"][sid] = {
            "counts": [[s["counts"][g].get(f, 0) for f in fams] for g in gens],
            "target": [pg[g]["target"] for g in gens],
            "acc": [pg[g]["accuracy"] for g in gens],
            "hard": [pg[g]["hard_false_id"] for g in gens],
            "hops": [pg[g]["mean_hops"] for g in gens],
        }
    return _r(out)


def id_baselines(d: dict) -> dict:
    """The bank against three conventional identifiers on identical episodes (exp22)."""
    meta = d["meta"]
    out = {"window": meta["window"], "n_seeds": meta["n_seeds"], "n_episodes": meta["n_episodes"],
           "generators": meta["generators"], "fixed_length_n": meta["fixed_length_n"],
           "classifier_train": meta["classifier"]["n_train"], "by_sensor": {}}
    for sid, s in d["sensors"].items():
        methods = {}
        for mid, m in s["methods"].items():
            methods[mid] = {
                "acc": _mean(m["in_bank_accuracy"]), "unknown": _mean(m["csprng_unknown"]),
                "hard": _mean(m["hard_false_id"]), "hard_oob": _mean(m["hard_false_id_out_of_bank"]),
                "hops": _mean(m["mean_hops"]),
                "per_gen": {g: [v["accuracy"], v["hard_false_id"]] for g, v in m["per_generator"].items()},
            }
        out["by_sensor"][sid] = {"methods": methods}
    return _r(out)


def predict(d: dict) -> dict:
    """Steady-state next-channel accuracy, proposed predictor vs identification-only bank."""
    out = {"by_sensor": {}}
    for sid, s in d["sensors"].items():
        meta = s["meta"]
        out.update(window=meta["window"], n_seeds=meta["n_seeds"], chance=meta["chance"])
        out["by_sensor"][sid] = {g: {"steady": v["proposed_steady"], "id_only": v["id_only_steady"],
                                     "top1": v["proposed_top1"], "top3": v["proposed_top3"],
                                     "markov": v["markov_top1"], "group": v["group"]}
                                 for g, v in s["operating_point"].items()}
    return _r(out)


def pred_baselines(d: dict) -> dict:
    """External predictors on the same episodes: Markov-1, PPM-C and a trained LSTM (exp23)."""
    meta = d["meta"]
    out = {"window": meta["window"], "n_seeds": meta["n_seeds"], "n_episodes": meta["n_episodes"],
           "chance": meta["chance"], "methods": meta["methods"], "train_families": meta["train_families"],
           "by_sensor": {}}
    for sid, s in d["sensors"].items():
        out["by_sensor"][sid] = {
            "lstm_train": s["lstm"]["n_train_sequences"],
            "gens": {g: {m: _mean(v) for m, v in ms.items()} for g, ms in s["generators"].items()},
        }
    return _r(out)


def conformal(d: dict) -> dict:
    """Conformal next-channel sets: calibration curve and per-generator coverage / set size."""
    out = {"by_sensor": {}}
    for sid, s in d["sensors"].items():
        meta = s["meta"]
        out.update(window=meta["window"], n_seeds=meta["n_seeds"], target=meta["target_coverage"])
        out["by_sensor"][sid] = {
            "calib": s["calibration"],
            "op": {g: {"cov": v["coverage"], "size": v["mean_set_size"], "kappa": v["kappa"],
                       "top3_cov": v["top3_coverage"], "group": v["group"]}
                   for g, v in s["operating_point"].items()},
        }
    return _r(out)


def denial(d: dict) -> dict:
    """Fraction of hops a conformal multi-tone response denies, against the tone budget (exp24)."""
    meta = d["meta"]
    out = {"window": meta["window"], "n_seeds": meta["n_seeds"], "n_episodes": meta["n_episodes"],
           "budgets": meta["budgets"], "target": meta["target"], "generators": meta["generators"],
           "by_sensor": {}}
    for sid, s in d["sensors"].items():
        out["by_sensor"][sid] = {g: {"denied": v["denied_vs_budget"], "tones": v["tones_vs_budget"],
                                     "topk": v["fixed_topk_denied"], "unlimited": v["denied_unlimited"],
                                     "mean_tones": v["mean_tones"]}
                                 for g, v in s["generators"].items()}
    return _r(out, 3)


def runtime(d: dict) -> dict:
    """Median and 95th-percentile per-slot time of each component (exp25)."""
    meta = d["meta"]
    out = {"cpu": meta["cpu"], "threads": meta["threads"], "p_d": meta["p_d"], "p_fa": meta["p_fa"],
           "by_M": {}}
    for M, v in d["by_M"].items():
        out["by_M"][M] = {
            **{k: {"med": x["median_ms"], "p95": x["p95_ms"]} for k, x in v.items()
               if isinstance(x, dict) and "median_ms" in x},
            "build_s": v["build_seconds"],
            "experts": {k: {"med": x["median_ms"], "n": x.get("candidates")} for k, x in v["experts"].items()},
        }
    return _r(out)


def switch(d: dict) -> dict:
    """Mid-engagement change of algorithm: detection, delay, dangerous locks, recovery (Table VI)."""
    meta = d["meta"]
    out = {"window": meta["window"], "n_plans": meta["n_plans"],
           "regime_hops": meta.get("regime_hops"), "by_sensor": {}}
    keep = ("n_switches", "n_regimes", "detect_rate", "delay_median", "delay_p90", "dated_within_1",
            "false_alarms_per_100_hops", "top1_since_switch", "dangerous_locks_per_regime",
            "n_dangerous", "lock_hops_median", "final_exact", "final_soft", "final_wrong")
    for sid, s in d["sensors"].items():
        out["by_sensor"][sid] = {k: s[k] for k in keep if k in s}
    return _r(out, 3)


def mismatch(d: dict) -> dict:
    """Identification when the assumed sensor differs from the true one."""
    meta = d["meta"]
    cell = lambda c: [c["accuracy"], c["hard_false_id"]]
    return _r({"window": meta["window"], "n_seeds": meta["n_seeds"], "generators": meta["generators"],
               "p_d_grid": meta["p_d_grid"], "p_fa_grid": meta["p_fa_grid"],
               "p_fa_fixed": meta["p_fa_fixed"], "p_d_fixed": meta["p_d_fixed"],
               "n_per_cell": d["pd"][0][0]["n"],
               "pd": [[cell(c) for c in row] for row in d["pd"]],
               "pfa": [[cell(c) for c in row] for row in d["pfa"]]})


_FILES = {
    "confusion": ("extra_confusion.json", confusion),
    "id_baselines": ("exp22_id_baselines.json", id_baselines),
    "predict": ("extra_predict.json", predict),
    "pred_baselines": ("exp23_prediction_baselines.json", pred_baselines),
    "conformal": ("extra_conformal.json", conformal),
    "denial": ("exp24_denial.json", denial),
    "runtime": ("exp25_runtime.json", runtime),
    "switch": ("extra_switch.json", switch),
    "mismatch": ("extra_mismatch.json", mismatch),
}


def load() -> dict:
    """The ``paper`` block: every saved result present on disk, plus :data:`DOCUMENTED`."""
    block = {"refreshed": dt.date.today().isoformat(), "source": "docs/paper/ieee/data",
             "sensors": SENSORS, "files": {}}
    for key, (name, fn) in _FILES.items():
        raw = _load(name)
        if raw is None:
            continue
        block[key] = fn(raw)
        block["files"][key] = name
    block["documented"] = DOCUMENTED
    return block


if __name__ == "__main__":  # pragma: no cover - quick look at the block size
    b = load()
    print(f"paper block: {len(json.dumps(b, separators=(',', ':'))):,} bytes, "
          f"sections: {', '.join(k for k in b if k not in ('refreshed', 'source', 'sensors', 'files'))}")
