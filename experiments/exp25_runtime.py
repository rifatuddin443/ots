"""Experiment 25: per-slot compute time.

An online system must finish its per-slot work inside one hop dwell. This
benchmark times, single-threaded and per slot, every stage of the engagement
loop on the lossy stream:

* the identification bank update (all experts + ``H_0`` + the stopping test),
  with a per-expert breakdown and each expert's candidate count;
* the full proposed predictor (bank experts + universal and rule-search
  experts, Fixed-Share update and forecast);
* the conformal set and tone selection;
* the switch tracker (run-length posterior over anchored banks);

plus the one-off cost of building the experts' trajectory tables. Times are
wall-clock medians and 95th percentiles over slots of a generator mix, at the
typical sensor, for ``M = 16`` and ``M = 64`` channels. They are for this
unoptimised NumPy implementation on one CPU core; the number that matters is
the margin against a hop dwell, not the absolute figure.
"""

from __future__ import annotations

import platform
import time
from pathlib import Path

import numpy as np

from experiments import common as C
from experiments import style
from experiments import exp14_next_channel_prediction as E14
from fhew.channels import ChannelSet
from fhew.conformal import ConformalChannelPredictor
from fhew.experts import H0Hypothesis, build_hypothesis
from fhew.likelihood import SensorAwareEmission
from fhew.logging_utils import load_json, save_json
from fhew.predict import ChannelPredictor
from fhew.sensor import SensorModel
from fhew.tracker import ExpertLibrary, SwitchTracker

NAME = "exp25_runtime"

GENERATORS = ("lfsr", "lcg", "markov", "lagged_fib", "csprng")
SENSOR = (0.8, 0.02)


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _stats(ms) -> dict:
    a = np.asarray(ms, dtype=np.float64)
    return {"median_ms": float(np.median(a)), "p95_ms": float(np.percentile(a, 95)),
            "mean_ms": float(a.mean()), "n_slots": int(a.size)}


def _episodes(ch, sensor, window, n_seeds):
    return [C.episode_for(g, ch, sensor, window, s)
            for g in GENERATORS for s in C.seeds(n_seeds)]


def _time_bank(ch, sensor, window, episodes) -> tuple[dict, float]:
    t0 = time.perf_counter()
    bank = C.make_bank(ch, window, sensor, C.id_specs(ch.size))
    build_s = time.perf_counter() - t0
    ms = []
    for ep in episodes:
        bank.reset()
        for o in ep.observations:
            t = time.perf_counter()
            bank.update(o)
            bank.predict()
            ms.append(1e3 * (time.perf_counter() - t))
    return _stats(ms), build_s


def _time_experts(ch, sensor, window, episodes) -> dict:
    """Per-expert cost of one slot: score, ingest the clean channel, forecast."""
    emit = SensorAwareEmission(sensor)
    specs = E14._pred_specs(ch.size)
    hyps = [H0Hypothesis(ch, mode="uniform", floor="uniform")]
    hyps += [build_hypothesis(s, ch, window + 8) for s in specs]
    labels = ["H0"] + [s.label for s in specs]
    ms = {lab: [] for lab in labels}
    for ep in episodes:
        for h in hyps:
            h.reset()
        for o in ep.observations:
            log_e = emit.log_emission(o)
            clean = o.detections[0].channel if len(o.detections) == 1 else None
            for lab, h in zip(labels, hyps):
                t = time.perf_counter()
                h.step(log_e)
                oc = getattr(h, "observe_clean", None)
                if oc is not None:
                    oc(clean)
                h.predict_next()
                ms[lab].append(1e3 * (time.perf_counter() - t))
    return {lab: {**_stats(ms[lab]), "candidates": int(h.n_particles)}
            for lab, h in zip(labels, hyps)}


def _time_predictor(ch, sensor, window, episodes) -> tuple[dict, dict]:
    pred = ChannelPredictor(ch, sensor, specs=E14._pred_specs(ch.size))
    cpred = ConformalChannelPredictor(pred, alpha=0.1, gamma=0.03)
    ms, ms_set = [], []
    for ep in episodes:
        cpred.reset()
        for o in ep.observations:
            t = time.perf_counter()
            cpred.observe(o)
            pred.predict_next()
            t1 = time.perf_counter()
            tones = cpred.prediction_set()          # the multi-tone plan
            cpred.calibrate(bool(tones))
            t2 = time.perf_counter()
            ms.append(1e3 * (t1 - t))
            ms_set.append(1e3 * (t2 - t1))
    return _stats(ms), _stats(ms_set)


def _time_tracker(ch, sensor, window, episodes) -> tuple[dict, float]:
    t0 = time.perf_counter()
    library = ExpertLibrary.build(ch, window + 2)
    build_s = time.perf_counter() - t0
    ms = []
    for ep in episodes:
        tracker = SwitchTracker(ch, sensor, library=library)
        for o in ep.observations:
            t = time.perf_counter()
            tracker.observe(o)
            tracker.predict_top1()
            ms.append(1e3 * (time.perf_counter() - t))
    return _stats(ms), build_s


def compute(*, window=120, n_seeds=3, channel_counts=(16, 64), **_) -> dict:
    n = max(1, min(int(n_seeds), 3))            # timing needs slots, not seeds
    out = {"meta": {"window": window, "n_seeds": n, "generators": list(GENERATORS),
                    "p_d": SENSOR[0], "p_fa": SENSOR[1], "cpu": _cpu_model(),
                    "threads": 1, "python": platform.python_version(),
                    "numpy": np.__version__},
           "by_M": {}}
    for M in channel_counts:
        ch = ChannelSet.of_size(M)
        sensor = SensorModel(p_d=SENSOR[0], p_fa=SENSOR[1])
        eps = _episodes(ch, sensor, window, n)
        bank, bank_build = _time_bank(ch, sensor, window, eps)
        predictor, cset = _time_predictor(ch, sensor, window, eps)
        tracker, lib_build = _time_tracker(ch, sensor, window, eps[::n])
        out["by_M"][str(M)] = {
            "bank": bank, "predictor": predictor, "conformal_set": cset, "tracker": tracker,
            "build_seconds": {"bank": bank_build, "tracker_library": lib_build},
            "experts": _time_experts(ch, sensor, window, eps),
        }
    return out


def plot(results: dict, path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    Ms = list(results["by_M"])
    stages = [("bank", "identification\nbank"), ("predictor", "predictor\n(bank + universal)"),
              ("conformal_set", "conformal set\n+ tones"), ("tracker", "switch\ntracker")]
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 4.0))
    x = np.arange(len(stages))
    w = 0.8 / len(Ms)
    for i, M in enumerate(Ms):
        med = [results["by_M"][M][k]["median_ms"] for k, _ in stages]
        p95 = [results["by_M"][M][k]["p95_ms"] for k, _ in stages]
        axA.bar(x + (i - (len(Ms) - 1) / 2) * w, med, w, label=f"$M$={M} (median)",
                color=style.SERIES_COLORS[i])
        axA.plot(x + (i - (len(Ms) - 1) / 2) * w, p95, "_", ms=14, color="k")
    axA.set_yscale("log")
    axA.set_xticks(x, [lab for _, lab in stages], fontsize=8.5)
    axA.set_ylabel("ms per slot (tick: 95th pct)")
    axA.set_title("Per-slot compute time by stage")
    axA.legend(fontsize=8.5)
    ex = results["by_M"][Ms[0]]["experts"]
    labs = sorted(ex, key=lambda k: ex[k]["median_ms"])
    axB.barh(labs, [ex[k]["median_ms"] for k in labs], color=style.PALETTE["blue"])
    for i, k in enumerate(labs):
        axB.text(ex[k]["median_ms"], i, f"  {ex[k]['candidates']} cand.", va="center", fontsize=8)
    axB.set_xlabel("ms per slot (median)")
    axB.set_title(f"Per-expert cost, $M$={Ms[0]}")
    fig.tight_layout()
    return style.save(fig, path)


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--plot-only", action="store_true")
    args = p.parse_args(argv)
    jpath = C.RESULTS_DIR / f"{NAME}.json"
    results = load_json(jpath) if args.plot_only else compute(n_seeds=args.n_seeds)
    if not args.plot_only:
        save_json(results, jpath)
    fig = plot(results, C.FIG_DIR / f"{NAME}.png")
    m = results["meta"]
    print(f"[{NAME}] wrote {jpath} and {fig}  ({m['cpu']}, 1 thread)")
    for M, d in results["by_M"].items():
        print(f"  M={M}:")
        for k in ("bank", "predictor", "conformal_set", "tracker"):
            print(f"    {k:14s} median {d[k]['median_ms']:8.3f} ms  p95 {d[k]['p95_ms']:8.3f} ms")
        print(f"    build: bank {d['build_seconds']['bank']:.1f} s, "
              f"tracker library {d['build_seconds']['tracker_library']:.1f} s")
        for lab, e in sorted(d["experts"].items(), key=lambda kv: -kv[1]["median_ms"]):
            print(f"      {lab:15s} {e['median_ms']:7.3f} ms  ({e['candidates']} candidates)")


if __name__ == "__main__":
    main()
