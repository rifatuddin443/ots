"""Build the conference paper's added figure and data from saved results.

Reads ``results/exp22..exp25*.json`` (run the experiments first), copies them
to ``docs/paper/ieee/data/`` so the paper's numbers are versioned, renders the
column-width response figure ``docs/paper/ieee/figures/fig_denial.png``, and
prints the rows of the three added tables.

Usage:
    python -m experiments.exp22_id_baselines
    python -m experiments.exp23_prediction_baselines
    python -m experiments.exp24_denial
    python -m experiments.exp25_runtime
    python -m scripts.paper_additions
"""

from __future__ import annotations

import shutil
from pathlib import Path

from experiments import common as C
from fhew.logging_utils import load_json

PAPER = Path("docs/paper/ieee")
NAMES = ("exp22_id_baselines", "exp23_prediction_baselines", "exp24_denial", "exp25_runtime")
GEN_LABELS = {"lfsr": "LFSR", "lcg": "LCG", "markov": "Markov", "lagged_fib": "lagged-Fib.",
              "sine_map": "sine map", "csprng": "CSPRNG"}


def fig_denial(results: dict, path: Path, sensor: str = "typical") -> Path:
    """Fraction of hops denied vs tone budget, one curve per generator."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    S = results["sensors"][sensor]
    B = results["meta"]["budgets"]
    target = results["meta"]["target"]
    colors = ["#0072B2", "#56B4E9", "#D55E00", "#009E73", "#E69F00", "#555555"]
    marks = ["o", "s", "^", "D", "v", "x"]
    with plt.rc_context({"font.size": 7.5, "axes.linewidth": 0.6, "font.family": "DejaVu Sans"}):
        fig, ax = plt.subplots(figsize=(3.45, 2.25))
        for g, col, mk in zip(results["meta"]["generators"], colors, marks):
            d = S["generators"][g]
            ax.plot(B, d["denied_vs_budget"], "-", marker=mk, ms=2.6, lw=1.0, color=col,
                    label=f"{GEN_LABELS[g]} ({d['mean_tones']:.1f})")
        ax.axhline(target, color="#333", ls=":", lw=0.8)
        ax.text(1.0, target + 0.02, f"target {target:g}", fontsize=6.5, color="#333")
        ax.set_xlabel("tone budget $B$ (channels funded per hop, at most)")
        ax.set_ylabel("fraction of hops denied")
        ax.set_xticks([1, 2, 4, 6, 8, 10, 12, 14, 16])
        ax.set_xlim(0.5, 16.5)
        ax.set_ylim(0, 1.05)
        ax.grid(True, lw=0.3, alpha=0.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(fontsize=6.3, loc="lower right", frameon=True, framealpha=0.9,
                  title="generator (mean tones used)", title_fontsize=6.3,
                  handlelength=1.6, borderpad=0.35, labelspacing=0.25)
        fig.tight_layout(pad=0.3)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=600)
        plt.close(fig)
    return path


def _table_id(r: dict) -> None:
    print("\nTable: identification baselines (in-bank acc / CSPRNG->unknown / hard false-ID "
          "all / out-of-bank / hops)")
    for name, S in r["sensors"].items():
        for m, d in S["methods"].items():
            print(f"  {name:10s} {m:14s} {d['in_bank_accuracy']['mean']:.2f} "
                  f"{d['csprng_unknown']['mean']:.2f} {d['hard_false_id']['mean']:.3f} "
                  f"{d['hard_false_id_out_of_bank']['mean']:.3f} {d['mean_hops']:.1f}")


def _table_pred(r: dict) -> None:
    print("\nTable: prediction baselines (steady-state top-1)")
    methods = r["meta"]["methods"]
    for name, S in r["sensors"].items():
        print(f"  {name} ({S['p_d']:g}, {S['p_fa']:g}): " + " ".join(f"{m:>9s}" for m in methods))
        for g, d in S["generators"].items():
            print(f"    {g:13s}" + " ".join(f"{d[m]['mean']:9.2f}" for m in methods))


def _table_runtime(r: dict) -> None:
    print(f"\nTable: per-slot compute time ({r['meta']['cpu']}, 1 thread)")
    for M, d in r["by_M"].items():
        row = "  ".join(f"{k} {d[k]['median_ms']:.2f}/{d[k]['p95_ms']:.2f}"
                        for k in ("bank", "predictor", "conformal_set", "tracker"))
        print(f"  M={M}: {row}  (median/p95 ms); build {d['build_seconds']['bank']:.1f} s")


def main(argv=None) -> None:
    data = PAPER / "data"
    data.mkdir(parents=True, exist_ok=True)
    loaded = {}
    for name in NAMES:
        src = C.RESULTS_DIR / f"{name}.json"
        if not src.exists():
            print(f"missing {src}; run python -m experiments.{name}")
            continue
        shutil.copyfile(src, data / f"{name}.json")
        loaded[name] = load_json(src)
    if "exp24_denial" in loaded:
        print("wrote", fig_denial(loaded["exp24_denial"], PAPER / "figures" / "fig_denial.png"))
    if "exp22_id_baselines" in loaded:
        _table_id(loaded["exp22_id_baselines"])
    if "exp23_prediction_baselines" in loaded:
        _table_pred(loaded["exp23_prediction_baselines"])
    if "exp25_runtime" in loaded:
        _table_runtime(loaded["exp25_runtime"])


if __name__ == "__main__":
    main()
