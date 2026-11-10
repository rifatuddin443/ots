"""Result figures for the IEEE conference paper.

Reads the refreshed experiment JSON in results/ and the larger paper runs in
docs/paper/ieee/data/ (written by experiments.paper_extras) and renders the
paper's result figures into docs/paper/ieee/figures/:

  fig_confusion.png   4 family-confusion matrices (one per sensor), 13 generators.
  fig_results.png     3 panels: hops-to-identify vs P_d; held-out next-channel
                      prediction (proposed vs id-only) across sensors; conformal
                      coverage calibration + jam-hit vs fixed top-3.
  fig_mismatch.png    mis-specified-sensor identification accuracy heatmap.

Run after paper_extras:  python -m experiments.paper_figures
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = Path("results")
DATA = Path("docs/paper/ieee/data")
OUT = Path("docs/paper/ieee/figures")

SENSOR_ORDER = ("perfect", "strong", "typical", "contested")
SENSOR_LABEL = {"perfect": "Perfect (1.0, 0)", "strong": "Strong (0.95, 0.01)",
                "typical": "Typical (0.8, 0.02)", "contested": "Contested (0.6, 0.05)"}


def _load(path):
    return json.load(open(path)) if Path(path).exists() else None


def confusion() -> Path | None:
    d = _load(DATA / "extra_confusion.json")
    if d is None:
        print("[confusion] no data yet"); return None
    fams = d["meta"]["families"]
    gens = d["meta"]["generators"]
    short = {"filtered_lfsr": "filt_lfsr", "lcg_full": "lcg_full", "lagged_fib": "lag_fib",
             "quadratic_cg": "quad_cg", "sine_map": "sine_map"}
    glabel = [short.get(g, g) for g in gens]
    flabel = [f if f != "unknown" else "H0/unk" for f in fams]

    fig, axes = plt.subplots(2, 2, figsize=(7.1, 7.0))
    for ax, name in zip(axes.ravel(), SENSOR_ORDER):
        sd = d["sensors"][name]
        n = d["meta"]["n_seeds"]
        mat = np.array([[sd["counts"][g][f] / n for f in fams] for g in gens])
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(fams))); ax.set_xticklabels(flabel, rotation=45, ha="right", fontsize=6.3)
        ax.set_yticks(range(len(gens)))
        ax.set_yticklabels([f"{gl}" for gl in glabel], fontsize=6.3)
        for i in range(len(gens)):
            for j in range(len(fams)):
                v = mat[i, j]
                if v > 0.02:
                    ax.text(j, i, f"{v:.2f}".lstrip("0") if v < 1 else "1",
                            ha="center", va="center", fontsize=5.2,
                            color="white" if v > 0.5 else "#333")
        ax.set_title(SENSOR_LABEL[name], fontsize=7.6)
        ax.tick_params(length=0)
    fig.supxlabel("declared family", fontsize=8); fig.supylabel("true generator", fontsize=8)
    fig.tight_layout(rect=(0.01, 0.0, 1, 1), pad=0.5)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "fig_confusion.png"; fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
    return p


def _hops_panel(ax):
    d = _load(RES / "exp13_coverage.json")
    if d is None:
        return
    pd_grid = d["meta"]["p_d_grid"]
    show = [("lfsr", "LFSR"), ("lcg", "LCG"), ("weak_seed", "weak-seed"),
            ("markov", "Markov"), ("csprng", "CSPRNG")]
    for g, lab in show:
        rows = d["by_type"][g]
        hops = []
        for pd in pd_grid:
            vs = [r["mean_hops_to_identify"] for r in rows
                  if r["p_d"] == pd and not np.isnan(r["mean_hops_to_identify"])]
            hops.append(np.mean(vs) if vs else np.nan)
        ax.plot(pd_grid, hops, "-o", ms=3.5, lw=1.2, label=lab)
    ax.invert_xaxis()
    ax.set_xlabel("detection probability $P_d$", fontsize=8)
    ax.set_ylabel("mean hops to identify", fontsize=8)
    ax.set_title("(a) Identification speed", fontsize=8.5)
    ax.legend(fontsize=6.3, ncol=1, loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.25); ax.tick_params(labelsize=7)


def _predict_panel(ax):
    """Held-out generator next-channel top-1 (steady), proposed vs id-only,
    across sensors.  strong = results/exp14; others = extra_predict."""
    held = [("lagged_fib", "lagged-Fib"), ("quadratic_cg", "quad-CG"), ("sine_map", "sine-map")]
    e14 = _load(RES / "exp14_next_channel_prediction.json")
    ex = _load(DATA / "extra_predict.json")
    series = {}
    if e14:
        series["strong"] = e14["operating_point"]
    if ex:
        for nm in ("typical", "contested"):
            if nm in ex["sensors"]:
                series[nm] = ex["sensors"][nm]["operating_point"]
    sensors = [s for s in ("strong", "typical", "contested") if s in series]
    x = np.arange(len(held)); width = 0.8 / max(1, len(sensors))
    base = plt.cm.Greens(np.linspace(0.75, 0.45, len(sensors)))
    hatch_cycle = ["", "", ""]
    for si, s in enumerate(sensors):
        op = series[s]
        prop = [op[g[0]]["proposed_steady"] for g in held]
        idon = [op[g[0]]["id_only_steady"] for g in held]
        off = (si - (len(sensors) - 1) / 2) * width
        ax.bar(x + off, prop, width * 0.92, color=base[si],
               label=f"proposed · {s}", edgecolor="#2a4d2a", lw=0.4)
        ax.plot(x + off, idon, "_", color="#b23a48", ms=7, mew=1.6,
                label="id-only" if si == 0 else None)
    ax.axhline(1 / 16, ls=":", color="gray", lw=1)
    ax.text(2.35, 1 / 16 + 0.01, "chance", fontsize=6, color="gray", ha="right")
    ax.set_xticks(x); ax.set_xticklabels([h[1] for h in held], fontsize=7)
    ax.set_ylim(0, 1.08); ax.set_ylabel("steady-state top-1", fontsize=8)
    ax.set_title("(b) Prediction, held-out generators", fontsize=8.5)
    ax.legend(fontsize=5.6, ncol=2, loc="lower left", framealpha=0.9)
    ax.tick_params(labelsize=7); ax.grid(axis="y", alpha=0.25)


def _conformal_panel(ax):
    d = _load(RES / "exp20_conformal_sets.json")
    if d is None:
        ax.set_axis_off(); return
    cal = d["calibration"]
    ax.plot(cal["target"], cal["target"], ":", color="gray", lw=1, label="ideal")
    ax.plot(cal["target"], cal["conformal"], "-o", ms=3.5, lw=1.3,
            color="#1f6fb2", label="conformal set")
    ax.axhline(cal["top3"], ls="--", color="#e08214", lw=1.1, label=f"fixed top-3 ({cal['top3']:.2f})")
    ax.axhline(cal["top1"], ls="--", color="#b23a48", lw=1.1, label=f"fixed top-1 ({cal['top1']:.2f})")
    ax.set_xlabel("target coverage $1-\\alpha$", fontsize=8)
    ax.set_ylabel("realised coverage", fontsize=8)
    ax.set_title("(c) Conformal coverage", fontsize=8.5)
    ax.set_ylim(0.4, 1.03); ax.legend(fontsize=6, loc="lower right", framealpha=0.9)
    ax.grid(alpha=0.25); ax.tick_params(labelsize=7)


def results() -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.6))
    _hops_panel(axes[0]); _predict_panel(axes[1]); _conformal_panel(axes[2])
    fig.tight_layout(pad=0.6)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "fig_results.png"; fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
    return p


def mismatch() -> Path | None:
    d = _load(DATA / "extra_mismatch.json")
    if d is None:
        print("[mismatch] no data yet"); return None
    grid = d["meta"]["p_d_grid"]
    fa_grid = d["meta"].get("p_fa_grid", [])
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.2))

    acc = np.array([[c["accuracy"] for c in row] for row in d["pd"]])
    im0 = axes[0].imshow(acc, cmap="RdYlGn", vmin=0.3, vmax=1.0, aspect="auto", origin="upper")
    axes[0].set_xticks(range(len(grid))); axes[0].set_xticklabels(grid, fontsize=7)
    axes[0].set_yticks(range(len(grid))); axes[0].set_yticklabels(grid, fontsize=7)
    axes[0].set_xlabel("assumed $P_d$", fontsize=8); axes[0].set_ylabel("true $P_d$", fontsize=8)
    axes[0].set_title("(a) Identification accuracy,\nmis-specified $P_d$", fontsize=8.2)
    for i in range(len(grid)):
        for j in range(len(grid)):
            axes[0].text(j, i, f"{acc[i, j]:.2f}".lstrip("0"), ha="center", va="center",
                         fontsize=6, color="black")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.03)

    if d.get("pfa"):
        hard = np.array([[c["hard_false_id"] for c in row] for row in d["pfa"]])
        im1 = axes[1].imshow(hard, cmap="Reds", vmin=0, vmax=0.2, aspect="auto", origin="upper")
        axes[1].set_xticks(range(len(fa_grid))); axes[1].set_xticklabels(fa_grid, fontsize=7)
        axes[1].set_yticks(range(len(fa_grid))); axes[1].set_yticklabels(fa_grid, fontsize=7)
        axes[1].set_xlabel("assumed $P_{fa}$", fontsize=8); axes[1].set_ylabel("true $P_{fa}$", fontsize=8)
        axes[1].set_title("(b) Hard false-ID,\nmis-specified $P_{fa}$", fontsize=8.2)
        for i in range(len(fa_grid)):
            for j in range(len(fa_grid)):
                axes[1].text(j, i, f"{hard[i, j]:.2f}".lstrip("0"), ha="center", va="center",
                             fontsize=6, color="black")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.03)
    else:
        axes[1].set_axis_off()
    fig.tight_layout(pad=0.6)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "fig_mismatch.png"; fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
    return p


if __name__ == "__main__":
    for fn in (confusion, results, mismatch):
        try:
            print(fn(), flush=True)
        except Exception as e:  # keep going if one input is missing
            print(f"[{fn.__name__}] skipped: {e}", flush=True)
