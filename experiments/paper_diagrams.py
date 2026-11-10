"""Schematic diagrams for the IEEE conference paper (not data plots).

Generates two vector-styled PNGs into docs/paper/ieee/figures/:
  fig_architecture.png  -- the end-to-end data flow (sensing -> bank -> predictor
                           -> conformal set -> allocation), with the feedback
                           loops (switch monitor, online calibration).
  fig_methodology.png   -- the per-slot engagement-loop flowchart (Algorithm 1).

Pure matplotlib (patches + annotations), so it renders anywhere the plots do.
Run:  python -m experiments.paper_diagrams
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path("docs/paper/ieee/figures")

# A restrained, print-safe palette (works in grayscale too).
C_SENSE = "#E8EEF6"
C_ID = "#FBE8D3"
C_PRED = "#DDEBD8"
C_ACT = "#E7E0F0"
C_EDGE = "#3A3A3A"
C_FB = "#B23A48"


def _box(ax, xy, w, h, text, fc, *, fs=8.2, lw=0.9, bold_first=True):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=lw, edgecolor=C_EDGE, facecolor=fc,
                                mutation_aspect=1.0, zorder=2))
    lines = text.split("\n")
    n = len(lines)
    # Stack lines evenly inside the box, top line optionally bold.
    gap = h / (n + 0.6)
    top = y + h - gap * 0.85
    for i, ln in enumerate(lines):
        bold = bold_first and i == 0
        ax.text(x + w / 2, top - i * gap, ln, ha="center", va="center",
                fontsize=fs if bold else fs - 0.7,
                fontweight="bold" if bold else "normal", zorder=3)
    return (x, y, w, h)


def _arrow(ax, p0, p1, *, color=C_EDGE, style="-|>", rad=0.0, lw=1.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=11,
                                 linewidth=lw, color=color, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=1.5,
                                 shrinkA=1.5, shrinkB=1.5))


def _r(b):   # right-edge midpoint
    x, y, w, h = b; return (x + w, y + h / 2)


def _l(b):
    x, y, w, h = b; return (x, y + h / 2)


def _t(b):
    x, y, w, h = b; return (x + w / 2, y + h)


def _bm(b):
    x, y, w, h = b; return (x + w / 2, y)


def architecture() -> Path:
    fig, ax = plt.subplots(figsize=(7.1, 3.15))
    ax.set_xlim(0, 10.25); ax.set_ylim(0, 4.35); ax.axis("off")

    em = _box(ax, (0.15, 2.55), 1.55, 1.0,
              "FH emitter\ngenerator $G$\n(unknown; may switch)", C_SENSE)
    sen = _box(ax, (0.15, 1.05), 1.55, 1.0,
               "Passive sensor\ndetection level\n$(P_d,\\,P_{fa})$", C_SENSE)
    emi = _box(ax, (2.15, 1.05), 1.6, 1.0,
               "Reduced emission\n$\\log e_t(s)$\nEq. (1)", C_SENSE)

    bank = _box(ax, (4.2, 2.45), 2.2, 1.35,
                "Hypothesis bank\n$\\{H_1,\\dots,H_K\\}\\cup H_0$\n"
                "sensor-aware likelihood\nMSPRT / e-process", C_ID)
    sw = _box(ax, (4.2, 0.95), 2.2, 1.05,
              "Switch monitor\nBOCPD + context flush\n(detect / date / re-ID)", C_ID)

    pred = _box(ax, (6.95, 2.55), 2.9, 1.25,
                "Next-channel predictor\nFixed-Share mixture +\n"
                "training-free universal expert\n$\\hat{p}_t(c)$, $\\kappa_t$", C_PRED)
    conf = _box(ax, (6.95, 1.2), 2.9, 1.05,
                "Conformal set $S_t$\nadaptive coverage $1-\\alpha$\n(ACI, erasure feedback)", C_PRED)
    act = _box(ax, (6.95, 0.05), 2.9, 0.85,
               "Multi-tone allocation\nthreshold-aware knapsack", C_ACT)

    # forward path
    _arrow(ax, _bm(em), _t(sen))
    _arrow(ax, _r(sen), _l(emi))
    _arrow(ax, _r(emi), (bank[0], bank[1] + 0.35))
    _arrow(ax, _r(emi), (sw[0], sw[1] + 0.55))
    _arrow(ax, _r(bank), _l(pred))
    _arrow(ax, _bm(pred), _t(conf))
    _arrow(ax, _bm(conf), _t(act))
    # switch monitor informs bank and predictor (dashed, red)
    _arrow(ax, _t(sw), _bm(bank), color=C_FB, rad=0.0, ls=(0, (4, 2)))
    _arrow(ax, _r(sw), (pred[0], pred[1] + 0.2), color=C_FB, rad=-0.16, ls=(0, (4, 2)))

    # feedback: realised-hop (erasure) calibration of the conformal level,
    # drawn as a clean loop up the right margin from the conformal box.
    _arrow(ax, _r(conf), (9.96, 1.72), color=C_FB, rad=0.0, ls=(0, (4, 2)))
    _arrow(ax, (9.96, 1.72), (9.85, pred[1] + 0.25), color=C_FB, rad=-0.5,
           ls=(0, (4, 2)))
    ax.text(5.3, 0.3,
            "erasure feedback (clean slots):  "
            "$\\alpha_{t+1}=\\alpha_t+\\gamma(\\alpha-\\mathrm{err}_t)$",
            ha="center", va="center", fontsize=6.8, color=C_FB)

    ax.text(0.9, 4.05, "Sensing", ha="center", fontsize=7.6, style="italic", color="#555")
    ax.text(5.3, 4.12, "Identification & tracking", ha="center", fontsize=7.6,
            style="italic", color="#555")
    ax.text(8.4, 4.12, "Prediction & action", ha="center", fontsize=7.6,
            style="italic", color="#555")

    fig.tight_layout(pad=0.2)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "fig_architecture.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p


def methodology() -> Path:
    fig, ax = plt.subplots(figsize=(3.4, 6.2))
    ax.set_xlim(0, 6); ax.set_ylim(0, 12.6); ax.axis("off")
    cx = 3.0; w = 4.3

    def center_box(y, h, text, fc, fs=8.0):
        return _box(ax, (cx - w / 2, y), w, h, text, fc, fs=fs)

    b0 = center_box(11.5, 0.85, "Observe slot $o_t$\n(lossy detections)", C_SENSE)
    b1 = center_box(10.1, 1.0, "Reduced emission $\\log e_t(s)$\n$F_t$ cancels across $H_k$", C_SENSE)
    b2 = center_box(8.55, 1.15,
                    "For each $H_k$: sensor-aware\nforward update $\\ell_k$; accumulate\n"
                    "$\\Lambda(k)$; online $\\mathrm{observe\\_clean}$", C_ID, fs=7.6)
    b3 = center_box(7.2, 0.95, "Bayesian update\n$\\pi_t\\propto\\pi_0\\,e^{\\Lambda}$", C_ID)
    b4 = center_box(5.85, 1.0, "Fixed-Share predictor\n$\\hat p_t(c)$, $\\kappa_t$, conformal $S_t$", C_PRED)

    # decision diamond (drawn as rotated box via polygon)
    import matplotlib.patches as mpatches
    dy, dh = 4.2, 1.25
    diamond = mpatches.FancyBboxPatch((cx - 1.55, dy), 3.1, dh,
                                      boxstyle="round,pad=0.01", linewidth=0.9,
                                      edgecolor=C_EDGE, facecolor=C_ACT, zorder=2)
    ax.add_patch(diamond)
    ax.text(cx, dy + dh / 2, "Stop rule?\n$\\pi_t(k^*)\\geq 1-\\delta$  or  $E_t\\geq 1/\\beta$",
            ha="center", va="center", fontsize=7.4, zorder=3)
    bD = (cx - 1.55, dy, 3.1, dh)

    b5 = center_box(2.5, 0.95, "Declare $\\arg\\max_k\\pi_t$\n($k^*{=}0\\Rightarrow$ unknown)", C_ID)
    b6 = center_box(1.15, 0.85, "Multi-tone response\nfrom $S_t$ / $\\hat p_t$", C_ACT)

    _arrow(ax, _bm(b0), _t(b1))
    _arrow(ax, _bm(b1), _t(b2))
    _arrow(ax, _bm(b2), _t(b3))
    _arrow(ax, _bm(b3), _t(b4))
    _arrow(ax, _bm(b4), (cx, dy + dh))
    _arrow(ax, _bm(bD), _t(b5))
    _arrow(ax, _bm(b5), _t(b6))

    # "no -> next slot" loop back to the top
    ax.text(cx + 1.75, dy + dh / 2, "no", fontsize=7.2, va="center")
    _arrow(ax, (cx + 1.55, dy + dh / 2), (5.75, dy + dh / 2))
    _arrow(ax, (5.75, dy + dh / 2), (5.75, 11.92))
    _arrow(ax, (5.75, 11.92), _r(b0))
    ax.text(5.85, 8.2, "$t\\leftarrow t{+}1$", fontsize=7.2, rotation=90, va="center")
    ax.text(cx, dy - 0.2, "yes", fontsize=7.2, ha="center", va="top")

    fig.tight_layout(pad=0.2)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "fig_methodology.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p


if __name__ == "__main__":
    print(architecture())
    print(methodology())
