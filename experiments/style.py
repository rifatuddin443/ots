"""Shared, accessible plotting style for every figure.

One coherent visual system: a colorblind-safe categorical palette, light
gridlines, direct labels, and confidence bands rendered consistently. Importing
this module configures matplotlib; helpers keep the per-figure code short.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / file output
import matplotlib.pyplot as plt  # noqa: E402

# Colorblind-safe qualitative palette (Okabe-Ito), mapped to recurring series.
PALETTE = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "vermillion": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9",
    "yellow": "#F0E442", "grey": "#999999", "black": "#000000",
}
SERIES_COLORS = list(PALETTE.values())

# Stable color per method / generator so every figure agrees.
METHOD_COLORS = {
    "bank": PALETTE["blue"], "bank_sensor_aware": PALETTE["blue"],
    "bank_naive": PALETTE["vermillion"], "pairwise_sprt": PALETTE["orange"],
    "fixed_length": PALETTE["green"], "learned": PALETTE["purple"],
    "lstm": PALETTE["purple"], "eprocess": PALETTE["sky"], "msprt": PALETTE["blue"],
}
GENERATOR_COLORS = {
    "lfsr": PALETTE["blue"], "filtered_lfsr": PALETTE["sky"], "lcg": PALETTE["orange"],
    "weak_seed": PALETTE["green"], "markov": PALETTE["purple"],
    "csprng": PALETTE["vermillion"], "unknown": PALETTE["grey"],
}


def configure() -> None:
    plt.rcParams.update({
        "figure.figsize": (6.4, 4.2),
        "figure.dpi": 120,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "legend.frameon": False,
    })


configure()


def color_for(name: str, mapping: dict, idx: int = 0) -> str:
    if name in mapping:
        return mapping[name]
    return SERIES_COLORS[idx % len(SERIES_COLORS)]


def save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def band(ax, x, lo, hi, color, alpha: float = 0.18):
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, linewidth=0)
