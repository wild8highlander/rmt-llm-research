"""Shared Black Aurora style for formal_proof figures (matches assets/charts)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

BG = "#050510"
PANEL = "#0a0a1c"
INK = "#e6f1ff"
MUTED = "#8ca3c7"
GRID = "#1c2340"
AUR = ["#00ffa3", "#22d3ee", "#a78bfa", "#f472b6", "#fb7185", "#38bdf8"]
AUR_MAP = LinearSegmentedColormap.from_list(
    "aurora", ["#050510", "#005f4b", "#00ffa3", "#22d3ee", "#a78bfa", "#f472b6", "#ffe3f0"]
)

plt.rcParams.update(
    {
        "font.sans-serif": ["DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.facecolor": BG,
        "axes.facecolor": PANEL,
        "savefig.facecolor": BG,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.labelcolor": INK,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.titlecolor": INK,
        "text.color": INK,
        "legend.facecolor": PANEL,
        "legend.edgecolor": GRID,
        "legend.labelcolor": INK,
        "legend.fontsize": 9,
        "grid.color": GRID,
        "grid.alpha": 0.6,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
    }
)

FIG_DIR = Path(__file__).resolve().parents[1] / "assets" / "figures"
ANIM_DIR = Path(__file__).resolve().parents[1] / "assets" / "animations"


def save(fig, name: str, dpi: int = 300) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / name
    fig.savefig(out, dpi=dpi, facecolor=BG)
    plt.close(fig)
    print(f"  saved {out.relative_to(FIG_DIR.parents[1])}")
    return out


def banner(fig, title: str, sub: str) -> None:
    """Two-line hero header; reserves the top 11% of the canvas."""
    fig.suptitle(title, color=INK, fontsize=12.5, y=1.0, va="top")
    fig.text(0.5, 0.952, sub, ha="center", va="top", color=MUTED, fontsize=8.5)
    try:
        fig.get_layout_engine().set(rect=(0, 0, 1, 0.885))
    except (AttributeError, ValueError):
        pass
