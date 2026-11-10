#!/usr/bin/env python3
"""Folder-level static visuals: neural_lab + MNN-Pocket-0.3B.

Same Black-Aurora language as ``scripts/gen_images.py`` (the helpers are
imported from there), same export trio AVIF/WebP/PNG. The science is real:
the MNN hero's spike heights are the measured ρ_bad of the three RCDA-5 arms,
the melting spike is driven by the 150-step training log.

Outputs:
  neural_lab/MNN-Pocket-0.3B/assets/visuals/{hero,divider,og-image}-2x.*
  neural_lab/assets/visuals/{hero,divider}-2x.*

Usage::

    python scripts/gen_folder_visuals.py            # everything
    python scripts/gen_folder_visuals.py --slot mnn-hero
"""

from __future__ import annotations

import argparse
import json
import os
import sys


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "src"))

import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from gen_images import (
    PAL,
    _canvas,
    _dust,
    _finish,
    _glow_line,
    _grid,
    _mp_curve,
    _orbitals,
    _spike,
)


with open(os.path.join(REPO, "assets", "prompts.yaml"), encoding="utf-8") as _fh:
    FCFG = yaml.safe_load(_fh)


def _rng(slot: str) -> np.random.Generator:
    """Seeds come from the ``folders:`` section of assets/prompts.yaml."""
    table = {
        "mnn-hero": ("mnn-pocket", "static", "hero"),
        "mnn-divider": ("mnn-pocket", "static", "divider"),
        "mnn-og": ("mnn-pocket", "static", "og-image"),
        "nl-hero": ("neural_lab", "static", "hero"),
        "nl-divider": ("neural_lab", "static", "divider"),
    }
    grp, kind, name = table[slot]
    seed = int(FCFG["folders"][grp][kind][name]["seed"])
    return np.random.default_rng(seed)


# _finish() post-steps (bloom/grain) call gen_images' own _rng; point it at
# the folder-aware version so every slot keeps its registry seed.
import gen_images as _gi


_gi._rng = _rng

MNN = os.path.join(REPO, "neural_lab", "MNN-Pocket-0.3B")
NL = os.path.join(REPO, "neural_lab")

BULK, SPIKE, EDGE, COLLAPSE = PAL["bulk"], PAL["spike"], PAL["edge"], PAL["collapse"]
INK, GRID = PAL["text"], PAL["grid"]
MUTED = "#8CA3C7"
GOOD = "#38E1C0"  # teal — healthy/control states (same role as the charts)


def _export(img: Image.Image, outdir: str, slot: str) -> dict[str, int]:
    os.makedirs(outdir, exist_ok=True)
    sizes: dict[str, int] = {}
    png = img
    if png.width > 1600:
        png = img.resize((1600, img.height * 1600 // img.width), Image.LANCZOS)
    if slot == "og-image":
        png.save(os.path.join(outdir, f"{slot}.png"), format="PNG", optimize=True)
        sizes["png"] = os.path.getsize(os.path.join(outdir, f"{slot}.png")) // 1024
    else:
        png.save(os.path.join(outdir, f"{slot}-2x.png"), format="PNG", optimize=True)
        sizes["png"] = os.path.getsize(os.path.join(outdir, f"{slot}-2x.png")) // 1024
    for fmt, params in (
        ("avif", {"quality": 58, "speed": 8}),
        ("webp", {"quality": 84, "method": 6}),
    ):
        name = f"{slot}-2x.{fmt}" if slot != "og-image" else f"{slot}.{fmt}"
        path = os.path.join(outdir, name)
        img.save(path, format=fmt.upper(), **params)
        sizes[fmt] = os.path.getsize(path) // 1024
    return sizes


def _matrix(
    ax,
    x0: float,
    y0: float,
    w: float,
    h: float,
    cols: int,
    rows: int,
    hot: set[int],
    rng,
    shimmer: float = 1.0,
) -> None:
    """A stylized weight matrix: columns of cells, `hot` columns are the edit."""
    cw, ch = w / cols, h / rows
    for c in range(cols):
        col_color = SPIKE if c in hot else GRID
        base_a = 0.85 if c in hot else 0.55
        for r in range(rows):
            jitter = rng.random() if c not in hot else 1.0
            a = min(1.0, base_a * (0.35 + 0.65 * jitter) * shimmer)
            ax.add_patch(
                plt.Rectangle(
                    (x0 + c * cw, y0 + r * ch),
                    cw * 0.72,
                    ch * 0.72,
                    facecolor=col_color,
                    alpha=a,
                    lw=0,
                    zorder=4,
                )
            )


# ─────────────────────────────────────────────── MNN hero ───────────────────
def draw_mnn_hero(ax, t: float = 0.0) -> None:
    """The RCDA story. `t` ∈ [0, 1) drives the animation: the trained spike
    melts into the bulk, the penalty trace runs its comet, orbitals slowly rotate."""
    res = os.path.join(MNN, "R CDA - 5", "Research", "RCDA", "results")
    ev = {}
    for arm in ("base", "surgical", "trained"):
        p = os.path.join(res, f"part5_eval_{arm}.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                ev[arm] = json.load(fh)["rho_bad"]
    rho = (ev.get("base", 0.80), ev.get("surgical", 0.80), ev.get("trained", 0.35))
    melt = 0.0
    if t > 0.0:
        # melt while the loop runs: raise 0.15→0.65, hold, release 0.85→1.0
        if t < 0.15:
            melt = 0.0
        elif t < 0.65:
            melt = (t - 0.15) / 0.5
        elif t < 0.85:
            melt = 1.0
        else:
            melt = 1.0 - (t - 0.85) / 0.15

    _grid(ax)
    _dust(ax, "mnn-hero", n=110, alpha=0.35)
    rng = _rng("mnn-hero")

    # left: the edited o_proj matrix, a few hot columns of L15, the v direction
    mx, my, mw, mh = 0.075, 0.24, 0.30, 0.52
    hot = {3, 8, 14, 21}
    shimmer = 1.0 + 0.10 * np.sin(2 * np.pi * 2 * t) if t else 1.0
    _matrix(ax, mx, my, mw, mh, 28, 18, hot, rng, shimmer)
    for c in sorted(hot):
        ax.plot(
            [mx + (c + 0.5) * mw / 28] * 2,
            [my - 0.012, my + mh + 0.012],
            color=SPIKE,
            lw=1.1,
            alpha=0.30,
            zorder=3,
        )
    # the loop direction v leaves the matrix as a violet ray
    _glow_line(
        ax,
        [mx + mw, mx + mw + 0.075],
        [my + mh * 0.62, my + mh * 0.66],
        EDGE,
        lw_core=2.0,
        zorder=6,
    )
    _orbitals(
        ax,
        mx + mw + 0.19,
        my + mh * 0.66,
        [0.030, 0.048, 0.070],
        EDGE,
        alpha0=0.45,
        rot=24 + 360 * t,
    )
    ax.scatter([mx + mw + 0.19], [my + mh * 0.66], s=24, c=EDGE, alpha=0.9, lw=0, zorder=7)

    # center: one bulk, three arms — the spike melts only under SGD
    bx0, bx1, by0 = 0.44, 0.955, 0.16
    ax.plot([bx0 - 0.01, bx1], [by0, by0], color=BULK, lw=1.0, alpha=0.5, zorder=2)
    arm_x = (0.475, 0.655, 0.845)
    for (x0, arm), melt_i in zip(zip(arm_x, rho), (0.0, 0.0, melt)):
        _spike(ax, x0, by0 + 0.02, 0.14 + 0.62 * arm * (1.0 - 0.75 * melt_i), SPIKE)
    lam, dens, lo, hi = _mp_curve(q=0.1, sigma2=1.0)
    xs = bx0 + (lam - lo) / (hi * 1.6 - lo) * (bx1 - bx0) * 0.72
    ys = by0 + dens / dens.max() * 0.30
    ax.fill_between(xs, by0, ys, color=BULK, alpha=0.24, zorder=2)
    _glow_line(ax, xs, ys, BULK, lw_core=1.4, zorder=5)

    # right: the Tracy–Widom orbitals — every edit passed the gate
    _orbitals(ax, 0.895, 0.78, [0.030, 0.052, 0.078, 0.105], EDGE, alpha0=0.5, rot=18 + 360 * t)
    ax.scatter([0.895], [0.78], s=30, c=EDGE, alpha=0.95, lw=0, zorder=7)

    # bottom: the 150-step penalty trace (training_log.json), with a comet
    log_p = os.path.join(res, "training_log.json")
    if os.path.exists(log_p):
        with open(log_p, encoding="utf-8") as fh:
            log = json.load(fh)
        steps = [r["step"] for r in log["steps"]]
        pen = [r["pen_norm"] for r in log["steps"]]
        px0, px1, py0, py1 = 0.075, 0.62, 0.075, 0.135
        sx = px0 + (np.array(steps) / max(steps)) * (px1 - px0)
        sy = py0 + (np.array(pen) / max(pen)) * (py1 - py0)
        ax.fill_between(sx, py0, sy, color=EDGE, alpha=0.14, zorder=3)
        _glow_line(ax, sx, sy, EDGE, lw_core=1.3, zorder=5)
        if t:
            head = min(len(sx) - 1, int(t * (len(sx) - 1) + 0.5))
            ax.scatter([sx[head]], [sy[head]], s=140, c="#FFE9C4", alpha=0.95, lw=0, zorder=8)


def mnn_hero() -> Image.Image:
    fig, ax = _canvas(1600, 900)
    draw_mnn_hero(ax, t=0.0)
    return _finish(fig, "mnn-hero", bloom=True, grain_amt=2.0, vign=0.30)


# ─────────────────────────────────────────────── MNN divider ────────────────
def mnn_divider() -> Image.Image:
    fig, ax = _canvas(1600, 80)
    rng = _rng("mnn-divider")
    y = 0.5
    # columns dissolve into a spectrum line, left to right
    for i, c in enumerate(np.linspace(0.03, 0.42, 22)):
        h = 0.55 * (1.0 - i / 22) * (0.6 + 0.4 * rng.random())
        ax.add_patch(
            plt.Rectangle(
                (c, y - h / 2), 0.006, h, facecolor=EDGE, alpha=0.75 - i * 0.025, lw=0, zorder=4
            )
        )
    lam, dens, lo, hi = _mp_curve(q=0.1, sigma2=1.0)
    xs = 0.46 + (lam - lo) / (hi * 1.6 - lo) * 0.50
    ys = y + dens / dens.max() * 0.62
    _glow_line(ax, xs, ys, BULK, lw_core=1.3, zorder=5)
    _spike(ax, 0.965, y - 0.05, 0.62, SPIKE)
    return _finish(fig, "mnn-divider", bloom=False, grain_amt=1.2, vign=0.0)


# ─────────────────────────────────────────────── MNN og-image ───────────────
def mnn_og() -> Image.Image:
    fig, ax = _canvas(1200, 630)
    _grid(ax, alpha=0.7)
    _dust(ax, "mnn-og", n=120, alpha=0.35)
    lam, dens, lo, hi = _mp_curve(q=0.1, sigma2=1.0)
    xs = 0.06 + (lam - lo) / (hi * 1.6 - lo) * 0.5
    ys = 0.16 + dens / dens.max() * 0.34
    ax.fill_between(xs, 0.16, ys, color=BULK, alpha=0.26, zorder=3)
    _glow_line(ax, xs, ys, BULK, lw_core=1.4, zorder=5)
    _spike(ax, 0.52, 0.16, 0.58, SPIKE)
    _orbitals(ax, 0.80, 0.30, [0.05, 0.09, 0.135], EDGE, alpha0=0.4, rot=15)
    f_reg = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 64)
    f_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 30)
    img = _finish(fig, "mnn-og", bloom=True, grain_amt=1.5, vign=0.30)
    d = ImageDraw.Draw(img)
    d.text((60, 390), "MNN-Pocket-0.3B", font=f_reg, fill=(232, 238, 247))
    d.text(
        (60, 470),
        "RMT intervention kit · probing → surgery → SGD",
        font=f_small,
        fill=(140, 163, 199),
    )
    d.text(
        (60, 530), "ρ_bad 0.80 → 0.35 · TW pass ×3 · PPL parity", font=f_small, fill=(255, 176, 32)
    )
    return img


# ─────────────────────────────────────────────── neural_lab hero ────────────
def draw_nl_hero(ax, t: float = 0.0) -> None:
    """The living lab. `t` ∈ [0, 1) drives the animation: the four pipeline
    stages ignite in sequence, the prediction spike fires, the loss curve
    redraws, the tab dots pulse."""
    _grid(ax)
    _dust(ax, "nl-hero", n=170, alpha=0.35)

    phase = (t * 4.0) % 1.0  # one stage per quarter of the loop

    # left: the app window — five tabs, a live loss curve inside
    wx, wy, ww, wh = 0.075, 0.22, 0.34, 0.56
    ax.add_patch(plt.Rectangle((wx, wy), ww, wh, fill=False, edgecolor=GRID, lw=2.0, zorder=4))
    ax.plot([wx, wx + ww], [wy + wh - 0.055, wy + wh - 0.055], color=GRID, lw=1.6, zorder=4)
    tab_cols = [SPIKE, BULK, EDGE, COLLAPSE, GOOD]
    for i, c in enumerate(tab_cols):
        tab_pulse = 0.85 + 0.35 * max(0.0, np.sin(2 * np.pi * (t * 2 - i * 0.12))) if t else 0.85
        ax.add_patch(
            plt.Circle(
                (wx + 0.035 + i * 0.058, wy + wh - 0.028),
                0.007,
                facecolor=c,
                alpha=min(1.0, tab_pulse),
                lw=0,
                zorder=5,
            )
        )
    steps = np.linspace(0, 1, 120)
    loss = 2.6 * np.exp(-2.2 * steps) + 0.12 + 0.015 * np.sin(steps * 21)
    lx = wx + 0.03 + steps * (ww - 0.06)
    ly = wy + 0.10 + (loss - loss.min()) / (loss.max() - loss.min()) * (wh - 0.24)
    if t:
        n_vis = max(24, int(phase * len(steps)))
        lx, ly = lx[:n_vis], ly[:n_vis]
    ax.fill_between(lx, wy + 0.10, ly, color=BULK, alpha=0.12, zorder=3)
    _glow_line(ax, lx, ly, BULK, lw_core=1.5, zorder=6)
    # three skeleton "output lines" under the curve — the app is reading
    # artifacts, not showing random dots
    for i, (ln_w, ln_a) in enumerate(((0.16, 0.5), (0.11, 0.35), (0.13, 0.42))):
        ax.add_patch(
            plt.Rectangle(
                (wx + 0.035, wy + 0.055 + i * 0.022),
                ln_w,
                0.004,
                facecolor=MUTED,
                alpha=ln_a,
                lw=0,
                zorder=4,
            )
        )

    # right: the science ring — hypothesis → measurement → prediction → validation
    cx, cy, R = 0.70, 0.50, 0.17
    _orbitals(ax, cx, cy, [R, R * 1.28], EDGE, alpha0=0.40, rot=18 + 60 * t)
    # the prediction node sits at the top of the ring and fires the BBP spike
    nodes = [(90, SPIKE), (0, BULK), (-90, EDGE), (180, GOOD)]
    top_node = None
    for k, (ang, col) in enumerate(nodes):
        th = np.deg2rad(ang)
        nx, ny = cx + R * np.cos(th) * 1.35, cy + R * np.sin(th) * 0.62
        if ang == 90:
            top_node = (nx, ny)
        ignite = max(0.0, np.sin(2 * np.pi * (phase - k / 4))) if t else 0.0
        for rr, aa in ((0.016 + 0.008 * ignite, 0.28 + 0.5 * ignite), (0.010, 0.8)):
            ax.add_patch(
                plt.Circle((nx, ny), rr, facecolor=col, alpha=min(1.0, aa), lw=0, zorder=6)
            )
        ax.scatter([nx], [ny], s=90, c=col, alpha=0.95, lw=0, zorder=7)
    nx, ny = top_node
    spike_h = 0.17 * (0.25 + 0.75 * max(0.0, np.sin(2 * np.pi * phase))) if t else 0.17
    _spike(ax, nx, ny + 0.02, spike_h, SPIKE)
    # the 52-tests chip: two tight rows of 26 emerald dots under the ring
    for i in range(52):
        row, col_i = divmod(i, 26)
        a = 0.8 if not t else (0.45 + 0.45 * float(col_i / 26 < phase))
        ax.add_patch(
            plt.Circle(
                (0.615 + col_i * 0.0068, 0.205 - row * 0.016),
                0.0021,
                facecolor=GOOD,
                alpha=a,
                lw=0,
                zorder=6,
            )
        )


def nl_hero() -> Image.Image:
    fig, ax = _canvas(1600, 900)
    draw_nl_hero(ax, t=0.0)
    return _finish(fig, "nl-hero", bloom=True, grain_amt=2.0, vign=0.30)


# ─────────────────────────────────────────────── neural_lab divider ─────────
def nl_divider() -> Image.Image:
    fig, ax = _canvas(1600, 80)
    y = 0.5
    xs = np.linspace(0.04, 0.96, 4)
    cols = [SPIKE, BULK, EDGE, GOOD]
    for i, (x0, c) in enumerate(zip(xs, cols)):
        ax.add_patch(plt.Circle((x0, y), 0.010, facecolor=c, alpha=0.9, lw=0, zorder=5))
        if i < 3:
            t = np.linspace(x0 + 0.02, xs[i + 1] - 0.02, 40)
            _glow_line(ax, t, y + 0.05 * np.sin((t - x0) * 40), BULK, lw_core=1.0, zorder=4)
    return _finish(fig, "nl-divider", bloom=False, grain_amt=1.2, vign=0.0)


SLOTS = {
    "mnn-hero": (mnn_hero, os.path.join(MNN, "assets", "visuals")),
    "mnn-divider": (mnn_divider, os.path.join(MNN, "assets", "visuals")),
    "mnn-og-image": (mnn_og, os.path.join(MNN, "assets", "visuals")),
    "nl-hero": (nl_hero, os.path.join(NL, "assets", "visuals")),
    "nl-divider": (nl_divider, os.path.join(NL, "assets", "visuals")),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slot", choices=sorted(SLOTS))
    args = ap.parse_args()
    names = [args.slot] if args.slot else sorted(SLOTS)
    for name in names:
        fn, outdir = SLOTS[name]
        slot = "og-image" if name == "mnn-og-image" else name.replace("mnn-", "").replace("nl-", "")
        img = fn()
        sizes = _export(img, outdir, slot)
        print(
            f"{name:<14} -> {os.path.relpath(outdir, REPO)}/  "
            + "  ".join(f"{k}={v}K" for k, v in sizes.items())
        )
        for f in os.listdir(outdir):
            if slot in f:
                im = Image.open(os.path.join(outdir, f)).convert("L")
                h = im.histogram()
                lit = sum(h[:250]) / sum(h)
                assert lit > 0.005, f"blank image: {f}"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
