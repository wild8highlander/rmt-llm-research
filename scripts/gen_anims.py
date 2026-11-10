#!/usr/bin/env python3
"""Render the animation layer from ``assets/prompts.yaml`` (procedural).

Every loop is seamless by construction: each motion driver is a periodic
function of the loop phase t in [0, 1), so frame 0 equals frame N and no
cut is visible. There is no text inside any frame (the terminal demo is
the one sanctioned exception), no flicker — only eased periodic motion.

Outputs per slot::

    assets/anim/<slot>-loop.webm   (VP9, primary)
    assets/anim/<slot>-loop.mp4    (H.264 fallback)
    assets/visuals/<slot>-poster.avif

plus the terminal recording ``assets/anim/cli-demo.gif``.

Usage::

    python scripts/gen_anims.py --all
    python scripts/gen_anims.py --slot hero
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from gen_images import (
    OUT_DIR as VISUALS_DIR,
)
from gen_images import (
    PAL,
    _canvas,
    _dust,
    _glow_line,
    _grid,
    _orbitals,
    _rgb,
    _spike,
)
from rmt_llm.bbp_transition import bbp_lambda_max
from rmt_llm.marchenko_pastur import mp_bounds, mp_density


FFMPEG = shutil.which("ffmpeg")

with open(os.path.join(REPO, "assets", "prompts.yaml"), encoding="utf-8") as _fh:
    CFG = yaml.safe_load(_fh)
ANIM_DIR = os.path.join(REPO, "assets", "anim")


def _ease(phase: float) -> float:
    """Smooth 0->1->0 raise over one loop, zero derivative at the ends."""
    return 0.5 - 0.5 * np.cos(2 * np.pi * phase)


def _frames(slot: str):
    return int(CFG["anims"][slot]["duration_s"] * CFG["anims"][slot]["fps"])


# ---------------------------------------------------------------- slots ----
def draw_hero(ax, t: float) -> None:
    _grid(ax, alpha=0.9)
    _dust(ax, "hero", n=220)
    breath = 1.0 + 0.055 * np.sin(2 * np.pi * t)
    sway = 0.012 * np.sin(2 * np.pi * t + 1.1)
    q, s2 = 0.1, 1.0
    lo, hi = mp_bounds(q, s2)
    lam = np.linspace(max(lo - (hi - lo) * 0.45, 1e-6), hi + (hi - lo) * 0.72, 700)
    rho = mp_density(lam, q, s2)
    theta = 1.15 * float(np.sqrt(q))
    out = float(bbp_lambda_max(theta, q, s2))
    xmax = out * 1.30
    xs = 0.03 + (lam / xmax) * 0.94
    ys = 0.09 + (rho / rho.max()) * 0.40 * breath + sway
    ax.fill_between(xs, 0.09, ys, color=PAL["bulk"], alpha=0.13, lw=0, zorder=2)
    _glow_line(ax, xs, ys, PAL["bulk"], lw_core=3.2)
    s = _ease(t)
    sx = 0.03 + (out / xmax) * 0.94
    _spike(ax, sx, 0.09, 0.10 + 0.38 * s, PAL["spike"])
    ax.plot([0.03, 0.97], [0.09, 0.09], color=PAL["grid"], lw=1.2, alpha=0.8, zorder=2)
    _orbitals(
        ax,
        0.80,
        0.76,
        [0.045, 0.075, 0.11, 0.15],
        PAL["edge"],
        alpha0=0.55,
        rot=24 + 9 * np.sin(2 * np.pi * t),
    )
    for s_, a in ((2600, 0.05), (900, 0.09)):
        ax.scatter(
            [0.45 + 0.02 * np.sin(2 * np.pi * t)],
            [0.30 + 0.015 * np.cos(2 * np.pi * t)],
            s=s_,
            c=PAL["bulk"],
            alpha=a,
            lw=0,
            zorder=1,
        )


def draw_thread1(ax, t: float) -> None:
    _grid(ax, alpha=0.8)
    _dust(ax, "thread-1-spectral", n=90)
    q, s2 = 0.3, 1.0
    lo, hi = mp_bounds(q, s2)
    lam = np.linspace(max(lo - (hi - lo) * 0.45, 1e-6), hi * 1.9, 600)
    rho = mp_density(lam, q, s2)
    xs = 0.06 + (lam / (hi * 1.9)) * 0.80
    ys = 0.16 + (rho / rho.max()) * 0.42
    ax.fill_between(xs, 0.16, ys, color=PAL["bulk"], alpha=0.12, lw=0, zorder=2)
    _glow_line(ax, xs, ys, PAL["bulk"], lw_core=2.4)
    edge_x = 0.06 + (hi / (hi * 1.9)) * 0.80
    ax.plot(
        [edge_x, edge_x],
        [0.12, 0.86],
        color="#E8EEF7",
        lw=1.4,
        ls=(0, (6, 5)),
        alpha=0.50,
        zorder=4,
    )
    shimmer = 0.30 + 0.22 * np.sin(2 * np.pi * (t * 2 + np.linspace(0, 3, 52)))
    rng_f = np.random.default_rng(20261109)
    flick_y = rng_f.uniform(0.16, 0.80, 52)
    ax.scatter(
        np.full(52, edge_x),
        flick_y,
        s=26,
        c=PAL["edge"],
        alpha=0.10 * np.ones(52),
        marker="|",
        lw=1.2,
        zorder=5,
    )
    ax.scatter(
        np.full(52, edge_x),
        flick_y,
        s=2.2,
        c=PAL["edge"],
        alpha=np.clip(shimmer, 0.05, 0.6),
        marker="|",
        lw=1.6,
        zorder=5,
    )
    s = _ease(t)
    x0 = edge_x - 0.10
    x1 = edge_x + 0.045
    px = x0 + (x1 - x0) * s
    py = 0.30 + 0.34 * s * 1.0
    _spike(ax, x1, 0.30, 0.05 + 0.30 * s, PAL["spike"], core_h=0.5)
    trail_t = np.clip(t - np.linspace(0.015, 0.09, 7), 0, 1)
    ts_e = _ease_array(trail_t)
    ax.scatter(
        x0 + (x1 - x0) * ts_e,
        0.30 + 0.34 * ts_e,
        s=40,
        c=PAL["spike"],
        alpha=0.35 * np.linspace(1, 0.1, 7),
        lw=0,
        zorder=6,
    )
    for sz, al in ((320, 0.10), (90, 0.35), (20, 1.0)):
        ax.scatter([px], [py], s=sz, c=PAL["spike"], alpha=al, lw=0, zorder=7)


def _ease_array(ph: np.ndarray) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.clip(ph, 0, 1))


def draw_thread2(ax, t: float) -> None:
    _grid(ax, alpha=0.75)
    _dust(ax, "thread-2-hallucination", n=80)
    cx, cy = 0.30, 0.52
    _orbitals(
        ax,
        cx,
        cy,
        [0.05, 0.085, 0.12, 0.155, 0.19],
        PAL["edge"],
        alpha0=0.5,
        rot=14 + 10 * np.sin(2 * np.pi * t),
    )
    pulse = 1.0 + 0.10 * np.sin(2 * np.pi * t)
    ax.scatter([cx], [cy], s=340 * pulse, c="#000000", zorder=4)
    ax.scatter([cx], [cy], s=60, c="#E8EEF7", alpha=0.9, zorder=5)
    ax.plot([0.62, 0.62], [0.08, 0.92], color="#E8EEF7", lw=1.6, alpha=0.55, zorder=4)
    u = _ease(t)
    tt = np.linspace(0, 1, 120)
    bx = cx + 0.10 + (0.62 - cx - 0.14) * tt + 0.055 * np.sin(tt * np.pi) * 0.9
    by = cy + 0.16 * np.sin(tt * np.pi) + 0.045 * tt * 2.4
    ax.plot(bx, by, color=PAL["spike"], lw=1.0, alpha=0.22, zorder=5)
    m = tt <= u
    _glow_line(ax, bx[m], by[m], PAL["spike"], lw_core=2.4)
    hx, hy = float(np.interp(u, tt, bx)), float(np.interp(u, tt, by))
    for sz, al in ((300, 0.10), (80, 0.4), (18, 1.0)):
        ax.scatter([hx], [hy], s=sz, c=PAL["spike"], alpha=al, lw=0, zorder=7)
    rng = np.random.default_rng(20261109)
    n = 420
    gx = rng.normal(0.70, 0.05, n)
    gy = rng.normal(0.56, 0.10, n) * (1 + 0.4 * np.cos(gx * 9))
    leak = 1.0 + 0.35 * _ease(t)
    ax.scatter(
        0.70 + (gx - 0.70) * leak,
        gy,
        s=5,
        c=PAL["collapse"],
        alpha=0.20 + 0.20 * _ease(t),
        lw=0,
        zorder=6,
    )
    ax.scatter(0.70 + (gx - 0.70) * leak, gy, s=30, c=PAL["collapse"], alpha=0.07, lw=0, zorder=5)


def draw_thread3(ax, t: float) -> None:
    _grid(ax, alpha=0.7)
    _dust(ax, "thread-3-rlhf", n=70)
    rng = np.random.default_rng(20261109)
    cx, cy = 0.56, 0.50
    n = 2400
    phi = rng.random(n) * 4.2 * np.pi
    r0 = 0.030 + 0.34 * np.exp(-phi / 7.5)
    jitter = rng.normal(0, 0.016, n)
    spin = 2 * np.pi * t * 0.5  # half a turn per loop
    contract = 1.0 - 0.48 * _ease(t)
    r = r0 * contract
    px = cx + (r + jitter * contract) * np.cos(phi + spin)
    py = cy + (r + jitter * contract) * np.sin(phi + spin) * 0.62
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("cv", [PAL["collapse"], PAL["edge"], PAL["bulk"]])
    col_mix = np.clip(r0 / r0.max(), 0, 1)
    ax.scatter(px, py, s=80, c=col_mix, cmap=cmap, alpha=0.06, lw=0, zorder=3)
    ax.scatter(px, py, s=12, c=col_mix, cmap=cmap, alpha=0.15, lw=0, zorder=3)
    ax.scatter(px, py, s=3.0, c=col_mix, cmap=cmap, alpha=0.7, lw=0, zorder=4)
    for k, fade in enumerate((0.40, 0.26, 0.15, 0.08)):
        ph = np.linspace(0, 4.6 * np.pi, 300)
        rr = (0.02 + 0.36 * np.exp(-ph / (6.0 - k * 0.7))) * contract
        ax.plot(
            cx + rr * np.cos(ph + 0.6 * k - spin),
            cy + rr * np.sin(ph + 0.6 * k - spin) * 0.62,
            color=PAL["edge"] if k % 2 else PAL["bulk"],
            lw=0.9,
            alpha=fade,
            zorder=2,
        )
    glow = 0.35 + 0.65 * _ease(t)
    for s_, a in ((2200 * glow, 0.14), (700 * glow, 0.30), (160 * glow, 0.7), (40, 0.95)):
        ax.scatter([cx], [cy], s=s_, c=PAL["collapse"], alpha=a, lw=0, zorder=7)
    ax.scatter([cx], [cy], s=10, c="#FFE3EA", alpha=1.0, zorder=8)


STAGE_X = (0.20, 0.50, 0.80)


def draw_dataflow(ax, t: float) -> None:
    _grid(ax, alpha=0.7)
    lane = 0.5
    for sx, col in zip(STAGE_X, (PAL["bulk"], PAL["spike"], PAL["edge"])):
        ax.scatter([sx], [lane], s=1500, c=col, alpha=0.05, lw=0, zorder=3)
        ax.scatter([sx], [lane], s=420, c=col, alpha=0.10, lw=0, zorder=3)
        th = np.linspace(0, 2 * np.pi, 120)
        ax.plot(
            sx + 0.035 * np.cos(th),
            lane + 0.055 * np.sin(th),
            color=col,
            lw=1.6,
            alpha=0.7,
            zorder=4,
        )
        ax.scatter([sx], [lane], s=26, c=col, alpha=0.95, lw=0, zorder=5)
    for a, b in ((STAGE_X[0], STAGE_X[1]), (STAGE_X[1], STAGE_X[2])):
        ax.annotate(
            "",
            xy=(b - 0.055, lane),
            xytext=(a + 0.055, lane),
            arrowprops={"arrowstyle": "-|>", "color": PAL["grid"], "lw": 1.4, "alpha": 0.9},
            zorder=2,
        )
    rng = np.random.default_rng(20261109)
    n = 110
    ph = rng.random(n)
    off = rng.uniform(-0.045, 0.045, n)
    sz = rng.uniform(4, 16, n)
    x = 0.05 + ((ph + t) % 1.0) * 0.90
    y = lane + off + 0.018 * np.sin(2 * np.pi * (x * 3 + ph * 7))
    stops = np.array([0.0, 0.5, 1.0])
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("flow", [PAL["bulk"], PAL["spike"], PAL["edge"]])
    col = np.clip(np.interp((x - 0.05) / 0.90, stops, [0, 0.5, 1]), 0, 1)
    ax.scatter(x, y, s=sz * 6, c=cmap(col), alpha=0.08, lw=0, zorder=5)
    ax.scatter(x, y, s=sz, c=cmap(col), alpha=0.75, lw=0, zorder=6)


def render_frames(slot: str, outdir: str) -> int:
    cfg = CFG["anims"][slot]
    w, h = cfg["size"]
    n = _frames(slot)
    for i in range(n):
        t = i / n
        fig, ax = _canvas(w, h)
        DRAW[slot](ax, t)
        fig.canvas.draw()
        img = Image.frombuffer(
            "RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()
        ).convert("RGB")
        plt.close(fig)
        img.save(os.path.join(outdir, f"f{i:05d}.png"))
    return n


def encode(slot: str, outdir: str, n: int) -> None:
    if FFMPEG is None:
        raise RuntimeError("ffmpeg not found on PATH — install ffmpeg first")
    cfg = CFG["anims"][slot]
    fps = cfg["fps"]
    poster = cfg["poster"]
    os.makedirs(ANIM_DIR, exist_ok=True)
    webm = os.path.join(ANIM_DIR, f"{slot}-loop.webm")
    mp4 = os.path.join(ANIM_DIR, f"{slot}-loop.mp4")
    subprocess.run(  # noqa: S603 — ffmpeg, resolved via shutil.which
        [
            FFMPEG,
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            os.path.join(outdir, "f%05d.png"),
            "-c:v",
            "libvpx-vp9",
            "-crf",
            "34",
            "-b:v",
            "1.2M",
            "-an",
            webm,
        ],
        check=True,
    )
    subprocess.run(  # noqa: S603 — ffmpeg, resolved via shutil.which
        [
            FFMPEG,
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            os.path.join(outdir, "f%05d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "24",
            "-an",
            mp4,
        ],
        check=True,
    )
    Image.open(os.path.join(outdir, "f00000.png")).save(
        os.path.join(VISUALS_DIR, f"{poster}.avif"), format="AVIF", quality=62, speed=8
    )
    # GitHub strips <video> tags whose src is not a user-attachment URL, so
    # the README gets a downscaled GIF twin of every loop; the WebM/MP4 pair
    # above stays the canonical (smaller, smoother) asset for docs and links.
    gif = os.path.join(ANIM_DIR, f"{slot}-loop.gif")
    gif_fps = 12
    step = max(1, fps // gif_fps)
    thumb = []
    for i in range(0, n, step):
        im = Image.open(os.path.join(outdir, f"f{i:05d}.png"))
        im.thumbnail((800, 450), Image.LANCZOS)
        thumb.append(im)

    def write_gif(colors: int) -> None:
        pal = thumb[0].quantize(colors=colors)
        pal.save(
            gif,
            format="GIF",
            save_all=True,
            append_images=[t.quantize(colors=colors) for t in thumb[1:]],
            duration=int(1000 / gif_fps),
            loop=0,
            optimize=True,
        )

    write_gif(192)
    if os.path.getsize(gif) > 2048 * 1024:  # inline GIF budget: 2 MB
        write_gif(128)


# --------------------------------------------------------- cli-demo.gif ----
# Real output of `rmt-llm-verify verify` (captured on this tree), rendered
# as (text, color) segments per line.
CYAN, AMBER, VIOLET, WHITE = PAL["bulk"], PAL["spike"], PAL["edge"], "#E8EEF7"
TERMINAL_LINES = [
    [("$ ", CYAN), ("rmt-llm-verify verify", WHITE)],
    [("   [OK]", AMBER), (" marchenko_pastur:  density finite: [0.4675, 1.7325]", WHITE)],
    [("   [OK]", AMBER), (" bbp_transition:    theta_c=0.3162 lambda_max=41.0", WHITE)],
    [("   [OK]", AMBER), (" tracy_widom:       mean=-1.7711 var=0.8132", WHITE)],
    [("   [OK]", AMBER), (" nhse:              wind=0 skin=0.0000 collapse=0.0000", WHITE)],
    [("   [OK]", AMBER), (" circular_ensembles: n=128 mean_spacing=1.0000", WHITE)],
    [("verify: 5/5 checks passed", VIOLET)],
]


def render_cli_demo(path: str) -> None:
    """Type `rmt-llm-verify verify`, reveal the five [OK] lines, hold, loop."""
    cfg = CFG["anims"]["cli-demo"]
    w, h = cfg["size"]
    fps = cfg["fps"]
    cmd = TERMINAL_LINES[0][1][0]
    type_frames = len(cmd) + 4
    reveal_frames = (len(TERMINAL_LINES) - 1) * 5
    hold = 18
    total = type_frames + reveal_frames + hold
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 30)
    pad, line_h, m = 46, 46, 60
    bg, bar = (10, 12, 18), (26, 32, 46)
    frames = []
    for i in range(total):
        img = Image.new("RGB", (w, h), _rgb(PAL["background"]))
        dr = ImageDraw.Draw(img)
        dr.rounded_rectangle(
            (m, m, w - m, h - m), radius=26, fill=bg, outline=_rgb(PAL["grid"]), width=2
        )
        dr.rounded_rectangle((m, m, w - m, m + 64), radius=26, fill=bar)
        dr.rectangle((m, m + 40, w - m, m + 64), fill=bar)
        for k, c in enumerate(("#FF5F57", "#FEBC2E", "#28C840")):
            dr.ellipse((m + 30 + k * 44, m + 20, m + 52 + k * 44, m + 42), fill=c)
        typed = min(i, len(cmd))
        visible = [[("$ ", CYAN), (cmd[:typed], WHITE)]]
        revealed = 0
        if i > type_frames:
            revealed = min((i - type_frames) // 5 + 1, len(TERMINAL_LINES) - 1)
            visible.extend(TERMINAL_LINES[1 : 1 + revealed])
        y = m + 64 + pad // 2
        cursor_x, cursor_y = m + pad, y
        for parts in visible:
            x = m + pad
            for text, col in parts:
                if text:
                    dr.text((x, y), text, font=font, fill=_rgb(col))
                    x += dr.textlength(text, font=font)
            cursor_x, cursor_y = x + 4, y
            y += line_h
        if (i // 3) % 2 == 0:
            dr.rectangle(
                (cursor_x, cursor_y + 8, cursor_x + 14, cursor_y + line_h - 10),
                fill=_rgb(PAL["bulk"]),
            )
        frames.append(img)
    frames[0].save(
        path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=True,
    )


DRAW = {
    "hero": draw_hero,
    "thread-1": draw_thread1,
    "thread-2": draw_thread2,
    "thread-3": draw_thread3,
    "dataflow": draw_dataflow,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--slot", action="append", choices=[*sorted(DRAW), "cli-demo"])
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    slots = args.slot or ([*sorted(DRAW), "cli-demo"] if args.all else None)
    if not slots:
        ap.error("pass --all or --slot NAME")
    rc = 0
    for slot in slots:
        if slot == "cli-demo":
            os.makedirs(ANIM_DIR, exist_ok=True)
            out = os.path.join(ANIM_DIR, "cli-demo.gif")
            render_cli_demo(out)
            kb = os.path.getsize(out) / 1024
            print(f"cli-demo.gif written: {kb:.0f} KB")
            if kb > 2048:
                print("  !! over 2 MB GIF budget", file=sys.stderr)
                rc = 1
            continue
        n = _frames(slot)
        tmp = tempfile.mkdtemp(prefix=f"anim-{slot}-")
        try:
            print(f"{slot}: rendering {n} frames ...", flush=True)
            render_frames(slot, tmp)
            encode(slot, tmp, n)
            for ext in ("webm", "mp4", "gif"):
                p = os.path.join(ANIM_DIR, f"{slot}-loop.{ext}")
                kb = os.path.getsize(p) / 1024
                lim = 3072 if ext == "webm" else 4096
                flag = "" if kb <= lim else "  !! OVER BUDGET"
                if flag:
                    rc = 1
                print(f"  {slot}-loop.{ext}: {kb:.0f} KB{flag}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
