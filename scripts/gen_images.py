#!/usr/bin/env python3
"""Render the static visual layer from ``assets/prompts.yaml`` (procedural).

Every slot is drawn deterministically in the project palette
(#05060A background, cyan bulk, amber spike, violet edge, crimson collapse).
The science is not decoration: the Marchenko-Pastur curve and the BBP
outlier position are computed with ``src/rmt_llm`` itself, so the hero
image is the package running — same claim as the live banner GIF.

Outputs (per slot): ``<slot>-2x.avif`` / ``.webp`` / ``.png`` in
``assets/visuals/``. The AVIF/WebP files are budget-checked to 400 KB;
the PNG is a larger universal fallback (allowed up to 2 MB).

Usage::

    python scripts/gen_images.py --all
    python scripts/gen_images.py --slot hero
    python scripts/gen_images.py --slot hero --from-raw   # gen-AI raw input
"""

from __future__ import annotations

import argparse
import os
import sys


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, REPO)

import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFilter

from rmt_llm.bbp_transition import bbp_lambda_max
from rmt_llm.marchenko_pastur import mp_bounds, mp_density, mp_sample


CFG_PATH = os.path.join(REPO, "assets", "prompts.yaml")
OUT_DIR = os.path.join(REPO, "assets", "visuals")

with open(CFG_PATH, encoding="utf-8") as fh:
    CFG = yaml.safe_load(fh)

PAL = dict(CFG["palette"])


def _rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def _rng(slot: str) -> np.random.Generator:
    seed = int(CFG["static"][slot]["seed"])
    return np.random.default_rng(seed)


def _canvas(w: int, h: int) -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor(PAL["background"])
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=PAL["background"], zorder=-10))
    return fig, ax


def _grid(ax, alpha: float = 1.0, hexes: bool = False) -> None:
    g = np.array(_rgb(PAL["grid"])) / 255.0
    for x in np.linspace(0.05, 0.95, 15):
        ax.axvline(x, color=g, lw=0.7, alpha=0.30 * alpha, zorder=0)
    for y in np.linspace(0.08, 0.92, 9):
        ax.axhline(y, color=g, lw=0.7, alpha=0.30 * alpha, zorder=0)
    if hexes:
        for i, x in enumerate(np.linspace(0.06, 0.96, 13)):
            for y in np.linspace(0.10, 0.90, 7):
                r = 0.022
                th = np.linspace(0, 2 * np.pi, 7) + (0.4 if i % 2 else 0.0)
                ax.plot(
                    x + r * np.cos(th),
                    y + r * np.sin(th) * 0.9,
                    color=g,
                    lw=0.5,
                    alpha=0.20 * alpha,
                    zorder=0,
                )


def _dust(ax, slot: str, n: int = 220, alpha: float = 0.5) -> None:
    rng = _rng(slot)
    xs, ys = rng.random(n), rng.random(n)
    ss = rng.random(n) * 3 + 0.4
    ax.scatter(xs, ys, s=ss, c="#E8EEF7", alpha=0.16 * alpha, lw=0, zorder=1)


def _glow_line(ax, x, y, color, lw_core=2.2, zorder=6) -> None:
    for lw, al in ((lw_core * 7, 0.05), (lw_core * 3, 0.16), (lw_core, 0.95)):
        ax.plot(x, y, color=color, lw=lw, alpha=al, solid_capstyle="round", zorder=zorder)


def _spike(ax, x0, base, height, color, core_h=0.55) -> None:
    xs = np.linspace(0, 1, 60)
    for w, al in ((0.026, 0.05), (0.013, 0.13), (0.006, 0.38), (0.0022, 0.95)):
        ax.fill_betweenx(
            [base, base + height],
            x0 - w,
            x0 + w,
            color=color,
            alpha=al,
            lw=0,
            zorder=6,
        )
    tip_y = base + height * core_h
    ax.plot(
        x0 + 0.002 * np.sin(xs * 6),
        base + height * xs**1.4 * core_h,
        color="#FFFFFF",
        lw=1.1,
        alpha=0.55,
        zorder=7,
    )
    for s, a in ((320, 0.10), (90, 0.35), (18, 0.95)):
        ax.scatter([x0], [tip_y + height * 0.02], s=s, c="#FFE9C4", alpha=a, lw=0, zorder=8)


def _orbitals(ax, cx, cy, radii, color, alpha0=0.42, rot=18.0, lw=1.0) -> None:
    th = np.linspace(0, 2 * np.pi, 240)
    for i, r in enumerate(radii):
        a = alpha0 * (0.85**i)
        for lwj, al in ((lw * 3, a * 0.30), (lw, a)):
            ax.plot(
                cx + r * np.cos(th) * 1.35,
                cy + r * np.sin(th) * 0.62,
                color=color,
                lw=lwj,
                alpha=al,
                zorder=3,
                transform=(
                    matplotlib.transforms.Affine2D().rotate_deg_around(cx, cy, rot + i * 6)
                    + ax.transData
                ),
            )


def _bloom(img: Image.Image, radius: int = 14, strength: float = 0.42) -> Image.Image:
    blur = img.filter(ImageFilter.GaussianBlur(radius))
    arr = np.asarray(img).astype(np.float32)
    blr = np.asarray(blur).astype(np.float32)
    out = 1.0 - (1.0 - arr / 255.0) * (1.0 - (blr / 255.0) * strength)
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))


def _vignette(img: Image.Image, strength: float = 0.34) -> Image.Image:
    w, h = img.size
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2, h / 2
    d = np.sqrt(((xx - cx) / (w * 0.72)) ** 2 + ((yy - cy) / (h * 0.72)) ** 2)
    mask = np.clip(1.0 - strength * np.clip(d - 0.45, 0, None) ** 1.5, 0, 1)
    arr = np.asarray(img).astype(np.float32) * mask[..., None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _grain(img: Image.Image, slot: str, amount: float = 5.0) -> Image.Image:
    rng = _rng(slot)
    noise = rng.normal(0, amount, img.size[::-1])[..., None]
    arr = np.asarray(img).astype(np.float32) + noise
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _finish(fig, slot: str, bloom=True, grain_amt=2.5, vign=0.34) -> Image.Image:
    fig.canvas.draw()
    img = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert(
        "RGB"
    )
    plt.close(fig)
    if bloom:
        img = _bloom(img)
    if vign:
        img = _vignette(img, vign)
    if grain_amt:
        img = _grain(img, slot, grain_amt)
    return img


def _mp_curve(q: float = 0.1, sigma2: float = 1.0, n: int = 900):
    lo, hi = mp_bounds(q, sigma2)
    pad = (hi - lo) * 0.45
    lam = np.linspace(max(lo - pad, 1e-6), hi + pad * 1.6, n)
    return lam, mp_density(lam, q, sigma2), lo, hi


# ---------------------------------------------------------------- slots ----
def render_hero() -> Image.Image:
    fig, ax = _canvas(3200, 1800)
    _grid(ax, alpha=0.9)
    _dust(ax, "hero", n=260)
    _orbitals(ax, 0.80, 0.76, [0.045, 0.075, 0.11, 0.15], PAL["edge"], alpha0=0.55, rot=24)
    q, s2 = 0.1, 1.0
    lam, rho, _lo, _hi = _mp_curve(q, s2)
    span_lo, span_hi = 0.03, 0.97
    theta_c = float(np.sqrt(q))
    theta = 1.15 * theta_c  # just past the transition
    out = float(bbp_lambda_max(theta, q, s2))
    xmax = out * 1.30
    xs = span_lo + (lam / xmax) * (span_hi - span_lo)
    ys = 0.09 + (rho / rho.max()) * 0.40
    ax.fill_between(xs, 0.09, ys, color=PAL["bulk"], alpha=0.13, lw=0, zorder=2)
    _glow_line(ax, xs, ys, PAL["bulk"], lw_core=3.2)
    sx = span_lo + (out / xmax) * (span_hi - span_lo)
    _spike(ax, sx, 0.09, 0.46, PAL["spike"])
    ax.plot([span_lo, span_hi], [0.09, 0.09], color=PAL["grid"], lw=1.2, alpha=0.8, zorder=2)
    for s, a in ((2600, 0.05), (900, 0.09)):
        ax.scatter([0.45], [0.30], s=s, c=PAL["bulk"], alpha=a, lw=0, zorder=1)
    return _finish(fig, "hero", grain_amt=3.0)


def render_thread1() -> Image.Image:
    fig, ax = _canvas(1600, 1000)
    _grid(ax, alpha=0.8)
    _dust(ax, "thread-1-spectral", n=120)
    q, s2 = 0.3, 1.0
    lam, rho, _lo, hi = _mp_curve(q, s2)
    xs = 0.06 + (lam / (hi * 1.9)) * 0.80
    ys = 0.16 + (rho / rho.max()) * 0.42
    ax.fill_between(xs, 0.16, ys, color=PAL["bulk"], alpha=0.12, lw=0, zorder=2)
    _glow_line(ax, xs, ys, PAL["bulk"], lw_core=2.4)
    rng = _rng("thread-1-spectral")
    lam_s = mp_sample(300, 1000, s2, rng=rng)
    n_pts = lam_s.size
    px = 0.06 + (lam_s / (hi * 1.9)) * 0.80
    py = 0.16 + rng.random(n_pts) * 0.30 * (np.interp(lam_s, lam, rho) / rho.max() + 0.06)
    ax.scatter(px, py, s=1.4, c=PAL["bulk"], alpha=0.22, lw=0, zorder=3)
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
    flick = _rng("thread-1-spectral").uniform(0.16, 0.80, 52)
    ax.scatter(
        np.full(52, edge_x), flick, s=2.2, c=PAL["edge"], alpha=0.30, marker="|", lw=1.6, zorder=5
    )
    ax.scatter(
        np.full(52, edge_x), flick, s=26, c=PAL["edge"], alpha=0.10, marker="|", lw=1.2, zorder=5
    )
    _spike(ax, edge_x + 0.045, 0.30, 0.34, PAL["spike"], core_h=0.5)
    return _finish(fig, "thread-1-spectral", grain_amt=3.0, vign=0.30)


def render_thread2() -> Image.Image:
    fig, ax = _canvas(1600, 1000)
    _grid(ax, alpha=0.75)
    _dust(ax, "thread-2-hallucination", n=110)
    cx, cy = 0.30, 0.52
    _orbitals(ax, cx, cy, [0.05, 0.085, 0.12, 0.155, 0.19], PAL["edge"], alpha0=0.5, rot=14)
    ax.scatter([cx], [cy], s=340, c="#000000", zorder=4)
    ax.scatter([cx], [cy], s=60, c="#E8EEF7", alpha=0.9, zorder=5)
    ax.plot([0.62, 0.62], [0.08, 0.92], color="#E8EEF7", lw=1.6, alpha=0.55, zorder=4)
    ax.plot([0.628, 0.628], [0.08, 0.92], color=PAL["grid"], lw=5, alpha=0.35, zorder=3)
    t = np.linspace(0, 1, 120)
    bx = cx + 0.10 + (0.62 - cx - 0.14) * t + 0.055 * np.sin(t * np.pi) * 0.9
    by = cy + 0.16 * np.sin(t * np.pi) + 0.045 * t * 2.4
    _glow_line(ax, bx, by, PAL["spike"], lw_core=2.2)
    ax.scatter([bx[-1]], [by[-1]], s=42, c=PAL["spike"], alpha=0.95, zorder=7)
    ax.annotate(
        "",
        xy=(bx[-1] + 0.028, by[-1] + 0.045),
        xytext=(bx[-1], by[-1]),
        arrowprops={"arrowstyle": "-|>", "color": PAL["spike"], "lw": 2.4, "alpha": 0.95},
        zorder=7,
    )
    rng = _rng("thread-2-hallucination")
    n = 420
    gx = rng.normal(0.70, 0.05, n)
    gy = rng.normal(0.56, 0.10, n) * (1 + 0.4 * np.cos(gx * 9))
    ax.scatter(gx, gy, s=190, c=PAL["collapse"], alpha=0.05, lw=0, zorder=5)
    ax.scatter(gx, gy, s=26, c=PAL["collapse"], alpha=0.10, lw=0, zorder=5)
    ax.scatter(gx, gy, s=5, c=PAL["collapse"], alpha=0.38, lw=0, zorder=6)
    return _finish(fig, "thread-2-hallucination", grain_amt=3.0, vign=0.30)


def render_thread3() -> Image.Image:
    fig, ax = _canvas(1600, 1000)
    _grid(ax, alpha=0.7)
    _dust(ax, "thread-3-rlhf", n=100)
    rng = _rng("thread-3-rlhf")
    cx, cy = 0.56, 0.50
    n = 2600
    phi = rng.random(n) * 4.2 * np.pi
    r = 0.030 + 0.34 * np.exp(-phi / 7.5)
    jitter = rng.normal(0, 0.016, n)
    px = cx + (r + jitter) * np.cos(phi)
    py = cy + (r + jitter) * np.sin(phi) * 0.62
    col_mix = np.clip(r / r.max(), 0, 1)
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("cv", [PAL["collapse"], PAL["edge"], PAL["bulk"]])
    ax.scatter(px, py, s=90, c=col_mix, cmap=cmap, alpha=0.06, lw=0, zorder=3)
    ax.scatter(px, py, s=14, c=col_mix, cmap=cmap, alpha=0.16, lw=0, zorder=3)
    ax.scatter(px, py, s=3.2, c=col_mix, cmap=cmap, alpha=0.75, lw=0, zorder=4)
    for k, fade in enumerate((0.45, 0.30, 0.18, 0.10)):
        ph = np.linspace(0, 4.6 * np.pi, 300)
        rr = 0.02 + 0.36 * np.exp(-ph / (6.0 - k * 0.7))
        ax.plot(
            cx + rr * np.cos(ph + 0.6 * k),
            cy + rr * np.sin(ph + 0.6 * k) * 0.62,
            color=PAL["edge"] if k % 2 else PAL["bulk"],
            lw=0.9,
            alpha=fade,
            zorder=2,
        )
    core = np.hypot(px - cx, (py - cy) / 0.62)
    inner = core < 0.055
    ax.scatter(px[inner], py[inner], s=4.2, c=PAL["collapse"], alpha=0.9, lw=0, zorder=6)
    for s, a in ((2200, 0.14), (700, 0.30), (160, 0.7), (40, 0.95)):
        ax.scatter([cx], [cy], s=s, c=PAL["collapse"], alpha=a, lw=0, zorder=7)
    ax.scatter([cx], [cy], s=10, c="#FFE3EA", alpha=1.0, zorder=8)
    return _finish(fig, "thread-3-rlhf", grain_amt=3.0, vign=0.32)


def render_architecture() -> Image.Image:
    fig, ax = _canvas(1600, 900)
    _grid(ax, alpha=1.0, hexes=True)
    for yy, al in ((0.30, 0.10), (0.52, 0.07), (0.74, 0.09)):
        x = np.linspace(0, 1, 500)
        ax.plot(
            x, yy + 0.012 * np.sin(x * 21 + yy * 40), color=PAL["bulk"], lw=1.0, alpha=al, zorder=1
        )
    fig.canvas.draw()
    img = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert(
        "RGB"
    )
    plt.close(fig)
    return _bloom(img, 8, 0.25)


def render_og_base() -> Image.Image:
    fig, ax = _canvas(1200, 630)
    _grid(ax, alpha=0.85)
    q, s2 = 0.1, 1.0
    lam, rho, _lo, _hi = _mp_curve(q, s2)
    theta = 0.55
    out = float(bbp_lambda_max(theta, q, s2))
    xmax = out * 1.15
    xs = 0.34 + (lam / xmax) * 0.60
    ys = 0.12 + (rho / rho.max()) * 0.30
    ax.fill_between(xs, 0.12, ys, color=PAL["bulk"], alpha=0.10, lw=0, zorder=2)
    _glow_line(ax, xs, ys, PAL["bulk"], lw_core=2.6)
    sx = 0.34 + (out / xmax) * 0.60
    _spike(ax, sx, 0.12, 0.36, PAL["spike"])
    _orbitals(ax, 0.86, 0.72, [0.05, 0.085, 0.12], PAL["edge"], rot=20)
    ax.plot([0.02, 0.98], [0.12, 0.12], color=PAL["grid"], lw=1.0, alpha=0.8, zorder=2)
    return _finish(fig, "og-base", grain_amt=3.0, vign=0.30)


def render_favicon() -> Image.Image:
    w = h = 512
    img = Image.new("RGB", (w, h), _rgb(PAL["background"]))
    dr = ImageDraw.Draw(img)
    dr.rounded_rectangle((6, 6, w - 6, h - 6), radius=96, outline=_rgb(PAL["grid"]), width=3)
    cyan, amber, violet = _rgb(PAL["bulk"]), _rgb(PAL["spike"]), _rgb(PAL["edge"])
    dr.ellipse((w // 2 - 150, h // 2 - 60, w // 2 + 150, h // 2 + 60), fill=cyan)
    dr.rectangle((w // 2 - 150, h // 2, w // 2 + 150, h // 2 + 2), fill=cyan)
    dr.rectangle((w // 2 - 13, h // 2 - 190, w // 2 + 13, h // 2), fill=amber)
    dr.ellipse((w // 2 - 11, h // 2 - 212, w // 2 + 11, h // 2 - 190), fill=amber)
    dr.arc(
        (w // 2 - 172, h // 2 - 172, w // 2 + 172, h // 2 + 172),
        start=200,
        end=340,
        fill=violet,
        width=9,
    )
    return img


def render_divider(kind: str) -> Image.Image:
    pairs = {
        "divider-thread-1": ("bulk", "edge"),
        "divider-thread-2": ("edge", "spike"),
        "divider-thread-3": ("spike", "collapse"),
    }
    c1, c2 = (_rgb(PAL[pairs[kind][0]]), _rgb(PAL[pairs[kind][1]]))
    w, h = 1600, 80
    img = Image.new("RGB", (w, h), _rgb(PAL["background"]))
    grad = np.linspace(0, 1, w)[:, None]
    line = (1 - grad) * np.array(c1)[None, None, :] + grad * np.array(c2)[None, None, :]
    arr = np.asarray(img).astype(np.float32)
    band = slice(h // 2 - 2, h // 2 + 2)
    arr[band, :, :] = line[0, :, :]
    glow = np.clip(line * 0.5, 0, 255)
    for off, k in ((5, 0.5), (10, 0.22), (16, 0.08)):
        arr[slice(h // 2 - 2 - off, h // 2 + 2 + off), :, :] = np.maximum(
            arr[slice(h // 2 - 2 - off, h // 2 + 2 + off), :, :], glow * k
        )
    rng = _rng(kind)
    for x in rng.integers(0, w, 230):
        hh = int(rng.integers(3, 18))
        al = float(rng.uniform(0.16, 0.55))
        c = arr[h // 2 - 2, x].astype(np.float32) * (0.5 + al)
        arr[h // 2 - hh : h // 2 + hh, x] = np.clip(c, 0, 255)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


RENDERERS = {
    "hero": render_hero,
    "thread-1-spectral": render_thread1,
    "thread-2-hallucination": render_thread2,
    "thread-3-rlhf": render_thread3,
    "architecture": render_architecture,
    "favicon": render_favicon,
    "divider-thread-1": lambda: render_divider("divider-thread-1"),
    "divider-thread-2": lambda: render_divider("divider-thread-2"),
    "divider-thread-3": lambda: render_divider("divider-thread-3"),
}

# Slots that skip the PNG fallback: the AVIF -> WebP <picture> chain covers
# every browser GitHub itself supports, and skipping PNG keeps the visual
# directory inside its 5 MB budget. `og-base` is rendered on demand by
# scripts/compose_og.py (import), so it is not an exported slot at all.
NO_PNG = {"thread-1-spectral", "thread-2-hallucination", "thread-3-rlhf"}

FORMATS = ("avif", "webp", "png")
KWARGS = {
    "avif": {"quality": 58, "speed": 8},
    "webp": {"quality": 84, "method": 6},
    "png": {"optimize": True},
}


def _export(img: Image.Image, slot: str) -> list[tuple[str, int]]:
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    png_img = img
    if img.width > 1600:
        # PNG is the universal fallback only — cap it at the 1x size so the
        # lossless format never dominates the repo budget.
        png_img = img.resize((1600, img.height * 1600 // img.width), Image.LANCZOS)
    for fmt in FORMATS:
        if fmt == "png" and slot in NO_PNG:
            continue
        path = os.path.join(OUT_DIR, f"{slot}-2x.{fmt}")
        if fmt == "png":
            png_img.save(path, format="PNG", optimize=True)
        else:
            for q in (KWARGS[fmt]["quality"], 76, 70, 62):
                params = {**KWARGS[fmt], "quality": q}
                img.save(path, format=fmt.upper(), **params)
                if os.path.getsize(path) <= 400 * 1024:
                    break
        rows.append((path, os.path.getsize(path)))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--slot", action="append", choices=sorted(RENDERERS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument(
        "--from-raw", action="store_true", help="use assets/visuals/<slot>-raw.png as source"
    )
    args = ap.parse_args()
    slots = args.slot or (list(RENDERERS) if args.all else None)
    if not slots:
        ap.error("pass --all or --slot NAME")
    print(f"{'slot':<26}{'avif KB':>9}{'webp KB':>9}{'png KB':>8}")
    rc = 0
    for slot in slots:
        if args.from_raw:
            raw = os.path.join(OUT_DIR, f"{slot}-raw.png")
            img = Image.open(raw).convert("RGB")
        else:
            img = RENDERERS[slot]()
        _export(img, slot)
        sizes = {
            f: os.path.getsize(os.path.join(OUT_DIR, f"{slot}-2x.{f}")) / 1024
            for f in FORMATS
            if os.path.exists(os.path.join(OUT_DIR, f"{slot}-2x.{f}"))
        }
        print(f"{slot:<26}{sizes['avif']:>8.0f}K{sizes['webp']:>8.0f}K{sizes.get('png', 0):>7.0f}K")
        for f in ("avif", "webp"):
            if sizes[f] > 400:
                print(f"  !! {slot}-2x.{f} over 400 KB budget", file=sys.stderr)
                rc = 1
        if sizes.get("png", 0) > 2048:
            print(f"  !! {slot}-2x.png over 2 MB fallback cap", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
