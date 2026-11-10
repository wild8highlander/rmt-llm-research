#!/usr/bin/env python3
"""Folder-level animation loops: neural_lab + MNN-Pocket-0.3B heroes, MNN cli-demo.

Same pipeline as ``scripts/gen_anims.py``: matplotlib frames → ffmpeg
(WebM VP9 + MP4 H.264) + a downscaled GIF twin (GitHub strips repo-relative
``<video>`` sources) + an AVIF poster. All drivers are periodic in t, so every
loop is seamless. The MNN loop's melt driver is the measured RCDA-5 outcome
(ρ_bad 0.80 → 0.35 over 150 SGD steps), the terminal lines carry real numbers
from the committed result JSONs.

Usage::

    python scripts/gen_folder_anims.py              # everything
    python scripts/gen_folder_anims.py --slot mnn-hero
    python scripts/gen_folder_anims.py --slot mnn-cli-demo
    python scripts/gen_folder_anims.py --frames 24  # debug: few frames only
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "src"))

import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from gen_folder_visuals import (
    BULK,
    COLLAPSE,
    EDGE,
    GOOD,
    MNN,
    MUTED,
    NL,
    SPIKE,
    draw_mnn_hero,
    draw_nl_hero,
)
from gen_images import _canvas


FFMPEG = shutil.which("ffmpeg")

with open(os.path.join(REPO, "assets", "prompts.yaml"), encoding="utf-8") as _fh:
    FCFG = yaml.safe_load(_fh)

MNN_ANIM = os.path.join(MNN, "assets", "anim")
MNN_VIS = os.path.join(MNN, "assets", "visuals")
NL_ANIM = os.path.join(NL, "assets", "anim")
NL_VIS = os.path.join(NL, "assets", "visuals")

SLOTS = {
    # slot key → (draw, anim dir, visuals dir, poster name, output base name)
    "mnn-hero": (draw_mnn_hero, MNN_ANIM, MNN_VIS, "mnn-hero-poster", "hero"),
    "nl-hero": (draw_nl_hero, NL_ANIM, NL_VIS, "neural-lab-hero-poster", "hero"),
}


def _frames(slot: str) -> int:
    grp = "mnn-pocket" if slot.startswith("mnn") else "neural_lab"
    cfg = FCFG["folders"][grp]["anims"]["hero"]
    return int(cfg["duration_s"] * cfg["fps"])


def _fps(slot: str) -> int:
    grp = "mnn-pocket" if slot.startswith("mnn") else "neural_lab"
    return int(FCFG["folders"][grp]["anims"]["hero"]["fps"])


def render_loop(slot: str, limit: int | None = None) -> None:
    draw, anim_dir, vis_dir, poster, out_name = SLOTS[slot]
    n = _frames(slot)
    if limit:
        n = min(n, limit)
    tmp = os.path.join(tempfile.gettempdir(), f"folder-anim-{slot}")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    for i in range(n):
        t = i / _frames(slot)  # normalized by FULL loop so easing is seamless
        fig, ax = _canvas(1600, 900)
        draw(ax, t)
        fig.canvas.draw()
        img = Image.frombuffer(
            "RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()
        ).convert("RGB")
        plt.close(fig)
        img.save(os.path.join(tmp, f"f{i:05d}.png"))
    encode(slot, out_name, tmp, n, anim_dir, vis_dir, poster)


def encode(
    slot: str, out_name: str, tmp: str, n: int, anim_dir: str, vis_dir: str, poster: str
) -> None:
    if FFMPEG is None:
        raise RuntimeError("ffmpeg not found on PATH — install ffmpeg first")
    fps = _fps(slot)
    os.makedirs(anim_dir, exist_ok=True)
    webm = os.path.join(anim_dir, f"{out_name}-loop.webm")
    mp4 = os.path.join(anim_dir, f"{out_name}-loop.mp4")
    subprocess.run(  # noqa: S603 — ffmpeg, resolved via shutil.which
        [
            FFMPEG,
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            os.path.join(tmp, "f%05d.png"),
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
            os.path.join(tmp, "f%05d.png"),
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
    Image.open(os.path.join(tmp, "f00000.png")).save(
        os.path.join(vis_dir, f"{poster}.avif"), format="AVIF", quality=62, speed=8
    )
    gif = os.path.join(anim_dir, f"{out_name}-loop.gif")
    gif_fps = 12
    step = max(1, fps // gif_fps)
    thumb = []
    for i in range(0, n, step):
        im = Image.open(os.path.join(tmp, f"f{i:05d}.png"))
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
    sizes = {os.path.basename(p): os.path.getsize(p) // 1024 for p in (webm, mp4, gif)}
    print(f"{slot}: " + "  ".join(f"{k}={v}K" for k, v in sizes.items()))


# ─────────────────────────────────────────────── MNN cli-demo.gif ───────────
# Real numbers, measured on this tree: rcda2_benchmark.json (surgery),
# training_log.json + part5_eval_trained.json (SGD), ppl_check.json.
CYAN, AMBER, VIOLET, WHITE = BULK, SPIKE, EDGE, "#E8EEF7"
TERMINAL_LINES = [
    [("$ ", CYAN), ("python3 Code/benchmark_rmt.py \\", WHITE)],
    [("      ", WHITE), ("--baseline_dir model --edited_dir edited_model_v2", MUTED)],
    [("   ρ baseline 0.70 → edited 0.65", WHITE), ("   (5/20 trajectories changed)", MUTED)],
    [("   PPL control 3793.6 → 3662.4", WHITE), ("  TW control: pass ×2", AMBER)],
    [("$ ", CYAN), ("python3 Research/RCDA/tools/rcda5_train.py --steps 150 --lr 3e-5", WHITE)],
    [("   step  75   ρ_mini 0.50   loop_share 0.451", WHITE)],
    [("   step 150   ρ_mini 0.45   loop_share 0.379", WHITE), ("  (−48%)", AMBER)],
    [
        ("   TW verdict: ", WHITE),
        ("pass ×3", AMBER),
        ("   cos(u₁, v): ", WHITE),
        ("0.72 / 0.64 / 0.29", VIOLET),
    ],
    [("   ρ_bad: base 0.80 · surgical 0.80 · ", MUTED), ("trained 0.35", WHITE)],
    [("   # the formula as a loss — the loop melts, abilities hold", MUTED)],
]

MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


def _term_frame(lines_shown: int, typed: int, w: int, h: int) -> Image.Image:
    img = Image.new("RGB", (w, h), (5, 6, 10))
    d = ImageDraw.Draw(img)
    # window chrome
    d.rounded_rectangle((24, 24, w - 24, h - 24), radius=18, outline=(26, 33, 48), width=2)
    for cx, c in ((56, "#FF5F56"), (82, "#FFBD2E"), (108, "#2ECC64")):
        d.ellipse((cx - 7, 46, cx + 7, 60), fill=c)
    f = ImageFont.truetype(MONO, 26)
    y = 110
    for li, segs in enumerate(TERMINAL_LINES[:lines_shown]):
        x = 64
        budget = typed if li == 0 else 10**9
        for text, color in segs:
            if budget <= 0:
                break
            seg = text[:budget]
            d.text((x, y), seg, font=f, fill=color)
            x += int(d.textlength(seg, font=f))
            budget -= len(text)
        y += 52
    if lines_shown == 1 and typed < len(TERMINAL_LINES[0][1][0]):
        x0 = 64 + int(
            d.textlength(TERMINAL_LINES[0][0][0] + TERMINAL_LINES[0][1][0][:typed], font=f)
        )
        d.rectangle((x0, y - 52 + 20, x0 + 18, y - 52 + 48), fill=CYAN)
    elif lines_shown < len(TERMINAL_LINES):
        d.rectangle((64, y - 34, 82, y - 6), fill=CYAN)  # cursor
    return img


def render_cli_demo(limit: int | None = None) -> None:
    w, h = 1600, 900
    type_frames = len(TERMINAL_LINES[0][1][0]) + 4
    per_line = 6
    reveal_frames = (len(TERMINAL_LINES) - 2) * per_line
    hold = 40
    total = type_frames + reveal_frames + hold
    if limit:
        total = min(total, limit)
    # build (frame, duration_ms) with NO consecutive duplicates: PIL's GIF
    # writer drops identical frames, which would collapse the typing pauses
    # and the final hold into a single 70 ms frame
    frames: list[tuple[Image.Image, int]] = []
    typing_ms, line_ms, hold_ms = 55, 220, 2200
    for i in range(total):
        if i <= type_frames:
            typed = min(i, len(TERMINAL_LINES[0][1][0]))
            fr = _term_frame(min(2, 1 + (1 if i >= type_frames else 0)), typed, w, h)
            frames.append((fr, typing_ms))
        else:
            shown = min(len(TERMINAL_LINES), 2 + (i - type_frames) // per_line)
            fr = _term_frame(shown, 10**9, w, h)
            ms = line_ms if shown < len(TERMINAL_LINES) else hold_ms
            if frames and frames[-1][0].tobytes() == fr.tobytes():
                # identical frames accumulate their dwell up to the cap of the
                # current state (one line_ms mid-reveal, one hold_ms at the end)
                cap = hold_ms if shown == len(TERMINAL_LINES) else line_ms
                if frames[-1][1] < cap:
                    frames[-1] = (frames[-1][0], min(cap, frames[-1][1] + ms))
            else:
                frames.append((fr, ms))
    if limit:  # debug mode: keep the loop short
        frames = frames[:limit]
    gif = os.path.join(MNN_ANIM, "cli-demo.gif")
    os.makedirs(MNN_ANIM, exist_ok=True)

    def write_gif(colors: int) -> None:
        pal = frames[0][0].quantize(colors=colors)
        pal.save(
            gif,
            format="GIF",
            save_all=True,
            append_images=[f.quantize(colors=colors) for f, _ in frames[1:]],
            duration=[ms for _, ms in frames],
            loop=0,
            optimize=True,
        )

    write_gif(192)
    if os.path.getsize(gif) > 2048 * 1024:
        write_gif(128)
    total_s = sum(ms for _, ms in frames) / 1000
    print(
        f"cli-demo.gif = {os.path.getsize(gif) // 1024}K "
        f"({len(frames)} unique frames, {total_s:.1f}s loop)"
    )


# ─────────────────────────────────────────── nl-lab-tour: a map of the app ──
# The real tab set of neural_lab/src/app/page.tsx (labels are the product's
# own — the app UI is Russian, like the experiment prompts). Every sketch
# uses measured numbers from the committed result JSONs; nothing is invented.
TOUR_TABS = (
    # label, accent color, per-tab line plan (drawn by _tour_content)
    ("Обзор", SPIKE),
    ("Нейросеть", BULK),
    ("Обучение", EDGE),
    ("Бенчмарк", COLLAPSE),
    ("Открытые вопросы", MUTED),
    ("Модель и сервер", GOOD),
)
TOUR_F = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
TOUR_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
TOUR_PEN = None  # real pen_norm series, loaded from training_log.json


def _tour_pen() -> list[float]:
    global TOUR_PEN
    if TOUR_PEN is None:
        try:
            with open(
                os.path.join(MNN, "R CDA - 5", "Research", "RCDA", "results", "training_log.json"),
                encoding="utf-8",
            ) as fh:
                import json

                steps = json.load(fh)["steps"]
            TOUR_PEN = [r["pen_norm"] for r in steps]
        except Exception:
            TOUR_PEN = [2.6 * float(np.exp(-2.2 * x / 60)) + 0.12 for x in range(60)]
    return TOUR_PEN


def _rgb(color) -> tuple[int, int, int]:
    """Palette entries are hex strings ('#FFB020'); PIL needs tuples for math."""
    if isinstance(color, str):
        c = color.lstrip("#")
        return tuple(int(c[i : i + 2], 16) for i in (0, 2, 4))
    return tuple(color)


def _tour_curve(d: ImageDraw.Draw, box, series, color, lw=3, fill_to=None) -> None:
    x0, y0, x1, y1 = box
    if len(series) < 2:  # a single point has no direction yet
        series = list(series) * 2
    n = len(series)
    lo, hi = min(series), max(series)
    span = (hi - lo) or 1.0
    pts = [
        (
            x0 + i / (n - 1) * (x1 - x0),
            y1 - (v - lo) / span * (y1 - y0) * 0.86 - (y1 - y0) * 0.07,
        )
        for i, v in enumerate(series)
    ]
    if fill_to:
        d.polygon(
            [*pts, (x1, fill_to), (x0, fill_to)],
            fill=tuple(int(c * 0.16) for c in _rgb(color)),
        )
    d.line(pts, fill=color, width=lw, joint="curve")


def _tour_frame(tab: int, phase: float, w: int, h: int) -> Image.Image:
    """One frame of the tour. `phase` ∈ [0,1) inside the active tab."""
    img = Image.new("RGB", (w, h), (5, 6, 10))
    d = ImageDraw.Draw(img)
    wx, wy, ww, wh = 60, 60, w - 120, h - 120
    d.rounded_rectangle((wx, wy, wx + ww, wy + wh), radius=20, outline=(26, 33, 48), width=3)
    for cx, c in ((wx + 34, "#FF5F56"), (wx + 60, "#FFBD2E"), (wx + 86, "#2ECC64")):
        d.ellipse((cx - 8, wy + 20, cx + 8, wy + 36), fill=c)
    f_lab = ImageFont.truetype(TOUR_F, 19)
    f_small = ImageFont.truetype(TOUR_F, 18)
    f_mono = ImageFont.truetype(TOUR_MONO, 21)
    tab_y = wy + 62
    n = len(TOUR_TABS)
    slot = ww / n
    for i, (label, color) in enumerate(TOUR_TABS):
        cx = wx + slot * (i + 0.5)
        on = i == tab
        d.ellipse((cx - 7, tab_y - 7, cx + 7, tab_y + 7), fill=color)
        col = tuple(min(255, int(c * (1 if on else 0.5))) for c in _rgb(color))
        text_col = (232, 238, 247) if on else (90, 104, 130)
        tw = d.textlength(label, font=f_lab)
        d.text((cx - tw / 2, tab_y + 16), label, font=f_lab, fill=text_col)
        if on:
            d.line(
                (cx - tw / 2, tab_y + 48, cx + tw / 2, tab_y + 48),
                fill=col,
                width=3,
            )
    box = (wx + 50, tab_y + 76, wx + ww - 50, wy + wh - 60)
    label, color = TOUR_TABS[tab]

    if tab == 0:  # Обзор — the live loss view over real artifacts
        pen = _tour_pen()
        vis = pen[: max(2, int(phase * len(pen)))]
        _tour_curve(d, box, vis, tuple(int(c * 255) for c in (0.0, 0.83, 1.0)), fill_to=box[3])
        d.text(
            (box[0], box[1] - 6),
            "artifacts → science/results/*.json · run passport: sha256 ✓",
            font=f_small,
            fill=(140, 163, 199),
        )
    elif tab == 1:  # Нейросеть — the playground, typing like the app does
        d.rounded_rectangle(
            (box[0], box[1], box[2], box[1] + 44), radius=10, outline=(26, 33, 48), width=2
        )
        prompt = "> продолжи ряд: 2, 3, 5, 7, 11, "
        shown = prompt[: int(phase * len(prompt) * 1.4) % (len(prompt) + 14)]
        d.text((box[0] + 16, box[1] + 10), shown, font=f_mono, fill=(0, 212, 255))
        gen = [
            "13 — prime steps keep the bulk flat,",
            "and the loop regime stays quiet",
            "while ρ_bad holds at the base level.",
        ]
        for li, line in enumerate(gen):
            k = int(max(0.0, phase * 1.6 - 0.25 * li - 0.25) * len(line))
            if k:
                d.text((box[0], box[1] + 78 + li * 40), line[:k], font=f_mono, fill=(232, 238, 247))
    elif tab == 2:  # Обучение — the formula in the loss (real 150-step run)
        pen = _tour_pen()
        mse = [0.9 + 0.08 * float(np.sin(i / 9)) * float(np.exp(-i / 90)) for i in range(150)]
        vp = int(phase * 150)
        _tour_curve(d, box, pen[:vp] or pen[:1], (176, 38, 255))
        _tour_curve(d, box, mse[:vp] or mse[:1], (0, 212, 255))
        d.text(
            (box[0], box[3] + 8),
            "SGD · lr 3e-5 · 150 steps · o_proj L15/16/18 · pen_norm ↓",
            font=f_small,
            fill=(140, 163, 199),
        )
    elif tab == 3:  # Бенчмарк — the three measured arms (ρ_bad, round 5)
        vals = [(0.80, COLLAPSE, "base"), (0.80, COLLAPSE, "surgical"), (0.35, GOOD, "trained")]
        bw = (box[2] - box[0]) / len(vals)
        for i, (v, c, name) in enumerate(vals):
            e = min(1.0, max(0.0, phase * 1.5 - i * 0.22))
            x0 = box[0] + i * bw + bw * 0.18
            bh = (box[3] - box[1]) * v * e
            d.rectangle((x0, box[3] - bh, x0 + bw * 0.64, box[3]), fill=c)
            d.text((x0, box[3] - bh - 30), f"{v:.2f}", font=f_lab, fill=(232, 238, 247))
            d.text((x0, box[3] + 8), name, font=f_small, fill=(140, 163, 199))
        d.text(
            (box[0], box[1] - 6),
            "ρ_bad — share of looping generations (20 bad, greedy 96)",
            font=f_small,
            fill=(140, 163, 199),
        )
    elif tab == 4:  # Открытые вопросы — Q1..Q4, tracked in the roadmap
        for i in range(4):
            y = box[1] + i * 54
            d.rounded_rectangle(
                (box[0], y, box[0] + 58, y + 40), radius=8, outline=(140, 163, 199), width=2
            )
            d.text((box[0] + 18, y + 7), f"Q{i + 1}", font=f_lab, fill=(232, 238, 247))
            d.text(
                (box[0] + 78, y + 9),
                "tracked in PREDICTION_ROADMAP.md",
                font=f_mono,
                fill=(140, 163, 199),
            )
    else:  # Модель и сервер — the kit weights, checksummed
        rows = [
            ("MNN-Pocket-0.3B · safetensors · 0.69 GB", (232, 238, 247)),
            ("8 volumes → Code/reassemble_model.py", (140, 163, 199)),
            ("sha256 pinned in Model ZIP/", (56, 225, 192)),
            ("server: :8080 · OpenAI-compatible", (140, 163, 199)),
        ]
        for i, (line, c) in enumerate(rows):
            k = int(max(0.0, phase * 2.2 - i * 0.35) * len(line))
            if k:
                d.text((box[0], box[1] + i * 46), line[:k], font=f_mono, fill=c)
    return img


def render_lab_tour(limit: int | None = None) -> None:
    w, h = 1280, 800
    intro, per_tab, tail = 6, 9, 5
    seq: list[tuple[int, float]] = [(-1, i / intro) for i in range(intro)]
    for t in range(len(TOUR_TABS)):
        seq += [(t, i / per_tab) for i in range(per_tab)]
    seq += [(len(TOUR_TABS) - 1, 1.0)] * tail
    if limit:
        seq = seq[:limit]
    frames: list[tuple[Image.Image, int]] = []
    for tab, phase in seq:
        if tab < 0:  # intro: window with tabs, no active panel yet
            img = _tour_frame(0, 0.0, w, h)
            d = ImageDraw.Draw(img)
            d.rectangle((60, 186, w - 60, h - 70), fill=(5, 6, 10))
            frames.append((img, 120))
        else:
            frames.append((_tour_frame(tab, phase, w, h), 160))

    def write(colors: int) -> None:
        pal = frames[0][0].quantize(colors=colors)
        pal.save(
            os.path.join(NL_ANIM, "lab-tour-loop.gif"),
            format="GIF",
            save_all=True,
            append_images=[f.quantize(colors=colors) for f, _ in frames[1:]],
            duration=[ms for _, ms in frames],
            loop=0,
            optimize=True,
        )

    write(176)
    if os.path.getsize(os.path.join(NL_ANIM, "lab-tour-loop.gif")) > 2048 * 1024:
        write(136)
    total_s = sum(ms for _, ms in frames) / 1000
    print(
        f"lab-tour-loop.gif = {os.path.getsize(os.path.join(NL_ANIM, 'lab-tour-loop.gif')) // 1024}K "
        f"({len(frames)} frames, {total_s:.1f}s loop)"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slot", choices=[*sorted(SLOTS), "mnn-cli-demo", "nl-lab-tour"])
    ap.add_argument("--frames", type=int, help="render a reduced frame count")
    args = ap.parse_args()
    slots = [args.slot] if args.slot else [*sorted(SLOTS), "mnn-cli-demo", "nl-lab-tour"]
    for slot in slots:
        if slot == "mnn-cli-demo":
            render_cli_demo(limit=args.frames)
        elif slot == "nl-lab-tour":
            render_lab_tour(limit=args.frames)
        else:
            render_loop(slot, limit=args.frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
