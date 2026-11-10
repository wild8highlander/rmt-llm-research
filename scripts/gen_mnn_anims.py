#!/usr/bin/env python3
"""Animated data GIFs for the MNN kit: the RCDA story, drawn from the JSONs.

Three loops in ``neural_lab/MNN-Pocket-0.3B/assets/anim/``, each an animated
counterpart of a static figure, using the *same data mappings* as
``scripts/gen_mnn_figures.py`` and the *committed* result JSONs of the
canonical snapshot (R CDA - 5) — no model, no re-running of experiments:

  scans-loop.gif     fig6  — matrix scans sweep (bars grow, supercritical flash)
  dose-loop.gif      fig8  — dose-response draw (KL line, dose bars, rep-4)
  training-loop.gif  fig15b — SGD training sweep (pen_norm/mse draw, bench pops)

Frames render through the same rcParams as the static figures, are assembled
with PIL (palette GIF, per-frame durations, duplicate-frame-safe) and are
budget-checked like every other asset (GIF ≤ 2 MB).

Usage::

    python scripts/gen_mnn_anims.py              # everything
    python scripts/gen_mnn_anims.py --only scans
"""

from __future__ import annotations

import argparse
import os
import sys


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from gen_mnn_figures import (
    BULK,
    COLLAPSE,
    EDGE,
    INK,
    MNN,
    MUTED,
    SPIKE,
    _load,
)


R5 = os.path.join(MNN, "R CDA - 5", "Research", "RCDA")
OUT = os.path.join(MNN, "assets", "anim")


def _series_head(
    xs_all: list[float], ys_all: list[float], head: float, i: int, frac: float
) -> tuple[list[float], list[float]]:
    """Prefix of a series up to `head`, with the last segment linearly interpolated."""
    xs, ys = xs_all[: i + 1], ys_all[: i + 1]
    if i + 1 < len(ys_all):
        tip = ys_all[i] + frac * (ys_all[i + 1] - ys_all[i])
        xs, ys = [*xs, head], [*ys, tip]
    return xs, ys


def _shot(fig: plt.Figure) -> Image.Image:
    fig.canvas.draw()
    return Image.frombuffer(
        "RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()
    ).convert("RGB")


def _write_gif(path: str, frames: list[Image.Image], durations: list[int]) -> None:
    # PIL's GIF writer drops consecutive identical frames; dedup explicitly and
    # keep every kept frame's own duration (the lesson from cli-demo.gif)
    keep: list[Image.Image] = []
    keep_ms: list[int] = []
    for fr, ms in zip(frames, durations):
        if keep and keep[-1].tobytes() == fr.tobytes():
            keep_ms[-1] += ms
        else:
            keep.append(fr)
            keep_ms.append(ms)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def write(colors: int) -> None:
        pal = keep[0].quantize(colors=colors)
        pal.save(
            path,
            format="GIF",
            save_all=True,
            append_images=[f.quantize(colors=colors) for f in keep[1:]],
            duration=keep_ms,
            loop=0,
            optimize=True,
        )

    for colors in (176, 144, 112, 96):
        write(colors)
        size = os.path.getsize(path)
        if size <= 2048 * 1024:
            break
    total = sum(keep_ms) / 1000
    print(
        f"  {os.path.relpath(path, REPO)}  {os.path.getsize(path) // 1024}K "
        f"({len(keep)} frames, {total:.1f}s, {colors} colors)"
    )


def _check(path: str) -> None:
    im = Image.open(path).convert("L")
    hist = im.histogram()
    lit = sum(hist[:250]) / sum(hist)
    assert lit > 0.01, f"GIF looks blank: {path}"
    assert os.path.getsize(path) <= 2048 * 1024, f"GIF over budget: {path}"


def _ease(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


# ───────────────────────────────────── scans-loop: fig6, bars grow + flash ──
def anim_scans() -> bool:
    rep = _load(os.path.join(R5, "results", "rcda_probe_report.json"))
    if not rep or not rep.get("matrix_scans"):
        print("  scans: rcda_probe_report.json missing — skipped")
        return False
    scans = rep["matrix_scans"]
    n = len(scans)
    vals = np.array([s["scan"]["lambda_over_edge"] for s in scans])
    crit = [bool(s["scan"]["supercritical"]) for s in scans]
    labels = [f"L{s['layer']}\n{s['matrix']}" for s in scans]
    vmax = max(1.15, float(vals.max()) * 1.10)

    grow, hold, ms = 56, 7, 80
    frames: list[Image.Image] = []
    durations: list[int] = []
    for f in range(grow + hold):
        p = min(1.0, f / (grow - 1)) if f < grow else 1.0
        fig, ax = plt.subplots(figsize=(9.4, 4.7), dpi=100)
        locs = []
        for k in range(n):
            start = 0.78 * k / max(1, n - 1)
            local = _ease((p - start) / 0.22)
            h = float(vals[k]) * local
            ax.bar(k, h, color=COLLAPSE if crit[k] else BULK, width=0.62)
            if 0.97 <= local < 1.0:  # completion flash: bright cap on the bar
                ax.bar(k, 0.012 * vmax, bottom=h, color=INK, width=0.62)
            locs.append(local)
        ax.axhline(
            1.0,
            color=INK,
            ls="--",
            lw=1.2,
            alpha=_ease((p - 0.72) / 0.15),
        )
        if p > 0.72:
            ax.text(-0.35, 1.06, r" $\lambda_v = \lambda_+$", fontsize=9, color=INK)
        ax.set_xticks(range(n), labels, fontsize=8)
        ax.set_ylim(0, vmax)
        ax.set_xlim(-0.7, n - 0.3)
        ax.set_ylabel(r"$\lambda_v / \lambda_+^{(W)}$")
        ax.set_title(
            "Matrix scans: projection mass of $\\Delta h$ vs the MP edge "
            "(crimson = supercritical) — sweep"
        )
        ax.annotate(
            "data: R CDA - 5/Research/RCDA/results/rcda_probe_report.json (round 1)",
            (0.995, 0.005),
            xycoords="axes fraction",
            ha="right",
            va="bottom",
            fontsize=6.5,
            color=MUTED,
        )
        frames.append(_shot(fig))
        done = all(x >= 1.0 for x in locs)
        durations.append(300 if f >= grow else (160 if done else ms))
        plt.close(fig)
    out = os.path.join(OUT, "scans-loop.gif")
    _write_gif(out, frames, durations)
    _check(out)
    return True


# ─────────────────────────────── dose-loop: fig8, KL draws + bars rise ──────
def anim_dose() -> bool:
    dose = _load(os.path.join(R5, "results", "rcda2_dose.json"))
    if not dose:
        er = _load(os.path.join(R5, "results", "rcda2_edit_report.json"))
        if er and er.get("meta", {}).get("dose_response"):
            m = er["meta"]
            dose = {"base_cal": m.get("base_cal"), "dose_response": m.get("dose_response")}
    if not dose:
        print("  dose: rcda2_dose.json missing — skipped")
        return False
    dd = dose["dose_response"]
    al = [d["alpha"] for d in dd]
    kl = [d["kl"] for d in dd]
    bad = [d["changed"]["bad"] for d in dd]
    good = [d["changed"]["good"] for d in dd]
    rep4 = [d["rep4_bad"] for d in dd]
    base_rep4 = dose["base_cal"]["rep4_bad"]
    npts = len(al)

    grow, hold, ms = 56, 7, 80
    frames: list[Image.Image] = []
    durations: list[int] = []
    for f in range(grow + hold):
        p = min(1.0, f / (grow - 1)) if f < grow else 1.0
        fig, ax1 = plt.subplots(figsize=(9.4, 4.7), dpi=100)
        head = p * (npts - 1)
        i_full = int(head)
        frac = head - i_full
        xs = al[: i_full + 1]
        ys = kl[: i_full + 1]
        if i_full + 1 < npts:
            tip = kl[i_full] + frac * (kl[i_full + 1] - kl[i_full])
            xs, ys = [*xs, al[i_full + 1]], [*ys, tip]
        ax1.plot(xs, ys, "o-", color=EDGE, ms=4, label="mean KL(base||edited)")
        ax1.set_xlim(-0.02, 1.06)
        ax1.set_xlabel(r"ablation strength $\alpha$")
        ax1.set_ylabel("KL", color=EDGE)
        ax1.tick_params(axis="y", labelcolor=EDGE)
        ax2 = ax1.twinx()
        ax2.spines["right"].set_visible(True)
        w = 0.03
        for j in range(npts):
            local = _ease((p - (0.14 + 0.15 * j)) / 0.12)
            if local <= 0:
                continue
            ax2.bar(al[j] - w, bad[j] * local, width=2 * w, color=COLLAPSE, alpha=0.75)
            ax2.bar(al[j] + w, good[j] * local, width=2 * w, color=SPIKE, alpha=0.75)
        r_head = max(0.0, (p - 0.18) / 0.82) * (npts - 1)
        ri = int(r_head)
        rfrac = r_head - ri
        rxs = al[: ri + 1]
        rys = rep4[: ri + 1]
        if ri + 1 < npts:
            tip = rep4[ri] + rfrac * (rep4[ri + 1] - rep4[ri])
            rxs, rys = [*rxs, al[ri + 1]], [*rys, tip]
        ax2.plot(rxs, rys, "s--", color=BULK, ms=4, label="mean RepRate-4 (bad)")
        ax2.axhline(
            base_rep4,
            color=BULK,
            ls=":",
            lw=1,
            alpha=_ease((p - 0.5) / 0.2),
            label="clean baseline RepRate-4",
        )
        ax2.set_ylabel("trajectories changed / rep4", color=COLLAPSE)
        ax2.tick_params(axis="y", labelcolor=COLLAPSE)
        if p > 0.1:
            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax1.legend(h1 + h2, l1 + l2, loc="upper left")
        ax1.set_title("Dose-response of the behavioral RCDA edit — draw")
        ax1.annotate(
            "data: R CDA - 5/Research/RCDA/results/rcda2_dose.json (round 2)",
            (0.995, 0.005),
            xycoords="axes fraction",
            ha="right",
            va="bottom",
            fontsize=6.5,
            color=MUTED,
        )
        frames.append(_shot(fig))
        durations.append(300 if f >= grow else ms)
        plt.close(fig)
    out = os.path.join(OUT, "dose-loop.gif")
    _write_gif(out, frames, durations)
    _check(out)
    return True


# ── training-loop: fig15(b), the formula in the loss over 150 real steps ────
def anim_training() -> bool:
    log = _load(os.path.join(R5, "results", "training_log.json"))
    if not log:
        print("  training: training_log.json missing — skipped")
        return False
    steps = [r["step"] for r in log["steps"]]
    pen = [r["pen_norm"] for r in log["steps"]]
    mse = [r["mse"] for r in log["steps"]]
    bench = log["bench"]  # [{tag, step0/75/150, rho_mini}]
    bsteps = [0, 75, 150]
    brho = [b["rho_mini"] for b in bench]

    grow, hold, ms = 62, 8, 85
    pmax = float(steps[-1])
    frames: list[Image.Image] = []
    durations: list[int] = []
    for f in range(grow + hold):
        p = min(1.0, f / (grow - 1)) if f < grow else 1.0
        head = p * pmax
        i = int(head)
        frac = head - i

        def _draw(
            series: list[float],
            _head: float = head,  # bind the loop frame: the closure is used
            _i: int = i,  # immediately, before the next iteration
            _frac: float = frac,
        ) -> tuple[list[float], list[float]]:
            return _series_head(steps, series, _head, _i, _frac)

        fig, ax1 = plt.subplots(figsize=(9.8, 5.0), dpi=100)
        ax1.plot(*_draw(pen), color=EDGE, lw=1.4, label="penalty pen_norm (v-projection)²")
        ax1.plot(*_draw(mse), color=BULK, lw=1.4, label="MSE anchor (good tail)")
        ax1.axhline(
            1.0,
            color=BULK,
            ls=":",
            lw=1,
            alpha=_ease((p - 0.25) / 0.15),
            label="anchor cap",
        )
        ax1.set_xlim(0, pmax)
        ax1.set_xlabel("training step")
        ax1.set_ylabel("loss term value")
        ax2 = ax1.twinx()
        ax2.spines["right"].set_visible(True)
        ax2.set_ylim(0, 1.0)
        ax2.set_ylabel(r"$\rho_{mini}$ (8 bad, greedy)", color=COLLAPSE)
        ax2.tick_params(axis="y", labelcolor=COLLAPSE)
        popped_all = True
        for bs, rho in zip(bsteps, brho):
            e = _ease((head - bs) / 12)
            popped_all = popped_all and e >= 1.0
            if e > 0:
                ax2.scatter(
                    [bs],
                    [rho],
                    s=40 + 90 * (1 - e),
                    color=COLLAPSE,
                    alpha=0.4 + 0.6 * e,
                    zorder=6,
                )
                ax2.annotate(
                    f"{rho:.2f}",
                    (bs, rho),
                    textcoords="offset points",
                    xytext=(0, 6),
                    ha="center",
                    fontsize=8,
                    color=COLLAPSE,
                    alpha=e,
                )
        if p > 0.2:
            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax1.legend(h1 + h2, l1 + l2, loc="center left", bbox_to_anchor=(0.40, 0.62))
        ax1.set_title("RCDA-5 training: the formula in the loss — real 150-step sweep")
        ax1.annotate(
            "data: R CDA - 5/Research/RCDA/results/training_log.json (bench: ρ_mini 0.80 → 0.50 → 0.45)",
            (0.995, 0.005),
            xycoords="axes fraction",
            ha="right",
            va="bottom",
            fontsize=6.5,
            color=MUTED,
        )
        frames.append(_shot(fig))
        durations.append(280 if f >= grow else (160 if popped_all else ms))
        plt.close(fig)
    out = os.path.join(OUT, "training-loop.gif")
    _write_gif(out, frames, durations)
    _check(out)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", choices=("scans", "dose", "training"))
    args = ap.parse_args()
    jobs = [args.only] if args.only else ("scans", "dose", "training")
    done = []
    if "scans" in jobs and anim_scans():
        done.append("scans-loop.gif")
    if "dose" in jobs and anim_dose():
        done.append("dose-loop.gif")
    if "training" in jobs and anim_training():
        done.append("training-loop.gif")
    print(f"done: {', '.join(done) if done else 'nothing (inputs missing)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
