#!/usr/bin/env python3
"""rcda5_figures.py — figures for the Gemma3-1B-IT RCDA-5 report
(the analog of the MNN kit's rcda5_figures.py / fig15).

  fig15_gemma_training.png   training curves: pen_raw / pen_norm, the anchor
                             MSE against its hard cap, and the mini-benchmark
                             rho_mini at steps 0/75/150
  figG1_probe_layers.png     the per-layer behavioral separation profile
                             (z_diff) with the writer band highlighted
  figG2_spectral_purity.png  singular spectrum of the edit delta per writer
                             layer vs the MP bulk edge of the v-orthogonal
                             residual + TW verdicts
  figG3_gemma_eval.png       the final three-arm summary: rho_bad / rho_good /
                             loop waste / KL / PPL, base vs trained, and
                             cos(u1, v) per writer layer

Usage: rcda5_figures.py --work DIR --results DIR [--graphic DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "Code"))

# aurora dark style (the repo's figure language)
BG, PANEL, TXT, GRID = "#0B0E14", "#12161F", "#E6EDF3", "#2A3140"
CYAN, VIOLET, AMBER, CRIMSON = "#38E1C0", "#B98CFF", "#FFB020", "#FF3B6B"


def style_ax(ax):
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=TXT, labelsize=9)
    ax.xaxis.label.set_color(TXT)
    ax.yaxis.label.set_color(TXT)
    ax.title.set_color(TXT)
    ax.grid(color=GRID, alpha=0.5, lw=0.6)


def fig_training(log, path):
    steps = [s["step"] for s in log["steps"]]
    pen_raw = [s["pen_raw"] for s in log["steps"]]
    pen_norm = [s["pen_norm"] for s in log["steps"]]
    mse = [s["mse"] for s in log["steps"]]
    cap = log["args"]["mse_cap"]
    bench = {b["tag"]: b["rho_mini"] for b in log["bench"]}
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.4), dpi=150,
                             constrained_layout=True)
    fig.patch.set_facecolor(BG)
    ax = axes[0]
    style_ax(ax)
    ax.plot(steps, pen_raw, color=VIOLET, lw=1.2, label="pen_raw  $(\\hat v^T h)^2$ mean")
    ax2 = ax.twinx()
    ax2.plot(steps, pen_norm, color=CYAN, lw=1.6,
             label="pen_norm (dimensionless, init = 1)")
    for s, c in bench.items():
        st = int(s.replace("step", ""))
        ax2.scatter([st], [bench[s]], color=CRIMSON, zorder=5, s=45,
                    marker="D")
        ax2.annotate(f"$\\rho_{{mini}}$={bench[s]}", (st, bench[s]),
                     xytext=(6, 6), textcoords="offset points",
                     color=CRIMSON, fontsize=9)
    ax.set_xlabel("SGD step")
    ax.set_ylabel("pen_raw", color=VIOLET)
    ax2.set_ylabel("pen_norm / $\\rho_{mini}$", color=CYAN)
    ax2.tick_params(colors=CYAN, labelsize=9)
    style_ax(ax2)
    ax.set_title(f"Gemma3-1B-IT RCDA-5: the formula as a loss — "
                 f"SGD {log['args']['steps']} steps, lr {log['args']['lr']}",
                 fontsize=11)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", framealpha=0.2,
              labelcolor=TXT, fontsize=8)
    ax = axes[1]
    style_ax(ax)
    ax.plot(steps, mse, color=AMBER, lw=1.3, label="anchor MSE (trailing 32 logits)")
    ax.axhline(cap, color=CRIMSON, ls="--", lw=1.2, label=f"hard cap = {cap}")
    ax.fill_between(steps, mse, cap, where=[m > cap for m in mse],
                    color=CRIMSON, alpha=0.25, label="cap violations (β·relu)")
    ax.set_xlabel("SGD step")
    ax.set_ylabel("MSE vs base logits", color=AMBER)
    ax.legend(loc="upper right", framealpha=0.2, labelcolor=TXT, fontsize=8)
    ax.set_title("the abilities anchor holds within tolerance", fontsize=10)
    fig.savefig(path, facecolor=BG)
    plt.close(fig)


def fig_probe(probe, path):
    lp = probe["layer_profile"]
    layers = sorted(int(k) for k in lp)
    zd = [lp[str(l)]["z_diff"] for l in layers]
    writers = set(probe["writers"])
    colors = [AMBER if l in writers else "#5A6B87" for l in layers]
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=150, constrained_layout=True)
    fig.patch.set_facecolor(BG)
    style_ax(ax)
    ax.bar(layers, zd, color=colors, width=0.72)
    for l in writers:
        ax.annotate(f"L{l}", (l, lp[str(l)]["z_diff"]), xytext=(0, 4),
                    textcoords="offset points", ha="center", color=AMBER,
                    fontsize=9, fontweight="bold")
    ax.set_xlabel("decoder layer")
    ax.set_ylabel("z_diff (loop vs pre-loop separability)")
    ax.set_title("Gemma3-1B-IT: the loop writes in the L10–L12 band "
                 "(analog of Pocket L15/16/18)", fontsize=11)
    fig.savefig(path, facecolor=BG)
    plt.close(fig)


def fig_spectral(digest, path):
    layers = [d["layer"] for d in digest.values()]
    fig, axes = plt.subplots(1, len(layers), figsize=(4.1 * len(layers), 4.4),
                             dpi=150, constrained_layout=True)
    fig.patch.set_facecolor(BG)
    if len(layers) == 1:
        axes = [axes]
    for ax, (k, d) in zip(axes, digest.items()):
        style_ax(ax)
        s = np.array(d["top16_singular_values_delta"])
        edge = d["lam_plus_resid"]
        ax.semilogy(range(1, len(s) + 1), s, "o-", color=CYAN, lw=1.4, ms=4,
                    label="singular values of $\\Delta W$")
        ax.axhline(edge, color=CRIMSON, ls="--", lw=1.2,
                   label="MP edge of the $v$-orthogonal residual bulk")
        ax.set_title(f"L{d['layer']}: TW {d['tw_verdict']}, "
                     f"cos$(u_1,v)$={d['cos_u1_v']:.2f}", fontsize=10)
        ax.set_xlabel("singular index")
        ax.set_ylabel("$\\sigma_i(\\Delta W)$")
        ax.legend(framealpha=0.2, labelcolor=TXT, fontsize=8)
    fig.savefig(path, facecolor=BG)
    plt.close(fig)


def fig_eval(base, trained, digest, path):
    metrics = [("rho_bad", "ρ_bad (bad loop share)", 1),
               ("rho_good", "ρ_good", 1),
               ("loop_waste_mean", "loop waste (tok, mean bad)", 1),
               ("kl_ctrl", "KL(base‖arm)", 1)]
    fig, axes = plt.subplots(1, 5, figsize=(13.2, 3.7), dpi=150,
                             constrained_layout=True)
    fig.patch.set_facecolor(BG)
    for ax, (k, label, _) in zip(axes[:4], metrics):
        style_ax(ax)
        vals = [base[k], trained[k]]
        ax.bar([0, 1], vals, color=["#5A6B87", CYAN], width=0.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["base", "trained"], fontsize=9)
        for i, v in enumerate(vals):
            ax.annotate(f"{v}", (i, v), xytext=(0, 4),
                        textcoords="offset points", ha="center", color=TXT,
                        fontsize=9)
        ax.set_title(label, fontsize=9)
    ax = axes[4]
    style_ax(ax)
    lays = [d["layer"] for d in digest.values()]
    cosv = [d["cos_u1_v"] for d in digest.values()]
    ax.bar([str(l) for l in lays], cosv, color=VIOLET, width=0.55)
    ax.set_ylim(0, 1)
    ax.set_title("cos(u₁, v) per writer layer", fontsize=9)
    fig.savefig(path, facecolor=BG)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True,
                    help="runtime work dir (probe/, rcda5_final/)")
    ap.add_argument("--results", required=True, help="results dir for figures")
    ap.add_argument("--graphic", default=None, help="mirror dir (Graphic/)")
    args = ap.parse_args()

    work = Path(args.work)
    out = Path(args.results)
    out.mkdir(parents=True, exist_ok=True)

    log = json.load(open(work / "rcda5_final" / "training_log.json"))
    probe = json.load(open(work / "probe" / "probe_report.json"))
    digest = json.load(open(out / "rcda5_spectral_digest.json"))["digest"]
    base = json.load(open(out / "part5_eval_base.json"))
    trained = json.load(open(out / "part5_eval_trained.json"))

    fig_training(log, out / "fig15_gemma_training.png")
    fig_probe(probe, out / "figG1_probe_layers.png")
    fig_spectral(digest, out / "figG2_spectral_purity.png")
    fig_eval(base, trained, digest, out / "figG3_gemma_eval.png")

    if args.graphic:
        g = Path(args.graphic)
        g.mkdir(parents=True, exist_ok=True)
        import shutil
        for f in ("fig15_gemma_training.png", "figG1_probe_layers.png",
                  "figG2_spectral_purity.png", "figG3_gemma_eval.png"):
            shutil.copy2(out / f, g / f)
    print("figures written to", out)


if __name__ == "__main__":
    main()
