"""Regenerate every figure and animation of the Gemma3-1B-IT RCDA kit.

Reads ONLY the committed result JSONs (probe report, training logs, part-evals,
channel decomposition, summary) - every pixel is data, nothing hand-typed.
Same visual identity as the repository: near-black field, cyan MP bulk,
amber BBP spike, violet structure, crimson defect.

  python make_graphic.py            # PNGs -> Graphic/, GIFs -> Graphic/anims/
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

KIT = Path(__file__).resolve().parent.parent
R5 = KIT / "RCDA-5" / "Research" / "RCDA" / "results"
R6 = KIT / "RCDA-6" / "Research" / "RCDA" / "results"
R7 = KIT / "RCDA-7" / "Research" / "RCDA" / "results"
OUT = KIT / "Graphic"
ANIM = OUT / "anims"

# palette
BG = "#0B0E14"; FG = "#E8ECF1"; GRID = "#232A36"
CYAN = "#00D4FF"; AMBER = "#FFB020"; VIOLET = "#B026FF"
CRIMSON = "#FF3B6B"; EMERALD = "#38E1C0"; PINK = "#FF6E9C"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": GRID, "axes.labelcolor": FG, "text.color": FG,
    "xtick.color": FG, "ytick.color": FG, "font.size": 10,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
})


def jload(p: Path) -> dict:
    return json.loads(Path(p).read_text())


def finish(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  {path.relative_to(KIT)}  {path.stat().st_size // 1024} KB")


def style(ax, title):
    ax.set_title(title, color=FG, fontsize=11, loc="left", pad=10)


# ---------------------------------------------------------------- figs
def fig05_probe():
    r = jload(R5 / "probe_report.json")
    layers = sorted(int(l) for l in r["layers"])
    z = [r["layers"][str(l)]["z_diff"] for l in layers]
    tv = [r["layers"][str(l)]["scan"]["theta_v"] for l in layers]
    writers = r["writer_layers"]
    fig, ax = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    colors = [CRIMSON if l in writers else CYAN for l in layers]
    ax.bar(layers, z, color=colors, alpha=0.9)
    ax.axhline(np.median(z), color=AMBER, lw=1.2, ls="--",
               label="median z_diff")
    ax2 = ax.twinx()
    ax2.plot(layers, tv, color=VIOLET, marker="o", ms=4, lw=1.4,
             label="theta_v (matrix scan)")
    ax2.axhline(1.0, color=PINK, lw=1.0, ls=":", label="theta_c = sqrt(q)")
    ax2.grid(False)
    style(ax, "RCDA-1 probe: behavioural separation peaks at the writers "
              "(crimson = selected writer layers)")
    ax.set_xlabel("transformer block"); ax.set_ylabel("z_diff")
    ax2.set_ylabel("theta_v", color=VIOLET)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", framealpha=0.2,
              fontsize=8, labelcolor=FG)
    finish(fig, OUT / "fig05_probe_zdiff_profile.png")


def fig06_scan():
    r = jload(R5 / "probe_report.json")
    layers = sorted(int(l) for l in r["layers"])
    lr = [r["layers"][str(l)]["scan"]["lambda_ratio"] for l in layers]
    fig, ax = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    ax.bar(layers, lr, color=CYAN, alpha=0.9)
    ax.axhline(1.0, color=AMBER, lw=1.4, label="lambda_+ (MP upper edge)")
    style(ax, "RCDA-2 matrix scan: lambda_v / lambda_+ per o_proj - the loop "
              "is below the single-matrix spike threshold (distributed defect)")
    ax.set_xlabel("transformer block"); ax.set_ylabel("lambda_v / lambda_+")
    ax.legend(framealpha=0.2, fontsize=8, labelcolor=FG)
    finish(fig, OUT / "fig06_matrix_scan.png")


def fig07_decomposition():
    d = jload(R5 / "rcda6_channel_decomposition.json")
    layers = sorted(int(l) for l in d["per_layer"])
    attn = [d["per_layer"][str(l)]["attn"] for l in layers]
    mlp = [d["per_layer"][str(l)]["mlp"] for l in layers]
    post = [d["per_layer"][str(l)]["post"] for l in layers]
    x = np.arange(len(layers)); w = 0.27
    fig, ax = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    ax.bar(x - w, post, w, color=VIOLET, label="residual stream (carried)")
    ax.bar(x, mlp, w, color=AMBER, label="MLP down_proj write")
    ax.bar(x + w, attn, w, color=CYAN, label="attention o_proj write")
    ax.set_xticks(x); ax.set_xticklabels([f"L{l}" for l in layers])
    ax.set_yscale("log")
    style(ax, f"RCDA-6 diagnostic: the defect is carried, not written - "
              f"MLP share of writes {d['totals']['mlp_share_of_writes']:.0%}")
    ax.set_ylabel("(v^T out)^2, log scale")
    ax.legend(framealpha=0.2, fontsize=8, labelcolor=FG)
    finish(fig, OUT / "fig07_channel_decomposition.png")


def fig08_training_curves():
    fig, ax = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    for variant, rdir, color in [("rcda5", R5, CYAN), ("rcda6", R6, AMBER),
                                 ("rcda7", R7, VIOLET)]:
        log = jload(rdir / f"{variant}_training_log.json")
        bad = [s for s in log["steps"] if s["kind"] == "bad"]
        steps = [s["step"] for s in bad]
        pen = [s["pen_norm"] for s in bad]
        k = 9
        smooth = np.convolve(pen, np.ones(k) / k, mode="valid")
        ax.plot(steps, pen, color=color, alpha=0.18, lw=0.8)
        ax.plot(steps[k // 2: -k // 2 + 1], smooth, color=color, lw=2.0,
                label=f"{variant}  ({(1 - smooth[-1] / smooth[0]):.0%} descent)")
    ax.axhline(1.0, color=CRIMSON, ls=":", lw=1.0, label="init (normalised)")
    style(ax, "The formula as a loss: (v^T h / ||v||)^2 melts on the bad "
              "trajectories (150 Adam steps, pen normalised per prompt)")
    ax.set_xlabel("training step (bad-step subset)")
    ax.set_ylabel("pen_norm = pen / pen0")
    ax.legend(framealpha=0.2, fontsize=8, labelcolor=FG)
    finish(fig, OUT / "fig08_training_pen_descent.png")


def fig09_rho_melt():
    fig, ax = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    for variant, rdir, color in [("rcda5", R5, CYAN), ("rcda6", R6, AMBER),
                                 ("rcda7", R7, VIOLET)]:
        log = jload(rdir / f"{variant}_training_log.json")
        b = log["bench"]
        ax.plot([x["step"] for x in b], [x["mean_loop_share"] for x in b],
                color=color, marker="o", ms=5, lw=1.8,
                label=f"{variant}  ({100 * (1 - b[-1]['mean_loop_share'] / b[0]['mean_loop_share']):.1f}%)")
    style(ax, "mini-benchmark (8 bad prompts, greedy 64): the loop bends "
              "under the penalty")
    ax.set_xlabel("training step"); ax.set_ylabel("mean loop_share")
    ax.set_ylim(0.7, 1.0)
    ax.legend(framealpha=0.2, fontsize=8, labelcolor=FG)
    finish(fig, OUT / "fig09_rho_mini_melt.png")


def fig10_headline():
    s = jload(R5 / "summary_all_variants.json")
    arms = ["base", "rcda5", "rcda6", "rcda7"]
    colors = [CRIMSON, CYAN, AMBER, VIOLET]
    metrics = [("mean_loop_share", "mean loop_share", False),
               ("loop_waste_mean", "loop waste (tokens)", False),
               ("rho_bad", "rho_bad", False)]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.2), constrained_layout=True)
    for ax, (key, label, _) in zip(axes, metrics):
        vals = [s["base"][key]] + [s["variants"][a][key] for a in arms[1:]]
        ax.bar(range(4), vals, color=colors, alpha=0.9)
        ax.set_xticks(range(4)); ax.set_xticklabels(arms, fontsize=8)
        ax.set_title(label, fontsize=9, color=FG, loc="left")
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7,
                    color=FG)
    finish(fig, OUT / "fig10_headline_defect.png")


def fig11_abilities():
    s = jload(R5 / "summary_all_variants.json")
    arms = ["base", "rcda5", "rcda6", "rcda7"]
    colors = [CRIMSON, CYAN, AMBER, VIOLET]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.2), constrained_layout=True)
    kl = [0.0] + [s["variants"][a]["kl_ctrl"] for a in arms[1:]]
    axes[0].bar(range(4), kl, color=colors, alpha=0.9)
    axes[0].set_title("KL(base||arm), control", fontsize=9, color=FG, loc="left")
    axes[0].set_ylim(0, max(kl) * 1.3 + 1e-3)
    ppl = [s["base"]["ppl_control_mean"]] + \
          [s["variants"][a]["ppl_control_mean"] for a in arms[1:]]
    axes[1].bar(range(4), ppl, color=colors, alpha=0.9)
    axes[1].set_ylim(min(ppl) * 0.9, max(ppl) * 1.08)
    axes[1].set_title("PPL on control corpus", fontsize=9, color=FG, loc="left")
    descent = [0] + [s["variants"][a]["pen_norm_descent_pct"] for a in arms[1:]]
    axes[2].bar(range(4), descent, color=colors, alpha=0.9)
    axes[2].set_title("penalty descent, %", fontsize=9, color=FG, loc="left")
    for ax in axes:
        ax.set_xticks(range(4)); ax.set_xticklabels(arms, fontsize=8)
    finish(fig, OUT / "fig11_abilities_preserved.png")


def fig12_cosine():
    s = jload(R5 / "summary_all_variants.json")
    arms = ["rcda5", "rcda6", "rcda7"]
    colors = {"rcda5": CYAN, "rcda6": AMBER, "rcda7": VIOLET}
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.4), constrained_layout=True,
                             sharey=True)
    for ax, a in zip(axes, arms):
        items = sorted(s["variants"][a]["cosine_edit_vs_v"].items())
        labels = [n.replace("transformer.h.", "L").replace(".attn.c_proj.weight", " attn")
                   .replace(".mlp.c_proj.weight", " mlp") for n, _ in items]
        vals = [v for _, v in items]
        ax.bar(range(len(vals)), vals, color=colors[a], alpha=0.9)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, fontsize=6, rotation=45)
        ax.set_ylim(0, 1.0)
        ax.set_title(a, fontsize=9, color=colors[a], loc="left")
        for i, v in enumerate(vals):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=6, color=FG)
    fig.suptitle("cos(u1 of the edit, v): the extended-tensor edits align with "
                 "the measured behavioural direction", color=FG, fontsize=10)
    finish(fig, OUT / "fig12_cosine_alignment.png")


def fig13_spectral():
    arms = [(5, "rcda5", CYAN), (6, "rcda6", AMBER), (7, "rcda7", VIOLET)]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.4), constrained_layout=True,
                             sharey=True)
    for ax, (r, variant, color) in zip(axes, arms):
        tw = jload(KIT / f"R CDA - {r}" / "Research" / "RCDA" / "results" /
                   "tw_relative.json")["tensors"]
        items = sorted(tw.items())
        labels = [n.replace("L", "") for n, _ in items]
        base_ks = [t["ks_base"] for _, t in items]
        trained = [t["ks_trained"] for _, t in items]
        x = np.arange(len(items)); w = 0.38
        ax.bar(x - w / 2, base_ks, w, color=CRIMSON, alpha=0.8,
               label="base (own KS)")
        ax.bar(x + w / 2, trained, w, color=color, alpha=0.9, label=variant)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(variant, fontsize=9, color=color, loc="left")
        ax.set_yscale("log")
        ax.legend(fontsize=6, framealpha=0.2, labelcolor=FG)
    fig.suptitle("Spectral purity, relative gate: bulk KS of edited matrices vs "
                 "the base's own KS (log scale; pass <= 1.5x)", color=FG, fontsize=10)
    finish(fig, OUT / "fig13_spectral_purity.png")


def fig14_efficiency():
    s = jload(R5 / "summary_all_variants.json")
    fig, ax = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    pts = {"rcda5": (CYAN, "o"), "rcda6": (AMBER, "s"), "rcda7": (VIOLET, "D")}
    for a, (color, mark) in pts.items():
        v = s["variants"][a]
        ax.scatter(v["kl_ctrl"], v["pen_norm_descent_pct"], s=140, color=color,
                   marker=mark, zorder=3)
        ax.annotate(f"{a}\npen -{v['pen_norm_descent_pct']}% @ KL {v['kl_ctrl']:.3f}",
                    (v["kl_ctrl"], v["pen_norm_descent_pct"]),
                    textcoords="offset points", xytext=(10, 4), fontsize=8,
                    color=FG)
    style(ax, "The leverage chart: defect-energy descent bought per unit of "
              "distribution drift (KL on control corpus)")
    ax.set_xlabel("KL(base||arm) on control texts")
    ax.set_ylabel("penalty descent, %")
    ax.set_xlim(-0.002, 0.016)
    finish(fig, OUT / "fig14_leverage_chart.png")


def fig15_ppp_parity():
    p = jload(R5 / "ppl_check.json")
    arms = [a for a in ["base", "trained"] if a in p]
    # ppl_check.json holds base (round-5 dir) + the round's trained arm; the
    # 6/7 dirs hold their own trained arms
    p6 = jload(R6 / "ppl_check.json")["trained"]
    p7 = jload(R7 / "ppl_check.json")["trained"]
    labels = ["base", "RCDA-5", "RCDA-6", "RCDA-7"]
    per_text = [p["base"]["ppl_per_text"], p["trained"]["ppl_per_text"],
                p6["ppl_per_text"], p7["ppl_per_text"]]
    means = [np.mean(x) for x in per_text]
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    ax.bar(x, means, color=[CRIMSON, CYAN, AMBER, VIOLET], alpha=0.9)
    for i, (m, arr) in enumerate(zip(means, per_text)):
        ax.scatter([i] * len(arr), arr, color=FG, s=14, zorder=3, alpha=0.8)
        ax.text(i, m, f"{m:.1f}", ha="center", va="bottom", fontsize=8, color=FG)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    style(ax, "Abilities parity: PPL on the control corpus, per text (dots) "
              "and mean (bar) - the anchor holds within tolerance")
    ax.set_ylabel("perplexity")
    finish(fig, OUT / "fig15_ppl_parity.png")


def fig16_overview():
    s = jload(R5 / "summary_all_variants.json")
    fig = plt.figure(figsize=(7.2, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, :])
    for variant, rdir, color in [("rcda5", R5, CYAN), ("rcda6", R6, AMBER),
                                 ("rcda7", R7, VIOLET)]:
        log = jload(rdir / f"{variant}_training_log.json")
        bad = [x for x in log["steps"] if x["kind"] == "bad"]
        pen = [x["pen_norm"] for x in bad]
        k = 15
        sm = np.convolve(pen, np.ones(k) / k, mode="valid")
        ax1.plot(sm, color=color, lw=1.8, label=variant)
    ax1.set_ylabel("pen_norm (smoothed)"); ax1.legend(fontsize=8,
                                                      framealpha=0.2, labelcolor=FG)
    ax1.set_title("the formula melting the defect, all three variants",
                  color=FG, fontsize=10, loc="left")
    ax2 = fig.add_subplot(gs[1, 0])
    arms = ["base", "rcda5", "rcda6", "rcda7"]
    ls = [s["base"]["mean_loop_share"]] + [s["variants"][a]["mean_loop_share"]
                                           for a in arms[1:]]
    ax2.bar(arms, ls, color=[CRIMSON, CYAN, AMBER, VIOLET], alpha=0.9)
    ax2.set_ylim(0.8, 0.95); ax2.set_title("loop_share (greedy 96)",
                                           fontsize=9, color=FG, loc="left")
    ax2.tick_params(axis="x", labelsize=7)
    ax3 = fig.add_subplot(gs[1, 1])
    cosmax = [0] + [max(s["variants"][a]["cosine_edit_vs_v"].values())
                    for a in arms[1:]]
    ax3.bar(arms, cosmax, color=[CRIMSON, CYAN, AMBER, VIOLET], alpha=0.9)
    ax3.set_ylim(0, 1); ax3.set_title("max |cos(u1, v)|", fontsize=9,
                                      color=FG, loc="left")
    ax3.tick_params(axis="x", labelsize=7)
    finish(fig, OUT / "fig16_overview_dashboard.png")


# ---------------------------------------------------------------- anims
def gif_training():
    frames = []
    fig, ax = plt.subplots(figsize=(4.8, 3.0), constrained_layout=True,
                           dpi=100)
    logs = {v: jload(rd / f"{v}_training_log.json")
            for v, rd in [("rcda5", R5), ("rcda6", R6), ("rcda7", R7)]}
    colors = {"rcda5": CYAN, "rcda6": AMBER, "rcda7": VIOLET}
    max_step = 150
    for step in range(5, max_step + 1, 5):
        ax.clear()
        ax.set_facecolor(BG); fig.patch.set_facecolor(BG)
        for v, log in logs.items():
            bad = [s for s in log["steps"] if s["kind"] == "bad" and s["step"] <= step]
            pen = [s["pen_norm"] for s in bad]
            k = 9
            if len(pen) > k:
                sm = np.convolve(pen, np.ones(k) / k, mode="valid")
                ax.plot(range(len(sm)), sm, color=colors[v], lw=2.0)
        ax.axhline(1.0, color=CRIMSON, ls=":", lw=1.0)
        ax.set_xlim(0, 75); ax.set_ylim(0, 1.05)
        ax.grid(color=GRID, lw=0.5)
        ax.tick_params(colors=FG, labelsize=7)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.set_title(f"pen_norm over bad steps - step {step}", color=FG,
                     fontsize=9, loc="left")
        fig.canvas.draw()
        frames.append(Image.frombytes("RGBA", fig.canvas.get_width_height(),
                                      fig.canvas.buffer_rgba()))
    out = ANIM / "training-loop.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=160, loop=0)
    print(f"  {out.relative_to(KIT)}  {out.stat().st_size // 1024} KB "
          f"({len(frames)} frames)")
    plt.close(fig)


def gif_defect_melt():
    s = jload(R5 / "summary_all_variants.json")
    stages = [("base", s["base"]["mean_loop_share"], s["base"]["loop_waste_mean"]),
              ("rcda5", s["variants"]["rcda5"]["mean_loop_share"],
               s["variants"]["rcda5"]["loop_waste_mean"]),
              ("rcda6", s["variants"]["rcda6"]["mean_loop_share"],
               s["variants"]["rcda6"]["loop_waste_mean"]),
              ("rcda7", s["variants"]["rcda7"]["mean_loop_share"],
               s["variants"]["rcda7"]["loop_waste_mean"])]
    frames = []
    for i in range(24):
        t = min(i / 6.0, 3.0)
        lo = int(t); hi = min(lo + 1, 3); f = t - lo
        ls = stages[lo][1] + (stages[hi][1] - stages[lo][1]) * f
        ws = stages[lo][2] + (stages[hi][2] - stages[lo][2]) * f
        fig, ax = plt.subplots(figsize=(4.8, 3.0), dpi=100, constrained_layout=True)
        fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
        cols = [CRIMSON, CYAN, AMBER, VIOLET]
        bars = ax.bar([0, 1], [ls, ws / 100], color=[CRIMSON, CYAN], alpha=0.9)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["loop_share", "waste /100"],
                                                  fontsize=8, color=FG)
        ax.set_ylim(0, 1.0)
        ax.text(0, ls, f"{ls:.3f}", ha="center", va="bottom", color=FG, fontsize=9)
        ax.text(1, ws / 100, f"{ws:.0f}", ha="center", va="bottom", color=FG,
                fontsize=9)
        name = stages[min(round(t), 3)][0]
        ax.set_title(f"defect state: {name}", color=FG, fontsize=10, loc="left")
        ax.grid(color=GRID, lw=0.5)
        ax.tick_params(colors=FG, labelsize=7)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        fig.canvas.draw()
        frames.append(Image.frombytes("RGBA", fig.canvas.get_width_height(),
                                      fig.canvas.buffer_rgba()))
        plt.close(fig)
    out = ANIM / "defect-melt.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=180, loop=0)
    print(f"  {out.relative_to(KIT)}  {out.stat().st_size // 1024} KB "
          f"({len(frames)} frames)")


if __name__ == "__main__":
    print("[graphic] rendering figures from committed JSONs ...")
    fig05_probe(); fig06_scan(); fig07_decomposition(); fig08_training_curves()
    fig09_rho_melt(); fig10_headline(); fig11_abilities(); fig12_cosine()
    fig13_spectral(); fig14_efficiency(); fig15_ppp_parity(); fig16_overview()
    print("[graphic] rendering animations ...")
    gif_training(); gif_defect_melt()
    print("[graphic] done")
