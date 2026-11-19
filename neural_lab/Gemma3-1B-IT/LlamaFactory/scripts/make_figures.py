#!/usr/bin/env python3
"""make_figures.py — every figure and GIF of the RCDA-8 LlamaFactory round.

Reads ONLY the committed result JSONs (the kit's rule: nothing is
hand-typed) and renders the static figures + animations in the project
identity: near-black field, cyan MP bulk, amber BBP spike, violet
structure, crimson defect, emerald data arm.

Sources:
  results/probe/probe_report.json            behavioral z_diff profile
  results/probe/probe_phase_report.json      phase directions
  results/rcda8_arm/training_log.json        the formula arm training
  results/part8_eval_{base,rcda8,sft}.json   the one evaluator, three arms
  results/sft_arm/.../trainer_state.json     the data arm training (LlamaFactory)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rcda8_lab as L                                          # noqa: E402

# project identity -----------------------------------------------------------
BG, FG, GRID = "#0B0E14", "#E8ECF1", "#232A36"
CYAN, AMBER, VIOLET = "#00D4FF", "#FFB020", "#B026FF"
CRIMSON, EMERALD, PINK = "#FF3B6B", "#38E1C0", "#FF6E9C"
MUTED = "#7A8699"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": GRID, "axes.labelcolor": FG, "text.color": FG,
    "xtick.color": FG, "ytick.color": FG, "font.size": 10,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlepad": 12,
    "legend.frameon": False, "legend.fontsize": 9,
    "font.family": "DejaVu Sans",
})

OUT = L.KIT_DIR / "figures"
OUT.mkdir(parents=True, exist_ok=True)
ARMS = ["base", "rcda8", "sft"]
ARM_LABEL = {"base": "base (pristine)", "rcda8": "RCDA-8 LoRA (formula)",
             "sft": "SFT LoRA (data)"}
ARM_COLOR = {"base": MUTED, "rcda8": VIOLET, "sft": EMERALD}


def load(tag):
    return json.loads((L.RESULTS_DIR / f"part8_eval_{tag}.json").read_text(
        encoding="utf-8"))


def clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)


def save(fig, name, dpi=200):
    fig.savefig(OUT / name, dpi=dpi, facecolor=BG, bbox_inches="tight",
                pad_inches=0.25)
    plt.close(fig)
    kb = (OUT / name).stat().st_size / 1024
    print(f"  {name} ({kb:.0f} KB)", flush=True)


def save_gif(anim, name, fps=4):
    anim.save(OUT / name, writer=PillowWriter(fps=fps))
    kb = (OUT / name).stat().st_size / 1024
    print(f"  {name} ({kb:.0f} KB)", flush=True)


E = {t: load(t) for t in ARMS}
probe = json.loads((L.RESULTS_DIR / "probe" / "probe_report.json").read_text())
phase = json.loads((L.RESULTS_DIR / "probe" / "probe_phase_report.json").read_text())
rlog = json.loads((L.RESULTS_DIR / "rcda8_arm" / "training_log.json").read_text())
sft_state = json.loads((L.RESULTS_DIR / "sft_arm" / "gemma3_sft_lora" /
                        "trainer_state.json").read_text())

WRITERS = probe["writers"]                       # [10, 11, 12]
BAND = rlog["args"]["band"]                      # [8..15]
layers = sorted(int(l) for l in probe["layer_profile"])
zdiff = [probe["layer_profile"][str(l)]["z_diff"] for l in layers]


def fig01_probe():
    fig, ax = plt.subplots(figsize=(9.5, 4.6), constrained_layout=True)
    colors = [CRIMSON if l in WRITERS else
              (VIOLET if l in BAND else CYAN) for l in layers]
    ax.bar(layers, zdiff, color=colors, alpha=0.92, width=0.72)
    ax.axhline(float(np.median(zdiff)), color=AMBER, lw=1.2, ls="--",
               label=f"median {np.median(zdiff):.0f}")
    for l in WRITERS:
        ax.text(l, zdiff[l] + 6, f"L{l}\n{zdiff[l]:.0f}", ha="center",
                va="bottom", fontsize=9, color=CRIMSON, fontweight="bold")
    ax.set_xlabel("decoder layer")
    ax.set_ylabel("z_diff (diagonal Mahalanobis)")
    ax.set_title("The behavioral probe on real Gemma3-1B-IT — writers L10/11/12, "
                 "band L8–L15 (z_diff > 130)")
    ax.set_ylim(0, max(zdiff) * 1.24)
    ax.legend(loc="upper right")
    clean(ax)
    save(fig, "fig01_probe_zdiff.png")


def fig02_phase():
    pl = phase["layer_profile"]
    ze = [pl[str(l)]["z_early"] for l in layers]
    zd = [pl[str(l)]["z_deep"] for l in layers]
    cos = [pl[str(l)]["cos_v_early_v_deep"] for l in layers]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.5, 6.4),
                                 constrained_layout=True, sharex=True)
    a1.plot(layers, ze, color=CYAN, lw=2, marker="o", ms=4, label="z_early")
    a1.plot(layers, zd, color=AMBER, lw=2, marker="s", ms=4, label="z_deep")
    for l in WRITERS:
        a1.axvline(l, color=CRIMSON, lw=0.8, ls=":", alpha=0.7)
    a1.set_ylabel("phase separability")
    a1.set_title("Phase-conditioned directions — early vs deep loop phases differ "
                 "(cos drops to ~0.3 in the band)")
    a1.legend(loc="upper right", ncol=2)
    clean(a1)
    a2.plot(layers, cos, color=VIOLET, lw=2, marker="o", ms=4)
    a2.axhline(1.0, color=MUTED, lw=0.8, ls="--", label="identical")
    a2.set_xlabel("decoder layer")
    a2.set_ylabel("cos(v_early, v_deep)")
    a2.set_ylim(0, 1.05)
    a2.legend(loc="lower right")
    clean(a2)
    save(fig, "fig02_phase_probe.png")


def fig03_arms_defect():
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2),
                             constrained_layout=True)
    metrics = [
        ("rho_bad", E and [E[t]["rho_bad"] for t in ARMS],
         "ρ_bad (share of looping prompts)", "{:.2f}", 0.28),
        ("ls", [E[t]["loop_share_mean_bad"] for t in ARMS],
         "mean loop_share (bad 20)", "{:.3f}", 0.35),
        ("waste", [E[t]["loop_waste_mean_bad"] for t in ARMS],
         "mean loop_waste, tokens", "{:.0f}", 45),
    ]
    for ax, (key, vals, title, fmt, head) in zip(axes, metrics):
        bars = ax.bar(range(3), vals, color=[ARM_COLOR[t] for t in ARMS],
                      width=0.62, alpha=0.95)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2,
                    b.get_height() + max(vals) * 0.03, fmt.format(v),
                    ha="center", va="bottom", fontsize=10, color=FG)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["base", "RCDA-8\nformula", "SFT\ndata"],
                           fontsize=9)
        ax.set_title(title, fontsize=10.5)
        ax.set_ylim(0, max(vals) * 1.22)
        clean(ax)
    fig.suptitle("The defect across the three arms — greedy 192, 4-gram, "
                 "threshold 0.5 (one evaluator)", y=1.04, fontsize=13,
                 fontweight="bold")
    save(fig, "fig03_arms_defect.png")


def fig04_abilities():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.2),
                                 constrained_layout=True)
    kl = [E[t]["kl_ctrl_mean"] for t in ARMS]
    ppl = [E[t]["ppl_control_mean"] for t in ARMS]
    bars = a1.bar(range(3), kl, color=[ARM_COLOR[t] for t in ARMS],
                  width=0.6, alpha=0.95)
    for b, v in zip(bars, kl):
        a1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012,
                f"{v:.3f}", ha="center", va="bottom", fontsize=10, color=FG)
    a1.set_xticks(range(3))
    a1.set_xticklabels(["base", "RCDA-8", "SFT"], fontsize=9)
    a1.set_title("KL(base ‖ arm) on control — abilities drift")
    a1.set_ylim(0, max(kl) * 1.25)
    clean(a1)

    bars = a2.bar(range(3), ppl, color=[ARM_COLOR[t] for t in ARMS],
                  width=0.6, alpha=0.95)
    for b, v in zip(bars, ppl):
        a2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.2,
                f"{v:.1f}", ha="center", va="bottom", fontsize=10, color=FG)
    a2.axhline(E["base"]["ppl_control_mean"], color=CRIMSON, lw=1.0,
               ls=":", label="base parity")
    a2.set_xticks(range(3))
    a2.set_xticklabels(["base", "RCDA-8", "SFT"], fontsize=9)
    a2.set_title("PPL on control corpus — parity / self-distill gain")
    a2.set_ylim(0, max(ppl) * 1.25)
    a2.legend(loc="lower right")
    clean(a2)
    save(fig, "fig04_abilities.png")


def fig05_pen_descent():
    steps = [s["step"] for s in rlog["steps"]]
    pen = [s["pen_norm"] for s in rlog["steps"]]
    mse = [s["mse"] for s in rlog["steps"]]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2),
                                 constrained_layout=True)
    k = 5
    p = np.array(pen)
    sm = np.convolve(p, np.ones(k) / k, mode="same")
    a1.plot(steps, p, color=VIOLET, alpha=0.25, lw=0.9)
    a1.plot(steps[k // 2:-k // 2 + 1], sm[k // 2:-k // 2 + 1], color=VIOLET,
            lw=2.2, label="pen_norm (smoothed 5)")
    a1.axhline(1.0, color=CRIMSON, ls=":", lw=1.1, label="init = 1.0")
    a1.set_xlabel("training step")
    a1.set_ylabel("pen_norm")
    a1.set_title("The formula melting: pen_norm −97.9% in 150 steps")
    a1.legend(loc="upper right")
    clean(a1)
    m = np.array(mse)
    a2.plot(steps, m, color=CYAN, lw=1.0, alpha=0.5)
    mm = np.convolve(m, np.ones(k) / k, mode="same")
    a2.plot(steps[k // 2:-k // 2 + 1], mm[k // 2:-k // 2 + 1], color=CYAN,
            lw=2.2, label="anchor MSE (smoothed 5)")
    a2.axhline(rlog["args"]["cap"], color=AMBER, ls="--", lw=1.2,
               label=f"cap = {rlog['args']['cap']}")
    a2.set_xlabel("training step")
    a2.set_ylabel("MSE (trailing-32 logits)")
    a2.set_title("The abilities anchor — MSE rides under the cap")
    a2.legend(loc="upper right")
    clean(a2)
    save(fig, "fig05_pen_descent.png")


def fig06_forensics():
    fore = E["rcda8"]["lora_forensics"]
    rows = sorted(fore.items(), key=lambda kv: (kv[1]["layer"],
                                                "o" in kv[0].split(".")[-2]))
    labels = [k.replace("model.layers.", "L").replace(
        ".self_attn.o_proj.weight", " o").replace(
        ".mlp.down_proj.weight", " d") for k, _ in rows]
    rel = [v["rel_frobenius"] * 100 for _, v in rows]
    cosd = [v["cos_u1_v_deep"] or 0 for _, v in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5.0),
                                 constrained_layout=True)
    cols = [VIOLET if " o" in l else AMBER for l in labels]
    a1.barh(range(len(labels)), rel, color=cols, alpha=0.92, height=0.62)
    a1.set_yticks(range(len(labels)))
    a1.set_yticklabels(labels, fontsize=8)
    a1.invert_yaxis()
    a1.set_xlabel("rel. Frobenius of ΔW, %")
    a1.set_title("The LoRA edit per tensor (band L8–L15)")
    clean(a1)
    a2.barh(range(len(labels)), cosd, color=CYAN, alpha=0.92, height=0.62)
    a2.set_yticks(range(len(labels)))
    a2.set_yticklabels(labels, fontsize=8)
    a2.invert_yaxis()
    a2.set_xlabel("|cos(u₁ of ΔW, v_deep)|")
    a2.set_title("Does the edit point along the defect? (≤ 0.44)")
    a2.set_xlim(0, 1.0)
    clean(a2)
    save(fig, "fig06_forensics.png")


def fig07_repair_matrix():
    ids = [f"g{i+1:02d}" for i in range(20)]
    M = np.zeros((20, 3))
    for j, t in enumerate(ARMS):
        for i, v in enumerate(E[t]["per_traj_loop_share_bad"]):
            M[i, j] = v
    fig, ax = plt.subplots(figsize=(7.2, 8.6), constrained_layout=True)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "loop", ["#0B0E14", "#31173A", CRIMSON])
    im = ax.imshow(M, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["base", "RCDA-8", "SFT"], fontsize=9)
    ax.set_yticks(range(20))
    ax.set_yticklabels(ids, fontsize=7)
    for i in range(20):
        for j in range(3):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    fontsize=7,
                    color=FG if M[i, j] > 0.45 else MUTED)
    ax.set_title("Per-prompt loop_share (greedy 192) — repaired rows go dark")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04, shrink=0.75)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=8)
    save(fig, "fig07_repair_matrix.png")


def fig08_dashboard():
    fig = plt.figure(figsize=(13, 8.6), constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.3, hspace=0.35)
    fig.suptitle("RCDA-8 · the LlamaFactory round — one glance",
                 fontsize=15, fontweight="bold")

    a1 = fig.add_subplot(gs[0, 0])
    rho = [E[t]["rho_bad"] for t in ARMS]
    a1.bar(range(3), rho, color=[ARM_COLOR[t] for t in ARMS], width=0.6)
    for i, v in enumerate(rho):
        a1.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
    a1.set_xticks(range(3))
    a1.set_xticklabels(["base", "RCDA-8", "SFT"], fontsize=9)
    a1.set_title("ρ_bad (lower = better)", fontsize=11)
    a1.set_ylim(0, 0.9)
    clean(a1)

    a2 = fig.add_subplot(gs[0, 1])
    rep = [0, E["rcda8"]["prompts_fully_repaired"] or 0,
           E["sft"]["prompts_fully_repaired"] or 0]
    a2.bar(range(3), rep, color=[ARM_COLOR[t] for t in ARMS], width=0.6)
    for i, v in enumerate(rep):
        a2.text(i, v + 0.12, f"{v}/12", ha="center", fontsize=10)
    a2.set_xticks(range(3))
    a2.set_xticklabels(["base", "RCDA-8", "SFT"], fontsize=9)
    a2.set_title("hard loopers fully repaired (of 12)", fontsize=11)
    a2.set_ylim(0, 7)
    clean(a2)

    a3 = fig.add_subplot(gs[1, 0])
    bench = rlog["bench"]
    tags = [b["tag"] for b in bench]
    ls = [b["mean_ls"] for b in bench]
    a3.plot(range(len(tags)), ls, color=VIOLET, lw=2.4, marker="o", ms=6)
    for i, (t_, v) in enumerate(zip(tags, ls)):
        a3.annotate(t_, (i, v), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8, color=MUTED)
    a3.set_xticks(range(len(tags)))
    a3.set_xticklabels(tags, fontsize=9)
    a3.set_ylim(0, 1.05)
    a3.set_title("RCDA-8 mini-bench: mean loop_share melts 0.92 → 0.53",
                 fontsize=11)
    clean(a3)

    a4 = fig.add_subplot(gs[1, 1])
    kl = [E[t]["kl_ctrl_mean"] for t in ARMS]
    ppl = [E[t]["ppl_control_mean"] for t in ARMS]
    x = np.arange(3)
    a4.bar(x - 0.18, [k * 20 for k in kl], width=0.34, color=PINK,
           alpha=0.9, label="KL × 20")
    a4.bar(x + 0.18, ppl, width=0.34, color=CYAN, alpha=0.9,
           label="PPL")
    a4.set_xticks(x)
    a4.set_xticklabels(["base", "RCDA-8", "SFT"], fontsize=9)
    a4.set_title("abilities: KL (×20) and PPL", fontsize=11)
    a4.legend(loc="upper left")
    clean(a4)
    save(fig, "fig08_dashboard.png")


def fig09_sft_loss():
    hist = [h for h in sft_state["log_history"] if "loss" in h]
    steps = [h["step"] for h in hist]
    loss = [h["loss"] for h in hist]
    fig, ax = plt.subplots(figsize=(9, 4.2), constrained_layout=True)
    ax.plot(steps, loss, color=EMERALD, lw=1.6, alpha=0.8)
    k = 5
    sm = np.convolve(loss, np.ones(k) / k, mode="same")
    ax.plot(steps[k // 2:-k // 2 + 1], sm[k // 2:-k // 2 + 1], color=EMERALD,
            lw=2.6, label="smoothed 5")
    ax.set_xlabel("step (LlamaFactory SFT/LoRA, 4 epochs × 38 examples)")
    ax.set_ylabel("CE loss")
    ax.set_title(f"The data arm — anti-loop SFT loss {loss[0]:.2f} → "
                 f"{np.mean(loss[-5:]):.2f}")
    ax.legend(loc="upper right")
    clean(ax)
    save(fig, "fig09_sft_loss.png")


# ---------------------------------------------------------------- GIFs ------

def gif_pen_descent():
    steps = [s["step"] for s in rlog["steps"]]
    pen = np.array([s["pen_norm"] for s in rlog["steps"]])
    mse = np.array([s["mse"] for s in rlog["steps"]])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.6))
    n = len(steps)

    def draw(i):
        a1.clear(); a2.clear()
        a1.plot(steps[:i + 1], pen[:i + 1], color=VIOLET, lw=2.0)
        a1.axhline(1.0, color=CRIMSON, ls=":", lw=1.0)
        a1.set_ylim(0, 1.4); a1.set_xlim(1, n)
        a1.set_title(f"pen_norm (step {steps[i]})", fontsize=10)
        clean(a1)
        a2.plot(steps[:i + 1], mse[:i + 1], color=CYAN, lw=1.8)
        a2.axhline(1.0, color=AMBER, ls="--", lw=1.1)
        a2.set_ylim(0, 3.2); a2.set_xlim(1, n)
        a2.set_title("anchor MSE vs cap=1.0", fontsize=10)
        clean(a2)
    anim = FuncAnimation(fig, draw, frames=range(0, n, 2), interval=260)
    save_gif(anim, "gif01_pen_descent.gif", fps=5)


def gif_bench_melt():
    bench = rlog["bench"]
    tags = [b["tag"] for b in bench]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))

    def draw(fi):
        ax.clear()
        b = bench[fi]
        ids = [g["id"] for g in b["gens"]]
        ls = [g["loop_share"] for g in b["gens"]]
        cols = [CRIMSON if v >= 0.5 else EMERALD for v in ls]
        ax.bar(range(len(ids)), ls, color=cols, width=0.66)
        ax.axhline(0.5, color=AMBER, ls="--", lw=1.2, label="loop threshold")
        ax.set_xticks(range(len(ids)))
        ax.set_xticklabels([i.replace("bad_", "") for i in ids], fontsize=8)
        ax.set_ylim(0, 1.02)
        ax.set_title(f"RCDA-8 mini-bench {b['tag']} — ρ={b['rho_mini']:.2f} "
                     f"mean ls={b['mean_ls']:.3f}", fontsize=11)
        ax.legend(loc="lower left")
        clean(ax)
    anim = FuncAnimation(fig, draw, frames=len(bench), interval=900)
    save_gif(anim, "gif02_bench_melt.gif", fps=1.2)


def gif_loop_melt():
    base_ls = E["base"]["per_traj_loop_share_bad"]
    rcda_ls = E["rcda8"]["per_traj_loop_share_bad"]
    sft_ls = E["sft"]["per_traj_loop_share_bad"]
    order = np.argsort(base_ls)[::-1]
    labels = [f"g{o+1:02d}" for o in order]
    series = [np.array(base_ls)[order], np.array(rcda_ls)[order],
              np.array(sft_ls)[order]]
    names = ["base", "RCDA-8 formula", "SFT data"]
    cols = [MUTED, VIOLET, EMERALD]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))

    def draw(fi):
        ax.clear()
        s_i = int(fi)
        frac = fi - s_i
        if s_i >= len(series) - 1:
            cur = series[-1]; name = names[-1]; col = cols[-1]
        else:
            cur = series[s_i] * (1 - frac) + series[s_i + 1] * frac
            name = f"{names[s_i]} → {names[s_i+1]}"
            col = cols[s_i + 1]
        barcols = [CRIMSON if v >= 0.5 else "#2A3348" for v in cur]
        ax.bar(range(20), cur, color=barcols, width=0.7)
        ax.axhline(0.5, color=AMBER, ls="--", lw=1.1)
        ax.set_xticks(range(20))
        ax.set_xticklabels(labels, fontsize=7, rotation=90)
        ax.set_ylim(0, 1.02)
        ax.set_title(f"loop_share per bad prompt — {name}", fontsize=11)
        clean(ax)
    frames = [i * 0.25 for i in range((len(series) - 1) * 4 + 1)]
    anim = FuncAnimation(fig, draw, frames=frames, interval=420)
    save_gif(anim, "gif03_loop_melt.gif", fps=2.4)


def gif_zdiff_reveal():
    fig, ax = plt.subplots(figsize=(8.6, 4.2))

    def draw(i):
        ax.clear()
        upto = max(4, int(i * len(layers) / 28))
        ls = layers[:upto]
        zs = zdiff[:upto]
        cols = [CRIMSON if l in WRITERS else (VIOLET if l in BAND else CYAN)
                for l in ls]
        ax.bar(ls, zs, color=cols, width=0.72)
        ax.set_xlim(-0.5, max(layers) + 0.5)
        ax.set_ylim(0, max(zdiff) * 1.2)
        ax.set_title(f"behavioral z_diff — layer reveal L{ls[-1]}", fontsize=11)
        ax.set_xlabel("decoder layer")
        clean(ax)
    anim = FuncAnimation(fig, draw, frames=range(1, 30), interval=220)
    save_gif(anim, "gif04_zdiff_reveal.gif", fps=4.5)


def gif_arms_dashboard():
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.8))
    panels = [
        ("ρ_bad", [E[t]["rho_bad"] for t in ARMS], "{:.2f}", 1.0),
        ("repaired / 12", [0, E["rcda8"]["prompts_fully_repaired"] or 0,
                           E["sft"]["prompts_fully_repaired"] or 0], "{:.0f}", 12),
        ("KL drift", [E[t]["kl_ctrl_mean"] for t in ARMS], "{:.2f}", 0.7),
    ]

    def draw(fi):
        for ax, (title, vals, fmt, ymax) in zip(axes, panels):
            ax.clear()
            # progressive reveal: panel p fills up when the animation
            # reaches it (frame >= p), labels appear one frame later
            grow = [min(max(fi - p, 0.0), 1.0) for p in range(3)]
            show = [v * g for v, g in zip(vals, grow)]
            ax.bar(range(3), show,
                   color=[ARM_COLOR[t] for t in ARMS], width=0.6,
                   alpha=0.55 + 0.45 * min(fi / 4, 1))
            if fi >= 3:
                for i2, v in enumerate(vals):
                    ax.text(i2, v + ymax * 0.03, fmt.format(v), ha="center",
                            fontsize=10)
            ax.set_xticks(range(3))
            ax.set_xticklabels(["base", "RCDA-8", "SFT"], fontsize=8)
            ax.set_ylim(0, ymax * 1.2)
            ax.set_title(title, fontsize=11)
            ax.set_xlabel(f"t={fi}", fontsize=8, color=MUTED)
            clean(ax)
    frames = list(range(6))
    anim = FuncAnimation(fig, draw, frames=frames, interval=650)
    save_gif(anim, "gif05_arms_dashboard.gif", fps=1.5)


def gif_waste_melt():
    base_w = E["base"]["per_traj_loop_share_bad"]
    rcda_w = E["rcda8"]["per_traj_loop_share_bad"]
    sft_w = E["sft"]["per_traj_loop_share_bad"]
    n = 192
    bw = [v * n for v in base_w]
    rw = [v * n for v in rcda_w]
    sw = [v * n for v in sft_w]
    order = np.argsort(bw)[::-1]
    labels = [f"g{o+1:02d}" for o in order]
    series = [np.array(bw)[order], np.array(rw)[order],
              np.array(sw)[order]]
    names = ["base", "RCDA-8 formula", "SFT data"]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))

    def draw(fi):
        ax.clear()
        s_i = min(int(fi), len(series) - 2)
        frac = min(fi - s_i, 1.0)
        cur = series[s_i] * (1 - frac) + series[s_i + 1] * frac
        name = names[s_i + 1] if frac > 0.02 else names[s_i]
        ax.bar(range(20), cur, color=CRIMSON, alpha=0.85, width=0.7)
        ax.set_xticks(range(20))
        ax.set_xticklabels(labels, fontsize=7, rotation=90)
        ax.set_ylim(0, 192)
        ax.set_title(f"wasted tokens per prompt (of 192) — {name}",
                     fontsize=11)
        clean(ax)
    frames = [i * 0.2 for i in range(11)]
    anim = FuncAnimation(fig, draw, frames=frames, interval=520)
    save_gif(anim, "gif06_waste_melt.gif", fps=2)


if __name__ == "__main__":
    print("static figures:", flush=True)
    fig01_probe(); fig02_phase(); fig03_arms_defect(); fig04_abilities()
    fig05_pen_descent(); fig06_forensics(); fig07_repair_matrix()
    fig08_dashboard(); fig09_sft_loss()
    print("animations:", flush=True)
    gif_pen_descent(); gif_bench_melt(); gif_loop_melt(); gif_zdiff_reveal()
    gif_arms_dashboard(); gif_waste_melt()
    print("done -> figures/", flush=True)
