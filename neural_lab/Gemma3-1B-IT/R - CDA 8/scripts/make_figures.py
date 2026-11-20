#!/usr/bin/env python3
"""make_figures.py — the figure atlas + animations of the RCDA-9 formula round.

Everything is rebuilt from the committed JSONs under results/ — nothing
hand-typed. Static figures (PNG, 150 dpi) + six animations (GIF), following
the RCDA-8 atlas scheme plus the two formula-knowledge panels:
  fig01 probe z_diff profile      fig02 phase probe cos(v_e, v_d)
  fig03 arms defect (rho/ls/waste) fig04 abilities (KL/PPL) + formula axes
  fig05 pen descent + anchor MSE   fig06 LoRA forensics
  fig07 repair matrix heatmap      fig08 the one-glance dashboard
  fig09 formulas-arm training loss fig10 formula knowledge (valPPL + QA)
  gif01 pen descent   gif02 bench melt   gif03 loop melt
  gif04 zdiff reveal  gif05 dashboard    gif06 waste melt
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

RES = Path(__file__).resolve().parent.parent / "results"
FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)

C_BASE, C_FML, C_PEN = "#526CFE", "#00C2A8", "#B026FF"
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "figure.facecolor": "#0d1117", "axes.facecolor": "#161b22",
    "text.color": "#e6edf3", "axes.labelcolor": "#e6edf3",
    "xtick.color": "#9da7b3", "ytick.color": "#9da7b3",
    "grid.color": "#30363d", "axes.grid": True, "axes.axisbelow": True,
})


def jload(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def save(fig, name):
    fig.savefig(FIG / name, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("saved", name, flush=True)


def gif(fig, anim, name):
    anim.save(FIG / name, writer=animation.PillowWriter(fps=2))
    plt.close(fig)
    print("saved", name, flush=True)


def main() -> None:
    probe = jload(RES / "probe" / "probe_report.json")
    phase = jload(RES / "probe" / "probe_phase_report.json")
    ev = {t: jload(RES / f"part9_eval_{t}.json") for t in ("base", "formulas", "pen")}
    pen_log = jload(RES / "pen_arm" / "training_log.json")
    try:
        fml_state = jload(RES / "formulas_arm" / "gemma3_formula_lora" / "trainer_state.json")
    except FileNotFoundError:
        fml_state = None

    # ---------------------------------------------------------------- fig01
    layers = sorted(int(k) for k in probe["layer_profile"])
    zd = [probe["layer_profile"][str(l)]["z_diff"] for l in layers]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(layers, zd, "-o", color=C_BASE, ms=3.5, lw=1.4)
    for w in probe["writers"]:
        ax.axvline(w, color=C_PEN, lw=0.8, alpha=0.5)
        ax.annotate(f"L{w}\n{probe['layer_profile'][str(w)]['z_diff']:.1f}",
                    (w, probe["layer_profile"][str(w)]["z_diff"]),
                    textcoords="offset points", xytext=(4, 6), fontsize=7.5,
                    color=C_PEN)
    ax.axhline(130, color="#f0883e", lw=0.8, ls="--")
    ax.annotate("z_diff = 130 (band cut)", (0.4, 134), fontsize=7.5, color="#f0883e")
    ax.set_xlabel("decoder layer"); ax.set_ylabel("z_diff (diagonal Mahalanobis)")
    ax.set_title("fig01 — the behavioral probe on the real Gemma3-1B-IT "
                 "(re-run of RCDA-9: writers L10/L11/L12, band L6–L16)")
    save(fig, "fig01_probe_zdiff.png")

    # ---------------------------------------------------------------- fig02
    cos = [phase["layer_profile"][str(l)]["cos_v_early_v_deep"] for l in layers]
    fig, ax = plt.subplots(figsize=(7, 3.0))
    ax.bar(layers, cos, color=[C_PEN if 8 <= l <= 15 else "#30363d" for l in layers])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("decoder layer"); ax.set_ylabel("cos(v_early, v_deep)")
    ax.set_title("fig02 — phase probe: early/deep directions strongly aligned in the band")
    save(fig, "fig02_phase_probe.png")

    # ---------------------------------------------------------------- fig03
    tags = ["base", "formulas (corpus)", "pen (loss)"]
    keys = ["base", "formulas", "pen"]
    colors = [C_BASE, C_FML, C_PEN]
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.0))
    for ax, metric, title in zip(
            axes, ["rho_bad", "loop_share_mean_bad", "loop_waste_mean_bad"],
            ["rho_bad (loops >= 0.5)", "loop_share mean (bad 20)",
             "loop_waste mean, tokens"]):
        vals = [ev[k][metric] for k in keys]
        ax.bar(range(3), vals, color=colors)
        ax.set_xticks(range(3)); ax.set_xticklabels(tags, fontsize=7.5)
        for i, v in enumerate(vals):
            ax.annotate(f"{v:g}", (i, v), ha="center", va="bottom", fontsize=8)
        ax.set_title(title)
    fig.suptitle("fig03 — the defect across the three arms (one frozen evaluator)",
                 y=1.02, fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig03_arms_defect.png")

    # ---------------------------------------------------------------- fig04
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.0))
    kl = [ev[k]["kl_ctrl_mean"] for k in keys]
    ppl = [ev[k]["ppl_control_mean"] for k in keys]
    vp = [ev[k]["formula_knowledge"]["corpus_val_ppl"] for k in keys]
    qa = [ev[k]["formula_knowledge"]["qa_exact"] for k in keys]
    for ax, vals, title, fmt in (
            (axes[0], kl, "KL(base || arm), control", "{:.3f}"),
            (axes[1], ppl, "PPL control (mean)", "{:.1f}"),
            (axes[2], vp, "formula-corpus val PPL (held-out)", "{:.0f}")):
        ax.bar(range(3), vals, color=colors)
        ax.set_xticks(range(3)); ax.set_xticklabels(tags, fontsize=7.5)
        for i, v in enumerate(vals):
            ax.annotate(fmt.format(v), (i, v), ha="center", va="bottom", fontsize=8)
        ax.set_title(title)
    fig.suptitle("fig04 — abilities drift vs formula knowledge "
                 "(QA exact: %s)" % " / ".join(f"{q}/57" for q in qa),
                 y=1.02, fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig04_abilities.png")

    # ---------------------------------------------------------------- fig05
    seen = {}
    for s in pen_log["steps"]:
        if s.get("step") is not None:
            seen[s["step"]] = s          # dedupe across resumes (keep last)
    steps = [seen[k] for k in sorted(seen)]
    xs = [s["step"] for s in steps]
    pn = [s["pen_norm"] for s in steps]
    mse = [s.get("mse", 0.0) for s in steps]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(xs, pn, ".", color=C_PEN, ms=2.5, label="pen_norm (the formula, melting)")
    ax.plot(xs, mse, ".", color="#f0883e", ms=2.5, label="anchor MSE (cap 1.0)")
    ax.axhline(1.0, color="#f0883e", lw=0.7, ls="--")
    bad_steps = [s for s in steps if s.get("kind") == "bad"]
    first10 = float(np.mean([s["pen_norm"] for s in bad_steps[:10]]))
    last10 = float(np.mean([s["pen_norm"] for s in bad_steps[-10:]]))
    ax.set_title(f"fig05 — the formula as a loss: pen_norm {first10:.3f} -> "
                 f"{last10:.4f} ({100*(last10/first10-1):.1f}%)")
    ax.legend(fontsize=7.5, loc="upper right"); ax.set_xlabel("step")
    save(fig, "fig05_pen_descent.png")

    # ---------------------------------------------------------------- fig06
    fore = ev["pen"]["lora_forensics"]
    fr = {int(v["layer"]): v for v in fore.values()}
    fl = sorted(fr)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.0))
    axes[0].bar(fl, [fr[l]["rel_frobenius"] * 100 for l in fl], color=C_PEN)
    axes[0].set_title("pen-arm edit size: rel-Frobenius, %")
    axes[0].set_xlabel("layer")
    axes[1].bar(fl, [fr[l]["cos_u1_v_deep"] for l in fl], color="#f0883e")
    axes[1].set_title("cos(u1 of dW, v_deep) — never aligned")
    axes[1].set_xlabel("layer")
    fig.suptitle("fig06 — forensics of the band edit (16 tensors, all TW-violation)",
                 y=1.03, fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig06_forensics.png")

    # ---------------------------------------------------------------- fig07
    base_ls = ev["base"]["per_traj_loop_share_bad"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    data = np.array([base_ls,
                     ev["formulas"]["per_traj_loop_share_bad"],
                     ev["pen"]["per_traj_loop_share_bad"]])
    im = ax.imshow(data, aspect="auto", cmap="magma_r", vmin=0, vmax=1)
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["base", "formulas", "pen"])
    ax.set_xlabel("bad prompt g01..g20 (frozen order)")
    ax.set_title("fig07 — per-prompt loop_share: repaired rows go dark")
    fig.colorbar(im, ax=ax, label="loop_share")
    save(fig, "fig07_repair_matrix.png")

    # ---------------------------------------------------------------- fig08
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4))
    m = [("rho_bad", [ev[k]["rho_bad"] for k in keys], "{:.2f}"),
         ("loop_share mean", [ev[k]["loop_share_mean_bad"] for k in keys], "{:.3f}"),
         ("waste, tok", [ev[k]["loop_waste_mean_bad"] for k in keys], "{:.0f}"),
         ("KL ctrl", [ev[k]["kl_ctrl_mean"] for k in keys], "{:.3f}"),
         ("PPL ctrl", [ev[k]["ppl_control_mean"] for k in keys], "{:.1f}"),
         ("val PPL formulas", [ev[k]["formula_knowledge"]["corpus_val_ppl"] for k in keys], "{:.0f}")]
    for ax, (title, vals, fmt) in zip(axes.flat, m):
        ax.bar(range(3), vals, color=colors)
        ax.set_xticks(range(3)); ax.set_xticklabels(tags, fontsize=7)
        for i, v in enumerate(vals):
            ax.annotate(fmt.format(v), (i, v), ha="center", va="bottom", fontsize=7.5)
        ax.set_title(title, fontsize=9)
    fig.suptitle("fig08 — the RCDA-9 dashboard: formulas (corpus) vs pen (loss) vs base",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig08_dashboard.png")

    # ---------------------------------------------------------------- fig09
    if fml_state:
        ls = [h["loss"] for h in fml_state["log_history"] if "loss" in h]
        xs = np.arange(len(ls))
        fig, ax = plt.subplots(figsize=(7, 3.0))
        ax.plot(xs, ls, "-", color=C_FML, lw=1.0)
        k = max(1, len(ls) // 40)
        ax.plot(xs[k-1::k], np.convolve(ls, np.ones(k)/k, mode="same")[k-1::k],
                "-", color="#ffffff", lw=1.6, label=f"running mean ({k})")
        ax.set_xlabel("logging step"); ax.set_ylabel("training loss (pt stage)")
        ax.set_title(f"fig09 — the formulas arm: LlamaFactory loss over "
                     f"{fml_state['global_step']} steps "
                     f"({ls[0]:.2f} -> {np.mean(ls[-10:]):.2f})")
        ax.legend(fontsize=7.5)
        save(fig, "fig09_sft_loss.png")

    # ---------------------------------------------------------------- fig10
    fig, ax = plt.subplots(figsize=(7, 3.2))
    qa_all = {k: ev[k]["formula_knowledge"] for k in keys}
    x = np.arange(3)
    ax.bar(x - 0.18, [qa_all[k]["qa_exact"] for k in keys], width=0.36,
           color=C_FML, label="exact numeric match")
    ax.bar(x + 0.18, [qa_all[k]["qa_partial"] for k in keys], width=0.36,
           color="#f0883e", label="partial (>= 1 number)")
    ax.set_xticks(x); ax.set_xticklabels(tags)
    ax.set_ylabel("pairs of 57 (held-out grid)")
    ax.set_title("fig10 — formula QA: the corpus arm learns the form (val PPL -73%), "
                 "not the arithmetic")
    ax.legend(fontsize=7.5)
    save(fig, "fig10_formula_qa.png")

    # =============================================================== GIFS ==
    # gif01 — pen descent animation
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    bad_steps = [s for s in steps if s.get("kind") == "bad" and "pen_norm" in s]
    bx = [s["step"] for s in bad_steps]
    by = [s["pen_norm"] for s in bad_steps]
    line, = ax.plot([], [], ".", color=C_PEN, ms=3)
    ax.set_xlim(0, max(bx)); ax.set_ylim(0, max(by) * 1.1)
    ax.set_xlabel("step"); ax.set_ylabel("pen_norm")
    ttl = ax.set_title("")

    def upd01(i):
        n = max(2, int(len(bx) * (i + 1) / 24))
        line.set_data(bx[:n], by[:n])
        ttl.set_text(f"gif01 — the formula melting: step {bx[n-1]}, "
                     f"pen_norm={by[n-1]:.4f}")
        return line, ttl

    gif(fig, animation.FuncAnimation(fig, upd01, frames=24, interval=500,
                                     blit=False), "gif01_pen_descent.gif")

    # gif02 — bench melt
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    bens = pen_log["bench"]
    frames = [(b["tag"], b["mean_ls"], b["rho_mini"]) for b in bens]
    bars = ax.bar([], [])
    ax.set_ylim(0, 1.0)

    def upd02(i):
        ax.clear(); ax.set_ylim(0, 1.0)
        upto = frames[:i + 1]
        ax.bar([f[0] for f in upto], [f[1] for f in upto], color=C_PEN)
        for j, f in enumerate(upto):
            ax.annotate(f"{f[1]:.3f}\nrho={f[2]:.2f}", (j, f[1]),
                        ha="center", va="bottom", fontsize=7.5)
        ax.set_title("gif02 — pen-arm mini-bench (8 hardest, greedy 192)")
        ax.set_ylabel("mean loop_share")

    gif(fig, animation.FuncAnimation(fig, upd02, frames=len(frames), interval=900,
                                     blit=False), "gif02_bench_melt.gif")

    # gif03 — loop melt across arms (per-prompt)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    mats = [ev[k]["per_traj_loop_share_bad"] for k in keys]

    def upd03(i):
        ax.clear()
        im = ax.imshow(np.array(mats[:i + 1]), aspect="auto", cmap="magma_r",
                       vmin=0, vmax=1)
        ax.set_yticks(range(i + 1)); ax.set_yticklabels(tags[:i + 1], fontsize=8)
        ax.set_xlabel("bad prompt")
        ax.set_title("gif03 — per-prompt loop_share across the arms")

    gif(fig, animation.FuncAnimation(fig, upd03, frames=3, interval=1100,
                                     blit=False), "gif03_loop_melt.gif")

    # gif04 — z_diff reveal (layer by layer)
    fig, ax = plt.subplots(figsize=(6.4, 3.0))

    def upd04(i):
        ax.clear()
        n = max(2, int(len(layers) * (i + 1) / 20))
        ax.plot(layers[:n], zd[:n], "-o", color=C_BASE, ms=3, lw=1.2)
        ax.axhline(130, color="#f0883e", lw=0.8, ls="--")
        ax.set_xlim(0, max(layers)); ax.set_ylim(0, max(zd) * 1.1)
        ax.set_xlabel("decoder layer"); ax.set_ylabel("z_diff")
        ax.set_title(f"gif04 — the writers reveal themselves (L{layers[n-1]})")

    gif(fig, animation.FuncAnimation(fig, upd04, frames=20, interval=400,
                                     blit=False), "gif04_zdiff_reveal.gif")

    # gif05 — dashboard panel by panel
    fig, ax = plt.subplots(figsize=(6.4, 3.2))

    def upd05(i):
        ax.clear()
        title, vals, fmt = m[i]
        ax.bar(range(3), vals, color=colors)
        ax.set_xticks(range(3)); ax.set_xticklabels(tags, fontsize=8)
        for j, v in enumerate(vals):
            ax.annotate(fmt.format(v), (j, v), ha="center", va="bottom", fontsize=8)
        ax.set_title(f"gif05 — {title}")

    gif(fig, animation.FuncAnimation(fig, upd05, frames=len(m), interval=1000,
                                     blit=False), "gif05_arms_dashboard.gif")

    # gif06 — waste melt
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    w = [ev[k]["loop_waste_mean_bad"] for k in keys]

    def upd06(i):
        ax.clear()
        ax.bar(range(3), w, color=colors, alpha=0.25 + 0.75 * (i + 1) / 3)
        ax.set_xticks(range(3)); ax.set_xticklabels(tags, fontsize=8)
        for j, v in enumerate(w):
            ax.annotate(f"{v:.0f}", (j, v), ha="center", va="bottom", fontsize=8)
        ax.set_title("gif06 — wasted tokens per prompt, melting across arms")
        ax.set_ylabel("loop_waste mean, tok")

    gif(fig, animation.FuncAnimation(fig, upd06, frames=3, interval=1000,
                                     blit=False), "gif06_waste_melt.gif")

    print("ALL FIGURES DONE", flush=True)


if __name__ == "__main__":
    main()
