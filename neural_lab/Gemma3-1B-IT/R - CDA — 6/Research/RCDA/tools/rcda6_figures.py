#!/usr/bin/env python3
"""rcda6_figures.py — the RCDA-6 figure set (mirrors the RCDA-5 kit).

  figC1_probe_phase.png   phase probe: z_early vs z_deep per layer + the
                          cos(v_early, v_deep) split — WHY the phase
                          conditioning matters (the RCDA-5 report hypothesis,
                          measured)
  figC2_band.png          the trainable set: RCDA-5 collapse-probe z_diff
                          profile with the L8-L15 band and the 16 edited
                          tensors marked
  figC3_training.png      150-step record: pen_norm (dimensionless), mse vs
                          the hard cap, deep-phase share of the penalty
                          positions
  figC4_eval_arms.png     the full-benchmark summary across the four states:
                          base / RCDA-5 SGD / RCDA-5 Adam / RCDA-6
  figC5_migration.png     per-prompt loop-share migration base -> RCDA-6
                          (hard loopers relieve, mild prompts break)
  figC6_spectral.png      the 16 edited tensors: rel_Frobenius, TW z_resid
                          and cos(u1, v) — the spectral record of the edit

Usage: rcda6_figures.py --work WORK --results RESULTS_DIR --graphic GRAPHIC_DIR
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CODE = Path(__file__).resolve().parents[3] / "Code"
sys.path.insert(0, str(CODE))
import gemma_lab as L                                    # noqa: E402

C_BASE = "#8a94a6"
C_RCDA5 = "#e05263"
C_RCDA6 = "#12b886"
C_ANCHOR = "#f59f00"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=None)
    ap.add_argument("--results", required=True)
    ap.add_argument("--graphic", required=True)
    ap.add_argument("--rcda5-results", default=None,
                    help="RCDA-5 results dir (for the arm comparison)")
    args = ap.parse_args()

    res = Path(args.results)
    out = Path(args.graphic)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(args.work) if args.work else L.resolve_work()
    R5 = Path(__file__).resolve().parents[4] / "R CDA - 5" / "Research/RCDA/results"

    pp = json.load((res / "probe_phase_report.json").open())
    log = json.load((res / "training_log.json").open())
    e6 = json.load((res / "part6_eval_rcda6_trained.json").open())
    digest = json.load((res / "rcda6_spectral_digest.json").open())
    ppl = json.load((res / "ppl_check.json").open())

    # ---------------- figC1: the phase probe ----------------
    layers = list(range(26))
    prof = pp["layer_profile"]
    z_e = [prof[str(l)].get("z_early") or 0 for l in layers]
    z_d = [prof[str(l)].get("z_deep") or 0 for l in layers]
    cos_ed = [prof[str(l)].get("cos_v_early_v_deep") or 0 for l in layers]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True,
                                   constrained_layout=True)
    band = pp["band"]
    ax1.axvspan(band[0] - 0.5, band[-1] + 0.5, color="#12b886", alpha=0.10)
    ax1.plot(layers, z_e, "o-", color="#f59f00", label="z early phase")
    ax1.plot(layers, z_d, "s-", color="#12b886", label="z deep phase")
    w5 = json.load((R5 / "probe_report.json").open())["writers"]
    for w in w5:
        ax1.axvline(w, color="#8a94a6", ls=":", lw=1)
    ax1.set_ylabel("diagonal-Mahalanobis z_diff")
    ax1.set_title("RCDA-6 phase probe (Gemma3-1B-IT): the loop direction is a "
                  "deep-phase phenomenon\n"
                  "green band = the hypothesis band L8-L15; dotted = RCDA-5 "
                  "writers L10-12")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.3)
    ax2.plot(layers, cos_ed, "d-", color="#e05263")
    ax2.axhline(1.0, color="#8a94a6", lw=0.8)
    ax2.set_ylabel("cos(v_early, v_deep)")
    ax2.set_xlabel("decoder layer")
    ax2.set_title("the two phase directions are distinct (cos < 1) — "
                  "a single collapse-point direction mixes them")
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 1.05)
    fig.savefig(out / "figC1_probe_phase.png", dpi=130)
    plt.close(fig)

    # ---------------- figC2: trainable set on the z_diff profile ----------------
    r5 = json.load((R5 / "probe_report.json").open())
    zd = [r5["layer_profile"][str(l)]["z_diff"] for l in layers]
    fig, ax = plt.subplots(figsize=(10, 4.6), constrained_layout=True)
    ax.axvspan(band[0] - 0.5, band[-1] + 0.5, color="#12b886", alpha=0.10,
               label="RCDA-6 trainable band L8-L15")
    ax.plot(layers, zd, "o-", color="#4263eb", label="collapse-probe z_diff (RCDA-5)")
    for l in band:
        for t, mk in (("o_proj", "s"), ("down_proj", "^")):
            ax.plot([l], [zd[l]], mk, color="#12b886", ms=9,
                    label="RCDA-6 edited tensor" if l == band[0]
                    and t == "o_proj" else None)
    for w in w5:
        ax.axvline(w, color="#e05263", ls=":", lw=1.2)
    ax.set_xlabel("decoder layer")
    ax.set_ylabel("z_diff")
    ax.set_title("Trainable set: 8 layers x {o_proj, down_proj} = 73.1 M "
                  "parameters (7.31 %) vs RCDA-5's 3.5 M (o_proj L10-12 only,\n"
                  "dotted red); the band is the RCDA-5 report hypothesis "
                  "(z_diff > 130 on all of L8-L15)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.savefig(out / "figC2_band.png", dpi=130)
    plt.close(fig)

    # ---------------- figC3: training curves ----------------
    steps = [s["step"] for s in log["steps"]]
    pn = [s["pen_norm"] for s in log["steps"]]
    mse = [s["mse"] for s in log["steps"]]
    ds = [s.get("deep_share", 0) for s in log["steps"]]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                   constrained_layout=True)
    ax1.semilogy(steps, pn, color="#12b886", lw=1.2)
    ax1.set_ylabel("pen_norm (dimensionless, init 1.0)")
    ax1.set_title("RCDA-6 training, 150 steps SGD (lr 3e-3, momentum 0.9, "
                  "clip 1.0), batch 1 — the phase-matched penalty\n"
                  f"pen_norm 1.0 -> {np.mean(pn[-50:]):.3f} (mean of last 50, "
                  f"-{100*(1-np.mean(pn[-50:])):.1f} %)")
    ax1.grid(alpha=0.3, which="both")
    ax1.axhline(0.2236, color="#e05263", ls="--", lw=1,
               label="RCDA-5 Adam arm final (0.224)")
    ax1.legend(loc="upper right", fontsize=9)
    ax2.plot(steps, mse, color="#4263eb", lw=1.0, label="anchor MSE (trailing 32 logits)")
    ax2.axhline(1.0, color="#f59f00", ls="--", lw=1.2, label="hard cap")
    ax2.set_ylabel("anchor MSE")
    ax2.set_xlabel("step")
    ax2.set_title(f"the abilities anchor rides the cap ({sum(1 for m in mse if m > 1)}/150 "
                  f"steps above cap) — 74 M trainable params push harder")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(alpha=0.3)
    ax3 = ax2.twinx()
    ax3.plot(steps, ds, color="#8a94a6", lw=0.8, alpha=0.6)
    ax3.set_ylabel("deep-phase share of penalty positions", color="#8a94a6")
    ax3.tick_params(axis="y", colors="#8a94a6")
    fig.savefig(out / "figC3_training.png", dpi=130)
    plt.close(fig)

    # ---------------- figC4: four arms, one evaluator ----------------
    r5dir = Path(args.rcda5_results) if args.rcda5_results else R5
    e5b = json.load((r5dir / "part5_eval_base.json").open())
    e5t = json.load((r5dir / "part5_eval_trained.json").open())
    e5a = json.load((r5dir / "part5_eval_adam.json").open())
    arms = [("base", e5b, C_BASE), ("RCDA-5 SGD-150", e5t, C_RCDA5),
            ("RCDA-5 Adam-150", e5a, "#be4b63"), ("RCDA-6 SGD-150", e6, C_RCDA6)]
    metrics = [("rho_bad", "ρ_bad", 1), ("loop_share_mean_bad", "loop share mean (bad)", 1),
               ("loop_waste_mean", "loop waste mean (bad), tok", 1)]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), constrained_layout=True)
    for ax, (key, label, _) in zip(axes, metrics):
        vals = [a[1][key] for a in arms]
        ax.bar(range(len(arms)), vals, color=[a[2] for a in arms], width=0.62)
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([a[0] for a in arms], rotation=18, fontsize=8)
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=0.3, axis="y")
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3g}" if isinstance(v, float) else str(v),
                    ha="center", va="bottom", fontsize=9)
    fig.suptitle("Full benchmark (20 bad + 20 good, greedy 192): RCDA-6 halves "
                 "the RCDA-5 damage but does not beat base on the full set", fontsize=11)
    fig.savefig(out / "figC4_eval_arms.png", dpi=130)
    plt.close(fig)

    # ---------------- figC5: per-prompt migration ----------------
    rows_b = {r["id"]: r for r in e5b["rows"]}
    rows_6 = {r["id"]: r for r in e6["rows"]}
    gids = sorted(g for g in rows_b if g.startswith("bad"))
    xb = [rows_b[g]["loop_share"] for g in gids]
    y6 = [rows_6[g]["loop_share"] for g in gids]
    fig, ax = plt.subplots(figsize=(7.2, 7.2), constrained_layout=True)
    lim = (0, 1.05)
    ax.plot(lim, lim, color="#8a94a6", lw=1, ls="--", label="no change")
    ax.scatter(xb, y6, c=["#e05263" if rows_6[g]["looping"] and not rows_b[g]["looping"]
                          else "#12b886" if rows_b[g]["looping"] and not rows_6[g]["looping"]
                          else "#4263eb" for g in gids], s=70, zorder=3)
    for g, x, y in zip(gids, xb, y6):
        ax.annotate(g.replace("bad_", ""), (x, y), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.axhline(0.5, color="#8a94a6", lw=0.8, ls=":")
    ax.axvline(0.5, color="#8a94a6", lw=0.8, ls=":")
    ax.set_xlabel("loop share, base")
    ax.set_ylabel("loop share, RCDA-6 trained")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_title("Per-prompt migration (20 bad prompts): green = unlooped by the edit,\n"
                 "red = pushed into looping; the hard loopers (right) relieve,\n"
                 "the mild prompts (left) break — the defect migrates")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.savefig(out / "figC5_migration.png", dpi=130)
    plt.close(fig)

    # ---------------- figC6: spectral record ----------------
    dg = digest["digest"]
    keys = sorted(dg)
    rel = [dg[k]["rel_frobenius"] for k in keys]
    zr = [dg[k]["z_resid_tw"] for k in keys]
    cd = [dg[k]["cos_u1_v_deep"] for k in keys]
    ce = [dg[k]["cos_u1_v_early"] for k in keys]
    labels = [k.replace("model.layers.", "L").replace(".self_attn.o_proj.weight", "/o")
              .replace(".mlp.down_proj.weight", "/d") for k in keys]
    x = np.arange(len(keys))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True,
                                   constrained_layout=True)
    ax1.bar(x, rel, color=["#12b886" if "/d" in lb else "#4263eb" for lb in labels],
            width=0.6)
    ax1.set_ylabel("rel_Frobenius of the edit")
    ax1.set_title("RCDA-6 spectral record: 16 edited tensors (green = down_proj, "
                  "blue = o_proj)\nTW verdict: violation x16 — the residual bulk stays "
                  "structured (no TW-clean solution on Gemma3, as in RCDA-5)")
    ax1.grid(alpha=0.3, axis="y")
    w = 0.38
    ax2.bar(x - w / 2, cd, w, label="cos(u1, v_deep)", color="#12b886")
    ax2.bar(x + w / 2, ce, w, label="cos(u1, v_early)", color="#f59f00")
    ax2.set_ylabel("cos(u1, v)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, fontsize=7.5)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3, axis="y")
    ax2.set_ylim(0, 0.75)
    fig.savefig(out / "figC6_spectral.png", dpi=130)
    plt.close(fig)

    print(f"figures -> {out}")
    for f in sorted(out.glob("figC*.png")):
        print(" ", f.name)


if __name__ == "__main__":
    main()
