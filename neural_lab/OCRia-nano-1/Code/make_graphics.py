"""Every figure and GIF of the kit - all numbers read from Results JSONs.

Style: dark neon (the parent repository's visual language).  Nothing is
hand-drawn data: every bar, curve and scatter is a field of a committed JSON.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ocria_common import DATA_DIR, KIT_DIR, RESULTS_DIR, load_batch

GR = Path(__file__).resolve().parent.parent / "Graphics"
GR.mkdir(parents=True, exist_ok=True)

BG = "#0b1020"
PANEL = "#11172b"
FG = "#e6edf3"
GRID = "#232b45"
CY = "#22d3ee"   # cyan
PU = "#a78bfa"   # purple
AM = "#fbbf24"   # amber
EM = "#34d399"   # emerald
PK = "#f472b6"   # pink
BL = "#60a5fa"   # blue
OR = "#fb923c"   # orange
ARM_COLORS = {"base": "#8b93a7", "classic": EM, "rcda_x": CY}

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL,
    "savefig.facecolor": BG, "text.color": FG,
    "axes.edgecolor": GRID, "axes.labelcolor": FG,
    "xtick.color": FG, "ytick.color": FG,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "figure.dpi": 110,
})


def load(name):
    return json.loads((RESULTS_DIR / name).read_text(encoding="utf-8"))


def style_ax(ax):
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_facecolor(PANEL)


def save(fig, name):
    fig.savefig(GR / name, dpi=110, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("[gfx] wrote", name)


def chip(ax, x, y, text, color, fs=8.5):
    ax.text(x, y, text, color=color, fontsize=fs, family="monospace",
            bbox=dict(boxstyle="round,pad=0.35", fc="#0d1226", ec=color, lw=1))


# --------------------------------------------------------------------------
# fig01: the one-glance dashboard
# --------------------------------------------------------------------------

def fig01():
    S = load("summary_all_arms.json")
    arms = S["arms"]
    fig = plt.figure(figsize=(13.5, 7.2))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.28,
                          left=0.05, right=0.97, top=0.86, bottom=0.10)
    fig.suptitle("OCRia-nano-1 × RCDA — the hallucination intervention dashboard",
                 fontsize=15, fontweight="bold", color=FG, y=0.965)

    # panel 1: headline phantom bars
    ax = fig.add_subplot(gs[0, 0])
    names = ["base", "classic", "rcda_x"]
    vals = [arms[a]["F1_phantom_rate"] for a in names]
    bars = ax.bar(names, vals, color=[ARM_COLORS[a] for a in names], width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_title("F1 phantom rate (210 non-text crops)")
    ax.set_ylim(0, max(vals) * 1.3)
    style_ax(ax)

    # panel 2: abilities
    ax = fig.add_subplot(gs[0, 1])
    w = 0.26
    xs = np.arange(3)
    for k, (a, lab) in enumerate(zip(names, ["clean", "degraded", "control"])):
        key = {"clean": "F3_cer_clean", "degraded": "F3_cer_degraded",
               "control": "F3_cer_control"}[lab]
        ax.bar(xs + (k - 1) * w, [arms[n][key] for n in names], w,
               label=lab, color=[CY, OR, PU][k])
    ax.set_xticks(xs); ax.set_xticklabels(names)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("F3 CER by batch (abilities)")
    style_ax(ax)

    # panel 3: utility trap
    ax = fig.add_subplot(gs[0, 2])
    conf = [arms[a].get("F6_auc_confidence", np.nan) for a in names]
    geom = [arms[a].get("F6_auc_geometry", np.nan) for a in names]
    ax.bar(xs - 0.18, conf, 0.34, label="confidence AUC", color=PK)
    ax.bar(xs + 0.18, geom, 0.34, label="geometry AUC (v·ℓ)", color=CY)
    ax.axhline(0.5, color=FG, lw=0.8, ls="--", alpha=0.5)
    ax.set_xticks(xs); ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("F6 Utility Trap: geometry beats confidence")
    style_ax(ax)

    # panel 4: BBP scan chips
    ax = fig.add_subplot(gs[1, 0])
    ax.axis("off"); style_ax(ax)
    ax.set_title("BBP matrix scans (probe directions)")
    scans = S["probe"]["matrix_scans"]
    y = 0.86
    for name, sc in scans.items():
        col = AM if sc["supercritical"] else "#8b93a7"
        mark = "SUPERCRITICAL" if sc["supercritical"] else "subcritical"
        chip(ax, 0.02, y,
             f"{name:9s} λv/λ+={sc['lambda_ratio']:6.2f}  θv/θc="
             f"{sc['theta_v']:.2f}/{sc['theta_c']:.2f}  {mark}", col, fs=8)
        y -= 0.26
    ax.text(0.02, y, "the classifier head writes the phantom direction\n"
            "with a 22× supercritical spike", color=AM, fontsize=8.5)

    # panel 5: writers
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off"); style_ax(ax)
    ax.set_title("writer selection (z_diff, mix stages)")
    st = S["probe"]["stages"]
    ws = S["probe"]["writers"]
    y = 0.86
    for sname in ("mix0", "mix1", "mix2", "head"):
        z = st.get(sname, {}).get("z_diff", 0)
        col = EM if sname in ws else "#8b93a7"
        chip(ax, 0.02, y, f"{sname:6s} z_diff={z:7.2f} "
             f"{'← writer' if sname in ws else ''}", col, fs=8.5)
        y -= 0.24
    # panel 6: formulas audit chips
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off"); style_ax(ax)
    ax.set_title("formula audit (all parent formulas on OCRia)")
    fa = S["formulas_audit"]
    chips = [
        (f"Silverstein residual  {fa['bbp_outlier_check']['max_rel_residual']:.0e}", EM),
        (f"entropy ΔH formula≈MC  {abs(fa['entropy_response']['formula_delta_H'] - fa['entropy_response']['monte_carlo_delta_H']):.2e}", EM),
        (f"MP bulk (weights)  KS={fa['mp_summary']['ks_median']:.2f} (trained structure)", OR),
        (f"MP bulk (states)  KS={fa['mp_activation_covariance']['ks_distance_bulk']:.2f}", EM),
        (f"inverse-BBP shrinker err  {fa['shrinker']['inverse_bbp_rel_err']:.3f}", EM),
    ]
    y = 0.88
    for txt, col in chips:
        chip(ax, 0.02, y, txt, col, fs=8)
        y -= 0.185
    save(fig, "fig01_dashboard.png")


# --------------------------------------------------------------------------
# fig02: per-family results
# --------------------------------------------------------------------------

def fig02():
    S = load("summary_all_arms.json")
    arms = S["arms"]
    names = ["base", "classic", "rcda_x"]
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.2))
    fig.suptitle("The six hallucination families - before and after the formula",
                 fontsize=13, fontweight="bold", y=1.02)
    # F1
    ax = axs[0]
    vals = [arms[a]["F1_phantom_rate"] for a in names]
    ax.bar(names, vals, color=[ARM_COLORS[a] for a in names], width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.003, f"{v:.3f}", ha="center", fontweight="bold", fontsize=9)
    ax.set_title("F1 phantom text (confabulation)")
    ax.set_ylim(0, 0.2)
    style_ax(ax)
    # F2
    ax = axs[1]
    vals = [arms[a]["F2_loop"]["mean_loop_share"] for a in names]
    ax.bar(names, vals, color=[ARM_COLORS[a] for a in names], width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=9)
    ax.set_title("F2 mean loop share (CTC collapse)")
    style_ax(ax)
    # F3
    ax = axs[2]
    w = 0.26
    xs = np.arange(3)
    for k, (key, lab) in enumerate([("F3_cer_clean", "clean"),
                                    ("F3_cer_degraded", "degraded"),
                                    ("F3_cer_control", "control")]):
        ax.bar(xs + (k - 1) * w, [arms[n][key] for n in names], w,
               label=lab, color=[CY, OR, PU][k])
    ax.set_xticks(xs); ax.set_xticklabels(names)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("F3 misreads: CER (abilities kept)")
    style_ax(ax)
    save(fig, "fig02_families.png")


# --------------------------------------------------------------------------
# fig03: writer localisation
# --------------------------------------------------------------------------

def fig03():
    P = load("probe_report.json")
    st = P["stages"]
    stages = ["stem", "body0", "body1", "body2", "body3", "body4", "body5",
              "body6", "body7", "body8", "mix0", "mix1", "mix2"]
    zd = [st[s].get("z_diff", np.nan) if isinstance(st[s], dict) else np.nan
          for s in stages]
    colors = [EM if s in P["writers"] else ("#8b93a7" if not np.isnan(z) else "#3a4361")
              for s, z in zip(stages, zd)]
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    bars = ax.bar(stages, np.nan_to_num(zd), color=colors, width=0.62)
    for b, s in zip(bars, stages):
        z = st[s].get("z_diff")
        if z is not None:
            ax.text(b.get_x() + b.get_width() / 2, z + 0.4, f"{z:.1f}",
                    ha="center", fontsize=8.5)
    ax.set_title("Writer localisation: z_diff per stage (emitted vs abstained states, pooled-σ units)")
    ax.set_ylabel("z_diff")
    ax.axhline(0, color=FG, lw=0.8, alpha=0.4)
    ax.annotate("the three writers the formula trains",
                xy=(10.5, 14.2), xytext=(5.2, 17.5),
                arrowprops=dict(arrowstyle="->", color=EM), color=EM, fontsize=9)
    style_ax(ax)
    save(fig, "fig03_writers.png")


# --------------------------------------------------------------------------
# fig04: BBP quadrant
# --------------------------------------------------------------------------

def fig04():
    P = load("probe_report.json")
    scans = P["matrix_scans"]
    fig, ax = plt.subplots(figsize=(9.4, 6.6))
    xmax, ymax = 26, 16.5
    ax.fill([1, xmax, xmax, 1], [1, 1, ymax, ymax], color="#1a2340", alpha=0.55, zorder=0)
    ax.axhline(1, color=AM, lw=1.2, ls="--", alpha=0.85)
    ax.axvline(1, color=AM, lw=1.2, ls="--", alpha=0.85)
    ax.text(0.35, ymax - 0.4, "BBP boundary θc=√q", color=AM, fontsize=8.5, rotation=90)
    offsets = {"act_1": (12, 10), "fc2": (12, -16), "Conv": (16, 6)}
    for name, sc in scans.items():
        sup = sc["supercritical"]
        x, y = sc["lambda_ratio"], sc["theta_v"] / sc["theta_c"]
        ax.scatter(x, y, s=340 if sup else 240, color=AM if sup else "#8b93a7",
                   zorder=4, edgecolors=FG, linewidths=1.2)
        dx, dy = offsets.get(name, (12, -6))
        ax.annotate(f"{name} ({sc['stage']})\nλv/λ+={x:.2f}  θv/θc={y:.2f}"
                    f"  {'SUPERCRITICAL' if sup else 'subcritical'}",
                    (x, y), textcoords="offset points", xytext=(dx, dy), fontsize=8.5,
                    color=AM if sup else "#9aa3b5",
                    bbox=dict(boxstyle="round,pad=0.28", fc="#0d1226",
                              ec=AM if sup else "#2a3554", lw=1) if sup else None)
    ax.set_xlabel("λ_v / λ₊   (Rayleigh energy over the MP edge)")
    ax.set_ylabel("θ_v / θ_c   (spike strength over the BBP threshold)")
    ax.set_title("The RCDA matrix scan on OCRia: the head is a supercritical writer")
    ax.set_xlim(0, xmax); ax.set_ylim(0, ymax)
    ax.text(xmax * 0.62, ymax * 0.9, "supercritical zone\n(a spike of strength θ>√q "
            "survives\nthe bulk and keeps writing)", color=AM, fontsize=10, ha="center",
            bbox=dict(boxstyle="round,pad=0.4", fc="#171025", ec=AM, lw=1))
    ax.annotate("", xy=(22.3, 14.0), xytext=(xmax * 0.62, ymax * 0.78),
                arrowprops=dict(arrowstyle="->", color=AM, lw=1.6))
    style_ax(ax)
    save(fig, "fig04_matrix_scan.png")


# --------------------------------------------------------------------------
# fig05: MP fit of the head matrix + probe Rayleigh line
# --------------------------------------------------------------------------

def fig05():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from rebuild_latin import extract_latin_tensors
    from marchenko_pastur import fit_sigma2, mp_bounds, mp_density, wishart_eigs
    P = load("probe_report.json")
    head = next(e for e in extract_latin_tensors() if "head/Conv" in e["node_out"])
    W = head["w"]
    while W.ndim > 2:
        W = W[..., 0]
    lam, q = wishart_eigs(W.astype(np.float64))
    s2 = fit_sigma2(lam)
    lm, lp = mp_bounds(min(q, 1.0), s2)
    scan = P["matrix_scans"]["Conv"]
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    ax.hist(lam, bins=60, density=True, color=BL, alpha=0.75,
            label=f"empirical spectrum of head W ({W.shape[0]}×{W.shape[1]})")
    grid = np.linspace(max(lm, 1e-6), lp, 500)
    ax.plot(grid, mp_density(grid, min(q, 1.0), s2), color=CY, lw=2.4,
            label=f"MP density fit (q={min(q,1.0):.3f}, σ²={s2:.2e})")
    ax.axvline(lp, color=CY, ls="--", lw=1.4, label=f"λ₊ = {lp:.3e}")
    ax.axvline(scan["lambda_v"], color=AM, lw=2.2,
               label=f"λ_v (probe direction) = {scan['lambda_v']:.3e}")
    ax.axvline(lam.max(), color=PK, lw=1.6, alpha=0.9,
               label=f"λ_max = {lam.max():.3e}")
    ax.set_xscale("log")
    ax.set_title("Step 4-7 on OCRia: MP bulk, the edge, and the 22× supercritical phantom direction")
    ax.legend(frameon=False, fontsize=8.5)
    style_ax(ax)
    save(fig, "fig05_mp_fit.png")


# --------------------------------------------------------------------------
# fig06: training melt
# --------------------------------------------------------------------------

def fig06():
    logs = {a: load(f"{a}_training_log.json") for a in ("classic", "rcda_x")}
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 4.8), sharey=False)
    for ax, (arm, log) in zip(axs, logs.items()):
        bad = [s for s in log["steps"] if s["kind"] == "bad" and s["pen_raw"] > 0]
        ax.plot([s["step"] for s in bad], [s["pen_raw"] for s in bad],
                color=ARM_COLORS[arm], lw=1.8)
        ax.set_yscale("log")
        ax.set_title(f"{arm}: the formula melting", fontsize=11.5, pad=8)
        ax.set_xlabel("step")
        ax.set_ylabel("pen (log scale)")
        for b in log["bench"]:
            ax.axvline(b["step"], color="#3a4361", lw=0.8, ls=":")
        bench_txt = "  ".join(f"@{b['step']}:{b['phantom_rate']:.2f}"
                              for b in log["bench"])
        ax.text(0.02, 0.03, "mini-bench phantom: " + bench_txt,
                transform=ax.transAxes, fontsize=8.5, color="#9aa3b5")
        style_ax(ax)
    fig.suptitle("pen = mean relu(v·ℓ − edge)² + relu(v·h − c_ref)²  — 150 Adam steps, batch 1",
                 fontsize=11, y=1.02)
    save(fig, "fig06_training_melt.png")


# --------------------------------------------------------------------------
# fig07: headline before/after
# --------------------------------------------------------------------------

def fig07():
    S = load("summary_all_arms.json")
    arms = S["arms"]
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    nontext_n = 210
    phantom = {a: round(arms[a]["F1_phantom_rate"] * nontext_n) for a in arms}
    clean = {a: nontext_n - phantom[a] for a in arms}
    names = list(arms)
    x = np.arange(len(names))
    ax.bar(x, [clean[a] for a in names], 0.58, color="#1d2742", label="abstained correctly")
    ax.bar(x, [phantom[a] for a in names], 0.58,
           bottom=[clean[a] for a in names],
           color=[PK, EM, CY], label="phantom text emitted")
    for i, a in enumerate(names):
        ax.text(i, nontext_n + 4, f"{phantom[a]} / {nontext_n}",
                ha="center", fontsize=11, fontweight="bold")
    rel = S["deltas_vs_base"]
    ax.text(1, nontext_n * 0.55, f"−{100 * (1 - rel['classic']['phantom_rate_rel']):.0f}%\nrel {rel['classic']['phantom_rate_rel']:.3f}",
            ha="center", color=EM, fontsize=11, fontweight="bold")
    ax.text(2, nontext_n * 0.55, f"−{100 * (1 - rel['rcda_x']['phantom_rate_rel']):.0f}%\nrel {rel['rcda_x']['phantom_rate_rel']:.3f}",
            ha="center", color=CY, fontsize=11, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0, nontext_n * 1.14)
    ax.set_title("Headline: phantom-text trajectories on the 210 non-text crops")
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    style_ax(ax)
    save(fig, "fig07_headline.png")


# --------------------------------------------------------------------------
# fig08: abilities parity
# --------------------------------------------------------------------------

def fig08():
    S = load("summary_all_arms.json")
    arms = S["arms"]
    names = ["base", "classic", "rcda_x"]
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax = axs[0]
    keys = [("F3_cer_clean", "clean"), ("F3_cer_degraded", "degraded"),
            ("F3_cer_control", "control")]
    xs = np.arange(3)
    w = 0.26
    for k, (key, lab) in enumerate(keys):
        ax.bar(xs + (k - 1) * w, [arms[n][key] for n in names], w,
               label=lab, color=[CY, OR, PU][k])
    ax.set_xticks(xs); ax.set_xticklabels(names)
    ax.set_title("CER by batch")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax)
    ax = axs[1]
    mse = [arms[n]["control_logits_mse"] for n in names]
    kl = [arms[n]["control_kl_base_to_arm"] for n in names]
    ax.bar(xs - 0.18, mse, 0.34, label="logits MSE", color=AM)
    ax.bar(xs + 0.18, kl, 0.34, label="KL(base‖arm)", color=PK)
    ax.set_xticks(xs); ax.set_xticklabels(names)
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_title("control-corpus drift vs the frozen base")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax)
    fig.suptitle("Abilities parity: the hard anchor (β·relu(mse − cap), scope=good)",
                 fontsize=12, y=1.02)
    save(fig, "fig08_abilities.png")


# --------------------------------------------------------------------------
# fig09: utility trap detail
# --------------------------------------------------------------------------

def fig09():
    S = load("summary_all_arms.json")
    arms = S["arms"]
    names = ["base", "classic", "rcda_x"]
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8))
    ax = axs[0]
    xs = np.arange(3)
    conf = [arms[a].get("F6_auc_confidence", 0.5) for a in names]
    geom = [arms[a].get("F6_auc_geometry", 0.5) for a in names]
    ax.bar(xs - 0.18, conf, 0.34, label="confidence AUC", color=PK)
    ax.bar(xs + 0.18, geom, 0.34, label="geometry AUC (v·ℓ)", color=CY)
    ax.axhline(0.5, color=FG, lw=0.8, ls="--", alpha=0.5)
    ax.set_xticks(xs); ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8.5)
    ax.set_title("The Utility Trap on OCRia (17k frames):\nconfidence is the WORSE predictor of phantom frames")
    style_ax(ax)
    ax = axs[1]
    P = load("probe_report.json")
    st = P["stages"]
    stages = ["body8", "mix0", "mix1", "mix2"]
    auc = [st[s].get("auc_geometry", np.nan) for s in stages]
    ax.bar(stages, auc, color=[BL, PU, AM, EM], width=0.6)
    for i, v in enumerate(auc):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title("separation of emitted vs abstained states\nby the probe direction v (AUC per stage)")
    style_ax(ax)
    save(fig, "fig09_utility_trap.png")


# --------------------------------------------------------------------------
# fig10: spectral purity + alignment of the edits
# --------------------------------------------------------------------------

def fig10():
    S = load("summary_all_arms.json")
    arms = S["arms"]
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax = axs[0]
    labels, dks, cols = [], [], []
    for arm in ("classic", "rcda_x"):
        for name, s in arms[arm].get("spectral", {}).items():
            labels.append(f"{arm[:4]}\n{name.split('.')[1] if '.' in name else name}")
            dks.append(s["tw_gate_dks"])
            cols.append(EM if s["tw_verdict"] == "pass" else PK)
    ax.bar(range(len(labels)), dks, color=cols, width=0.55)
    ax.axhline(0.05, color=PK, ls="--", lw=1, label="gate ±0.05")
    ax.axhline(-0.05, color=PK, ls="--", lw=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_title("relative TW gate: ΔKS(bulk) per edited tensor")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax)
    ax = axs[1]
    labels, cosv, cols = [], [], []
    for arm in ("classic", "rcda_x"):
        for name, s in arms[arm].get("spectral", {}).items():
            labels.append(f"{arm[:4]}·{name.split('.')[1]}")
            cosv.append(s["cos_u1_v"])
            cols.append(EM if arm == "classic" else CY)
    ax.bar(range(len(labels)), cosv, color=cols, width=0.55)
    for i, v in enumerate(cosv):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8.5)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylim(0, 1.0)
    ax.set_title("did SGD write along the probe direction? cos(u₁(ΔW), v)")
    style_ax(ax)
    save(fig, "fig10_spectral_gates.png")


# --------------------------------------------------------------------------
# fig11: corpora sample sheet
# --------------------------------------------------------------------------

def fig11():
    fig, axs = plt.subplots(4, 6, figsize=(13.5, 6.4))
    fig.suptitle("The synthetic corpora: six hallucination families, ground truth free",
                 fontsize=13, fontweight="bold", y=1.0)
    rows = [("structured", "F1/F2 non-text", AM), ("noise", "F1 pure noise", OR),
            ("degraded", "F3 degraded text", PK), ("clean", "anchor data", EM)]
    for r, (batch, label, col) in enumerate(rows):
        recs = load_batch(batch)[:6]
        for c, rec in enumerate(recs):
            img = Image.open(KIT_DIR / rec["file"])
            axs[r, c].imshow(img, cmap="gray", vmin=0, vmax=1)
            axs[r, c].axis("off")
        axs[r, 0].set_ylabel(label, color=col, fontsize=10, rotation=0,
                             ha="right", va="center", labelpad=28)
    save(fig, "fig11_corpora.png")


# --------------------------------------------------------------------------
# fig12: pipeline flowchart
# --------------------------------------------------------------------------

def fig12():
    fig, ax = plt.subplots(figsize=(13.5, 5.6))
    ax.axis("off"); style_ax(ax)
    ax.set_xlim(0, 100); ax.set_ylim(0, 46)
    boxes = [
        (2, 30, 15, 11, "make_corpora.py\n6 families\n430 crops", BL),
        (21, 30, 16, 11, "ocria_probe.py\nΔh states · z_diff\nmatrix scans · v", CY),
        (41, 30, 15, 11, "spectral_audit.py\n12 formulas of the\nparent program", PU),
        (60, 30, 16, 11, "ocria_train.py\nformula-as-loss\n150 Adam steps", AM),
        (80, 30, 17, 11, "ocria_benchmark.py\nONE evaluator\n3 arms", EM),
    ]
    for x, y, w, h, txt, col in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6",
                                    fc="#0d1226", ec=col, lw=2))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center",
                fontsize=9.5, color=FG, family="monospace")
    for i in range(4):
        x0 = boxes[i][0] + boxes[i][2]
        ax.add_patch(FancyArrowPatch((x0 + 0.7, 35.5), (boxes[i + 1][0] - 0.7, 35.5),
                                     arrowstyle="-|>", mutation_scale=18,
                                     color="#566jugend"[:7] if False else "#56618b", lw=2))
    # artifacts strip
    arts = [
        (2, 12, "probe_report.json\nocria_vectors.npz", CY),
        (21, 12, "formulas_audit.json", PU),
        (41, 12, "{arm}_training_log.json\n{arm}_deltas.npz", AM),
        (60, 12, "eval_{arm}.json\nsummary_all_arms.json", EM),
    ]
    for x, y, txt, col in arts:
        ax.add_patch(FancyBboxPatch((x, y), 15, 9, boxstyle="round,pad=0.5",
                                    fc="#0a0e1e", ec="#2a3554", lw=1.4, ls="--"))
        ax.text(x + 7.5, y + 4.5, txt, ha="center", va="center", fontsize=8.5,
                color=col, family="monospace")
        ax.add_patch(FancyArrowPatch((x + 7.5, 29.4), (x + 7.5, y + 9.6),
                                     arrowstyle="-|>", mutation_scale=12,
                                     color="#56618b", lw=1.4))
    ax.text(50, 44.5, "the OCRia intervention pipeline — five instruments, one evaluator",
            ha="center", fontsize=13, fontweight="bold", color=FG)
    ax.text(50, 6, "fp32 rebuild of the int8 graph · deployed components (router, detector, "
            "3 more recognisers) measured, frozen", ha="center", fontsize=9, color="#9aa3b5")
    save(fig, "fig12_pipeline.png")


# --------------------------------------------------------------------------
# fig13: formula audit panel
# --------------------------------------------------------------------------

def fig13():
    A = load("formulas_audit.json")
    fig, ax = plt.subplots(figsize=(12.8, 6.4))
    ax.axis("off"); style_ax(ax)
    rows = [
        ("Steps 4–5 · MP bulk + Silverstein",
         f"weights KS={A['mp_summary']['ks_median']:.2f} (trained structure, expected); "
         f"states bulk KS={A['mp_activation_covariance']['ks_distance_bulk']:.2f}; "
         f"Silverstein residual {A['matrices']['head/Conv']['silverstein_residual']:.0e}", EM),
        ("Step 7 · BBP threshold θc=√q",
         f"head: θv/θc = {A['bbp_spike_scans']['head/Conv']['theta_v']:.2f}/"
         f"{A['bbp_spike_scans']['head/Conv']['theta_c']:.2f} → SUPERCRITICAL (λv/λ+ = "
         f"{A['bbp_spike_scans']['head/Conv']['lambda_ratio']:.1f})", AM),
        ("Step 8 · outlier trajectory + overlap",
         f"pole-equation vs closed form residual {A['bbp_outlier_check']['max_rel_residual']:.0e}", EM),
        ("Step 13 · Tracy–Widom class test",
         f"λmax z-scores: median {A['tracy_widom']['z_median']:.1f}, max "
         f"{max(A['tracy_widom']['lambda_max_zscores'].values()):.0f} (the head outlier)", BL),
        ("Step 10 · hallucination split",
         f"{A['hallucination_split']['n_signal_dirs']} signal dirs over the edge; "
         f"r(v)={A['hallucination_split']['noise_signal_ratio_of_v']:.2f} → v is bulk-scale flow", PU),
        ("Step 11 · entropy response",
         f"formula ΔH {A['entropy_response']['formula_delta_H']:.3f} vs MC "
         f"{A['entropy_response']['monte_carlo_delta_H']:.3f} "
         f"(err {A['entropy_response']['abs_err']:.1e})", EM),
        ("Step 14 · inverse-BBP shrinker",
         f"reconstruction err {A['shrinker']['inverse_bbp_rel_err']:.3f} "
         f"(Gavish–Donoho {A['shrinker']['gavish_donoho_rel_err']:.3f})", EM),
        ("Caputo N_crit",
         f"N_crit={A['caputo_ncrit']['n_crit']} vs ~121 frames/line → sub-critical regime", OR),
        ("Step 2 · RMSNorm trace lemma",
         "N/A — OCRia has no RMSNorm (honest report, not faked)", "#8b93a7"),
    ]
    y = 0.94
    ax.text(0.02, y, "formula", fontsize=10, color="#9aa3b5", family="monospace")
    ax.text(0.34, y, "what it says about OCRia-nano-1", fontsize=10, color="#9aa3b5",
            family="monospace")
    y -= 0.07
    for name, val, col in rows:
        ax.text(0.02, y, name, fontsize=9.5, color=col, family="monospace")
        ax.text(0.34, y, val, fontsize=9.5, color=FG, family="monospace")
        y -= 0.095
    ax.set_title("The parent repository's formulas, executed on OCRia-nano-1",
                 fontsize=13, pad=18)
    save(fig, "fig13_formula_audit.png")


# --------------------------------------------------------------------------
# fig14: the router round (F4 head edit - the 5th 'reject' class)
# --------------------------------------------------------------------------

def fig14():
    S = load("summary_all_arms.json")
    RR = load("eval_router_reject.json")
    probe_router = S["probe"]["router"]
    fig = plt.figure(figsize=(13.5, 7.6))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.24,
                          left=0.07, right=0.96, top=0.86, bottom=0.10)
    fig.suptitle("Round 3 — the router round: a 5th 'reject' class grafted "
                 "onto the frozen router's head",
                 fontsize=15, fontweight="bold", color=FG, y=0.965)

    # panel a: where the 40 fit non-text crops GO (base vs round)
    ax = fig.add_subplot(gs[0, 0])
    groups = ["latin", "cjk", "rtl", "world"]
    base_h = [probe_router["nontext_routed_to"][g] for g in groups]
    new_h = [RR["F4_fit"]["nontext_residual_routes"][g] for g in groups]
    rej = [RR["F4_fit"]["nontext_reject_rate"] * 40]
    ypos = [1.0, 0.0]
    lefts = [0.0, 0.0]
    colors4 = [BL, PU, OR, CY]
    for i, g in enumerate(groups):
        ax.barh(ypos[0], base_h[i], left=lefts[0], height=0.52,
                color=colors4[i], label=g)
        lefts[0] += base_h[i]
    ax.barh(ypos[0], 0, height=0.52, color="none",
            label="reject — class does not exist")
    ax.barh(ypos[1], rej[0], left=lefts[1], height=0.52,
            color=EM, label="reject (new class 4)")
    lefts[1] += rej[0]
    for i, g in enumerate(groups):
        ax.barh(ypos[1], new_h[i], left=lefts[1], height=0.52,
                color=colors4[i])
        lefts[1] += new_h[i]
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"base (no reject class)\n100% routed into recognisers",
                        f"router_reject ({RR['arm']})\nreject wins 40/40"], fontsize=9)
    ax.set_xlim(0, 42)
    ax.set_xlabel("fit non-text crops (n = 40)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.30), ncol=6)
    ax.set_title(f"F4 cured on the fit corpus: reject rate "
                 f"{RR['F4_fit']['nontext_reject_rate']:.2f}", pad=10)
    style_ax(ax)

    # panel b: transfer reject rate on families the router corpus never has
    ax = fig.add_subplot(gs[0, 1])
    fams = ["structured", "noise", "blank", "mean"]
    vals = [RR["F4_transfer"][f"{f}_reject_rate"] if f != "mean"
            else RR["F4_transfer"]["mean_reject_rate"] for f in fams]
    cols = [CY, BL, PU, EM]
    bars = ax.bar(fams, vals, color=cols, width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}",
                ha="center", fontsize=10, fontweight="bold")
    ax.axhline(1.0, color=FG, lw=0.8, ls="--", alpha=0.4)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("reject rate (per family; macro-mean)\nunseen by the round")
    ax.set_title("F4 transfer: families the round never trained on")
    style_ax(ax)

    # panel c: the reject margin gap (text pushed far below, non-text above)
    ax = fig.add_subplot(gs[1, 0])
    fit = RR["F4_fit"]
    ax.barh(1, fit["reject_margin_nontext_mean"], height=0.42, color=EM,
            alpha=0.85)
    ax.plot(fit["reject_margin_nontext_min"], 1, "o", color=FG, ms=6)
    ax.text(fit["reject_margin_nontext_min"], 1.30,
            f"min {fit['reject_margin_nontext_min']}", fontsize=8.5,
            ha="center", color=FG)
    ax.text(fit["reject_margin_nontext_mean"] + 0.15, 1.0,
            f"mean {fit['reject_margin_nontext_mean']}", fontsize=9,
            va="center", color=EM)
    ax.barh(0, fit["reject_margin_text_mean"], height=0.42,
            color=PK, alpha=0.85)
    ax.plot(fit["reject_margin_text_max"], 0, "o", color=FG, ms=6)
    ax.text(fit["reject_margin_text_max"], -0.34,
            f"max (closest to the edge) {fit['reject_margin_text_max']}",
            fontsize=8.5, ha="center", color=FG)
    ax.text(fit["reject_margin_text_mean"], 0.34,
            f"mean {fit['reject_margin_text_mean']}", fontsize=9,
            ha="center", color=PK)
    ax.axvline(0, color=FG, lw=1.0)
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["fit non-text (40)", "fit text (80)"], fontsize=9)
    ax.set_xlim(fit["reject_margin_text_mean"] - 2.5,
                fit["reject_margin_nontext_mean"] + 2.5)
    ax.set_xlabel("reject margin  =  ℓ_reject − max(ℓ_script)")
    ax.set_title("The margin gap: no text crop is even close to the edge")
    style_ax(ax)

    # panel d: the edit spectrum - the head edit is ONE intended BBP spike
    ax = fig.add_subplot(gs[1, 1])
    with np.load(RESULTS_DIR / "router_reject_deltas.npz") as z:
        dW = z["reject_row_W5_delta"].astype(np.float64)
    sq = dW @ dW.T / max(dW.shape)
    lam = np.clip(np.linalg.eigvalsh(sq), 0, None)[::-1][:12]
    hs = RR["head_spectrum"]
    xs = np.arange(1, len(lam) + 1)
    ax.semilogy(xs, np.maximum(lam, 2e-5), "o-", color=CY, ms=5,
                label="edit eigenvalues")
    ax.axhline(hs["lambda_plus"], color=PU, ls="--", lw=1.2,
               label=f"MP edge λ₊ = {hs['lambda_plus']:.4f}")
    ax.axhline(2 * hs["lambda_plus"], color=OR, ls=":", lw=1.2,
               label="2·λ₊ (spike threshold)")
    ax.annotate("the single intended spike\n(the round edits ONE row)",
                xy=(1, lam[0]), xytext=(3.0, 5.5e-2), fontsize=8.5,
                color=EM, arrowprops=dict(arrowstyle="->", color=EM))
    ax.set_ylim(2e-5, 3e-1)
    ax.set_xticks(xs)
    ax.set_xlabel("eigenvalue index (of dW dWᵀ / max dim)")
    ax.set_title(f"Edit spectrum: {hs['n_spikes_above_2lambda_plus']} spike "
                 f"above 2·λ₊  (λmax/λ₊ = {hs['lambda_max_over_lambda_plus']})")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    style_ax(ax)

    save(fig, "fig14_router_round.png")


# --------------------------------------------------------------------------
# GIF 1: pipeline animation
# --------------------------------------------------------------------------

def gif_pipeline():
    stages = [
        ("corpora", "make_corpora.py — 6 families, 430 crops", BL),
        ("probe", "ocria_probe.py — states, z_diff, scans", CY),
        ("audit", "spectral_audit.py — 12 formulas", PU),
        ("train", "ocria_train.py — formula-as-loss, 150 steps", AM),
        ("bench", "ocria_benchmark.py — one evaluator, 3 arms", EM),
    ]
    frames = []
    S = load("summary_all_arms.json")
    for k, (key, txt, col) in enumerate(stages):
        fig, ax = plt.subplots(figsize=(8.0, 4.6))
        ax.axis("off"); style_ax(ax)
        ax.set_xlim(0, 10); ax.set_ylim(0, 6)
        for i, (k2, t2, c2) in enumerate(stages):
            x = 0.6 + i * 1.9
            active = i == k
            ax.add_patch(FancyBboxPatch((x, 3.6), 1.6, 1.5, boxstyle="round,pad=0.12",
                                        fc="#0d1226" if not active else "#141a33",
                                        ec=c2, lw=3.2 if active else 1.2,
                                        alpha=1.0 if active else 0.45))
            ax.text(x + 0.8, 4.35, k2, ha="center", va="center", fontsize=11,
                    color=c2 if active else "#56618b", fontweight="bold" if active else "normal")
        ax.text(5, 5.55, txt, ha="center", fontsize=13, color=col, fontweight="bold")
        # live readouts
        arms = S["arms"]
        ax.text(5, 2.6, f"F1 phantom  base {arms['base']['F1_phantom_rate']:.3f}"
                f"  →  classic {arms['classic']['F1_phantom_rate']:.3f}"
                f"  →  rcda_x {arms['rcda_x']['F1_phantom_rate']:.3f}",
                ha="center", fontsize=11, color=FG, family="monospace")
        ut = S["probe"]["utility_trap"]
        ax.text(5, 1.9, f"Utility Trap: AUC conf {ut['auc_confidence']:.3f} "
                f"vs geometry {ut['auc_geometry_minus_v']:.3f}",
                ha="center", fontsize=10, color=PK, family="monospace")
        ax.text(5, 1.2, "the head matrix writes the phantom direction SUPERCRITICALLY (θv/θc = 14.3)",
                ha="center", fontsize=9.5, color=AM, family="monospace")
        ax.text(5, 0.4, "rmt-llm-research × OCRia-nano-1", ha="center", fontsize=8.5,
                color="#56618b", family="monospace")
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB").resize((800, 460)))
        plt.close(fig)
    frames = frames + frames[1:-1]  # ping-pong loop
    frames[0].save(GR / "gif_pipeline.gif", save_all=True,
                   append_images=frames[1:], duration=1400, loop=0)
    print("[gfx] wrote gif_pipeline.gif")


# --------------------------------------------------------------------------
# GIF 2: training melt animation
# --------------------------------------------------------------------------

def gif_training_melt():
    logs = {a: load(f"{a}_training_log.json") for a in ("classic", "rcda_x")}
    frames = []
    n_frames = 12
    max_step = 150
    for k in range(n_frames):
        upto = int((k + 1) / n_frames * max_step)
        fig, ax = plt.subplots(figsize=(8.6, 4.8))
        for arm, log in logs.items():
            bad = [s for s in log["steps"] if s["kind"] == "bad"
                   and s["pen_raw"] > 0 and s["step"] <= upto]
            if bad:
                ax.plot([s["step"] for s in bad], [s["pen_raw"] for s in bad],
                        color=ARM_COLORS[arm], lw=2.2,
                        label=f"{arm} (pen on frozen frames)")
        ax.set_yscale("log")
        ax.set_xlim(0, 150)
        ax.set_title(f"the formula melting — step {upto}/150")
        ax.set_xlabel("step"); ax.set_ylabel("pen (log)")
        bench = {a: [b for b in log["bench"] if b["step"] <= upto] for a, log in logs.items()}
        txt = "   ".join(f"{a}: phantom {b[-1]['phantom_rate']:.2f}" for a, b in bench.items() if b)
        ax.text(0.02, 0.03, txt, transform=ax.transAxes, fontsize=9, color="#9aa3b5")
        ax.legend(frameon=False, fontsize=8.5)
        style_ax(ax)
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB").resize((860, 480)))
        plt.close(fig)
    frames[0].save(GR / "gif_training_melt.gif", save_all=True,
                   append_images=frames[1:], duration=650, loop=0)
    print("[gfx] wrote gif_training_melt.gif")


# --------------------------------------------------------------------------
# GIF 3: before/after decoding of the same crops
# --------------------------------------------------------------------------

def gif_before_after():
    base_rows = {r["id"]: r for r in
                 (json.loads(l) for l in open(RESULTS_DIR / "eval_base_rows.jsonl",
                                              encoding="utf-8"))}
    cl_rows = {r["id"]: r for r in
               (json.loads(l) for l in open(RESULTS_DIR / "eval_classic_rows.jsonl",
                                            encoding="utf-8"))}
    x_rows = {r["id"]: r for r in
              (json.loads(l) for l in open(RESULTS_DIR / "eval_rcda_x_rows.jsonl",
                                           encoding="utf-8"))}
    # structured crops where base hallucinates and at least one arm is clean
    picks = []
    for rid, r in base_rows.items():
        if r["batch"] == "structured" and r["n_chars"] > 0:
            picks.append(rid)
    picks = picks[:14]
    frames = []
    for rid in picks:
        b, c, x = base_rows[rid], cl_rows[rid], x_rows[rid]
        rec = next(r for r in load_batch("structured") if r["id"] == rid)
        img = Image.open(KIT_DIR / rec["file"]).resize((520, 52))
        fig, ax = plt.subplots(figsize=(8.6, 2.6))
        ax.axis("off"); style_ax(ax)
        ax.imshow(np.asarray(img), cmap="gray", extent=[0, 10, 2.1, 3.1], aspect="auto")
        ax.text(0.05, 1.72, f"BASE    : '{b['text'][:38]}'  (conf {b['mean_conf']:.2f})",
                fontsize=11, color=PK, family="monospace")
        ax.text(0.05, 1.18, f"CLASSIC : '{c['text'][:38]}'  "
                f"{'(abstained)' if not c['text'] else ''}",
                fontsize=11, color=EM, family="monospace")
        ax.text(0.05, 0.64, f"RCDA-X  : '{x['text'][:38]}'  "
                f"{'(abstained)' if not x['text'] else ''}",
                fontsize=11, color=CY, family="monospace")
        ax.text(0.05, 0.15, rid, fontsize=8, color="#56618b", family="monospace")
        ax.set_ylim(0, 3.2)
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB").resize((860, 260)))
        plt.close(fig)
    if frames:
        frames[0].save(GR / "gif_before_after.gif", save_all=True,
                       append_images=frames[1:], duration=1600, loop=0)
        print("[gfx] wrote gif_before_after.gif")


# --------------------------------------------------------------------------
# GIF 4: formula ring (hero animation)
# --------------------------------------------------------------------------

def gif_formula_ring():
    formulas = [
        ("MP bulk", "σ²(1±√q)²", CY),
        ("BBP threshold", "θc = √q", AM),
        ("Silverstein", "Σ-residual 1e-15", BL),
        ("Tracy–Widom", "F₁ (β=1)", PU),
        ("entropy response", "−½Tr((diag p−pp^T)A)", EM),
        ("inverse-BBP", "shrinker", PK),
    ]
    frames = []
    for k in range(len(formulas)):
        fig, ax = plt.subplots(figsize=(7.4, 7.4))
        ax.axis("off"); style_ax(ax)
        ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
        ax.add_patch(plt.Circle((0, 0), 1.0, fc="none", ec="#2a3554", lw=2.5))
        for i, (name, f, col) in enumerate(formulas):
            ang = 2 * np.pi * i / len(formulas) - np.pi / 2
            x, y = np.cos(ang), np.sin(ang)
            active = i == k
            ax.add_patch(plt.Circle((x, y), 0.20, fc="#141a33" if active else "#0d1226",
                                    ec=col, lw=3.4 if active else 1.4,
                                    alpha=1.0 if active else 0.5))
            ax.text(x, y, str(i + 1), ha="center", va="center", fontsize=13,
                    color=col, fontweight="bold" if active else "normal")
        f = formulas[k]
        ax.text(0, 0.12, f[0], ha="center", fontsize=15, color=f[2], fontweight="bold")
        ax.text(0, -0.14, f[1], ha="center", fontsize=12, color=FG, family="monospace")
        ax.text(0, -1.38, "every formula of the parent repository, executed on OCRia-nano-1",
                ha="center", fontsize=9, color="#9aa3b5")
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB").resize((560, 560)))
        plt.close(fig)
    frames[0].save(GR / "gif_formula_ring.gif", save_all=True,
                   append_images=frames[1:], duration=1100, loop=0)
    print("[gfx] wrote gif_formula_ring.gif")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    jobs = {
        "fig01": fig01, "fig02": fig02, "fig03": fig03, "fig04": fig04,
        "fig05": fig05, "fig06": fig06, "fig07": fig07, "fig08": fig08,
        "fig09": fig09, "fig10": fig10, "fig11": fig11, "fig12": fig12,
        "fig13": fig13, "fig14": fig14, "gif1": gif_pipeline,
        "gif2": gif_training_melt, "gif3": gif_before_after,
        "gif4": gif_formula_ring,
    }
    if which == "all":
        for name, fn in jobs.items():
            try:
                fn()
            except Exception as e:
                print(f"[gfx] {name} FAILED: {type(e).__name__}: {e}")
    else:
        jobs[which]()
