#!/usr/bin/env python3
"""Aurora re-render of the RCDA figure set (fig5–fig15) + Graphic additions.

Every figure is rebuilt from the *committed* result JSONs of the RCDA rounds —
no model, no torch, no re-running of experiments. The data mappings follow the
original tools (`Research/RCDA/tools/rcda*_figures.py`, `Code/pocket_rmt_probe.py`)
one-to-one; only the style (Black-Aurora palette of ``assets/prompts.yaml``) and
the language (English) change. Per snapshot, a figure is produced only when its
inputs are present, mirroring the original guarded behaviour.

Outputs:
  neural_lab/MNN-Pocket-0.3B/R CDA - {2,3,4,5}/Research/RCDA/figures/fig*.png
  neural_lab/MNN-Pocket-0.3B/Graphic/aurora_layer_profile.png
  neural_lab/MNN-Pocket-0.3B/Graphic/aurora_benchmark_summary.png
  neural_lab/MNN-Pocket-0.3B/assets/charts/chart-*.png (README summaries)

Usage::

    python scripts/gen_mnn_figures.py            # everything
    python scripts/gen_mnn_figures.py --snap 5   # one snapshot only
    python scripts/gen_mnn_figures.py --charts   # only the branded summaries
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib


matplotlib.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


fm.fontManager.addfont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
# token-attribution labels carry CJK glyphs from the model's own generations
fm.fontManager.addfont("/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MNN = os.path.join(REPO, "neural_lab", "MNN-Pocket-0.3B")

# ─── Black Aurora palette (assets/prompts.yaml) ─────────────────────────────
BG = "#05060A"
PANEL = "#0B1220"
INK = "#E8EEF7"
MUTED = "#8CA3C7"
GRID = "#1A2130"
BULK = "#00D4FF"  # cyan  — Marchenko–Pastur noise
SPIKE = "#FFB020"  # amber — BBP signal / the loop direction
EDGE = "#B026FF"  # violet — Tracy–Widom edge / orbitals
COLLAPSE = "#FF3B6B"  # crimson — the defect / collapse
GOOD = "#38E1C0"  # teal — healthy / control

plt.rcParams.update(
    {
        "font.sans-serif": ["DejaVu Sans", "Noto Sans SC"],
        "axes.unicode_minus": False,
        "figure.facecolor": BG,
        "axes.facecolor": PANEL,
        "savefig.facecolor": BG,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.labelcolor": INK,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "text.color": INK,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "legend.labelcolor": INK,
        "grid.color": GRID,
        "grid.alpha": 0.6,
        "axes.grid": True,
    }
)

FIGS = (
    "fig5_rcds_layers",
    "fig6_matrix_scans",
    "fig7_behavioral_probe",
    "fig8_dose_response",
    "fig9_precursor_pr",
    "fig10_token_attribution",
    "fig11_sentinel",
    "fig12_heads",
    "fig13_therapy",
    "fig14_rcda4",
    "fig15_rcda5",
)


def _load(path: str):
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else None


def _jl(path: str):
    """Read a JSONL file, skipping malformed lines."""
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def _save(fig, path: str, dpi: int = 150) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor=BG)
    plt.close(fig)
    im = Image.open(path).convert("L")
    hist = im.histogram()
    # dark theme: "blank" means ~everything is background; count lit pixels
    lit = sum(hist[:250]) / sum(hist)
    print(f"  {os.path.relpath(path, REPO)}  {im.size}  lit={lit:.3f}")
    assert lit > 0.01, f"figure looks blank: {path}"


def _src(ax, text: str) -> None:
    ax.annotate(
        text,
        (0.995, 0.005),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=6.5,
        color=MUTED,
    )


# ─────────────────────────────────────────────── fig5: RCDS probe per layer ──
def fig5(rcda_dir: str, out: str) -> bool:
    rep = _load(os.path.join(rcda_dir, "results", "rcda_probe_report.json"))
    if not rep:
        return False
    layers = rep["layers"]
    li = [d["layer"] for d in layers]
    candidates = rep.get("candidate_layers", [])
    fig, axes = plt.subplots(3, 1, figsize=(9, 8.4), constrained_layout=True, sharex=True)
    axes[0].semilogy(
        li,
        [d["lambda_max"] for d in layers],
        "o-",
        ms=3,
        lw=1.2,
        color=BULK,
        label=r"$\lambda_{max}$ (pooled cov)",
    )
    axes[0].semilogy(
        li,
        [d["lambda_plus_mp"] for d in layers],
        "--",
        lw=1.2,
        color=SPIKE,
        label=r"$\lambda_+$ (MP)",
    )
    axes[0].set_ylabel(r"$\lambda$")
    axes[0].set_title("RCDS probe: layer spectra vs the MP bulk edge")
    axes[1].semilogy(
        li,
        [max(d["z_rmt"], 1e-3) for d in layers],
        "o-",
        ms=3,
        lw=1.2,
        color=BULK,
        label=r"$z_{RMT}$",
    )
    axes[1].semilogy(
        li,
        [max(d["z_diff"], 1e-3) for d in layers],
        "s-",
        ms=3,
        lw=1.2,
        color=EDGE,
        label=r"$z_{diff}$ (Hotelling on $\Delta h$)",
    )
    axes[1].axhline(2.0, color=MUTED, ls=":", lw=1.0)
    axes[1].text(li[0], 2.3, " z = 2", fontsize=8, color=MUTED)
    axes[1].set_ylabel("z-score")
    for cd in candidates:
        for ax in axes:
            ax.axvline(cd, color=SPIKE, ls="--", lw=0.8, alpha=0.7)
    axes[2].plot(
        li,
        [abs(d["beta"]) for d in layers],
        "o-",
        ms=3,
        lw=1.2,
        color=EDGE,
        label=r"$|\beta| = |\langle\Delta h, u_1\rangle|$",
    )
    axes[2].plot(
        li,
        [d["beta_noise_floor"] for d in layers],
        "--",
        lw=1.2,
        color=MUTED,
        label=r"$\sqrt{\lambda_1/N}$ (noise floor)",
    )
    axes[2].set_yscale("log")
    axes[2].set_ylabel(r"$|\beta|$")
    axes[2].set_xlabel("block index (0-27)")
    for ax in axes:
        ax.legend(loc="upper left")
    fig.suptitle(
        f"RCDS probe: {rep['meta']['n_prompts']['bad']} bad vs "
        f"{rep['meta']['n_prompts']['good']} good prompts, "
        f"candidates {candidates}",
        color=INK,
    )
    _src(ax, "data: results/rcda_probe_report.json (round 1)")
    _save(fig, out)
    return True


# ─────────────────────────────────────────────── fig6: matrix scans ─────────
def fig6(rcda_dir: str, out: str) -> bool:
    rep = _load(os.path.join(rcda_dir, "results", "rcda_probe_report.json"))
    if not rep or not rep.get("matrix_scans"):
        return False
    scans = rep["matrix_scans"]
    fig, ax = plt.subplots(figsize=(9, 4.6), constrained_layout=True)
    xs = np.arange(len(scans))
    vals = [s["scan"]["lambda_over_edge"] for s in scans]
    cols = [COLLAPSE if s["scan"]["supercritical"] else BULK for s in scans]
    ax.bar(xs, vals, color=cols)
    ax.axhline(1.0, color=INK, ls="--", lw=1.2)
    ax.text(xs[0], 1.05, r" $\lambda_v = \lambda_+$", fontsize=9, color=INK)
    ax.set_xticks(xs, [f"L{s['layer']}\n{s['matrix']}" for s in scans], fontsize=8)
    ax.set_ylabel(r"$\lambda_v / \lambda_+^{(W)}$")
    ax.set_title(
        "Matrix scans: projection mass of $\\Delta h$ vs the MP edge (crimson = supercritical)"
    )
    _src(ax, "data: results/rcda_probe_report.json (round 1)")
    _save(fig, out)
    return True


# ─────────────────────────────────────── fig7: behavioral probe (round 2) ───
def fig7(rcda_dir: str, out: str) -> bool:
    probe = _load(os.path.join(rcda_dir, "results", "rcda2_probe.json"))
    if not probe:
        return False
    layers = probe["layers"]
    li = [d["layer"] for d in layers]
    top = probe.get("top_layers", [])
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.8), constrained_layout=True)
    axes[0].plot(
        li,
        [d["z_diff"] for d in layers],
        "o-",
        ms=3,
        lw=1.2,
        color=EDGE,
        label=r"$z_{diff}$ (Hotelling, behavioral $\Delta h$)",
    )
    for cd in top:
        axes[0].axvline(cd, color=SPIKE, ls="--", lw=0.8, alpha=0.7)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("z-score")
    axes[0].set_title("RCDA-2: behavioral contrast (loop vs clean tokens, same prompts)")
    cos_l = [d for d in layers if d.get("cosine_vs_style")]
    if cos_l:
        names = sorted({k for d in cos_l for k in d["cosine_vs_style"]})
        xs = np.arange(len(cos_l))
        for k in names:
            axes[1].plot(
                xs,
                [abs(d["cosine_vs_style"].get(k, np.nan)) for d in cos_l],
                "s-",
                ms=4,
                lw=1.2,
                color=BULK,
                label=f"|cos| vs style {k}",
            )
        axes[1].set_xticks(xs, [f"L{d['layer']}" for d in cos_l])
        axes[1].axhline(0.3, color=MUTED, ls=":", lw=1.0)
        axes[1].text(0, 0.31, " 0.3", fontsize=8, color=MUTED)
        axes[1].set_ylabel("|cosine|")
        axes[1].set_title(
            "Overlap of the round-1 style direction with the round-2 behavioral direction"
        )
        axes[1].legend()
    _src(axes[0], "data: results/rcda2_probe.json")
    _save(fig, out)
    return True


# ─────────────────────────────────────────────── fig8: dose-response ────────
def fig8(rcda_dir: str, out: str) -> bool:
    dose = _load(os.path.join(rcda_dir, "results", "rcda2_dose.json"))
    if not dose:
        er = _load(os.path.join(rcda_dir, "results", "rcda2_edit_report.json"))
        if er and er.get("meta", {}).get("dose_response"):
            m = er["meta"]
            dose = {
                "alpha_star": m.get("alpha_star"),
                "note": m.get("note"),
                "base_cal": m.get("base_cal"),
                "dose_response": m.get("dose_response"),
            }
    if not dose:
        return False
    dd = dose["dose_response"]
    al = [d["alpha"] for d in dd]
    fig, ax1 = plt.subplots(figsize=(9, 4.6), constrained_layout=True)
    ax1.plot(al, [d["kl"] for d in dd], "o-", color=EDGE, label="mean KL(base||edited)")
    ax1.set_xlabel(r"ablation strength $\alpha$")
    ax1.set_ylabel("KL", color=EDGE)
    ax1.tick_params(axis="y", labelcolor=EDGE)
    ax2 = ax1.twinx()
    ax2.spines["right"].set_visible(True)
    w = 0.03
    ax2.bar(
        [a - w for a in al],
        [d["changed"]["bad"] for d in dd],
        width=2 * w,
        color=COLLAPSE,
        alpha=0.75,
        label="changed bad trajectories",
    )
    ax2.bar(
        [a + w for a in al],
        [d["changed"]["good"] for d in dd],
        width=2 * w,
        color=SPIKE,
        alpha=0.75,
        label="changed good trajectories",
    )
    ax2.plot(al, [d["rep4_bad"] for d in dd], "s--", color=BULK, label="mean RepRate-4 (bad)")
    ax2.axhline(dose["base_cal"]["rep4_bad"], color=BULK, ls=":", lw=1)
    ax2.set_ylabel("trajectories changed / rep4", color=COLLAPSE)
    ax2.tick_params(axis="y", labelcolor=COLLAPSE)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")
    ax1.set_title("Dose-response of the behavioral RCDA edit (calibration subset)")
    _src(ax1, "data: results/rcda2_dose.json")
    _save(fig, out)
    return True


# ─────────────────────────────────────────────── fig9: PR precursor ─────────
def fig9(rcda_dir: str, out: str) -> bool:
    probe = _load(os.path.join(rcda_dir, "results", "rcda2_probe.json"))
    if not probe or "precursor" not in probe:
        return False
    prec = probe["precursor"]
    sel_layers = sorted(prec["loop"].keys(), key=int)
    fig, axes = plt.subplots(
        1,
        len(sel_layers),
        figsize=(4.2 * len(sel_layers), 4.2),
        constrained_layout=True,
        sharey=True,
    )
    if len(sel_layers) == 1:
        axes = [axes]
    for ax, lk in zip(axes, sel_layers):
        for key, color, lab in (
            ("pre", BULK, "pre-collapse"),
            ("loop", COLLAPSE, "looping regime"),
        ):
            data = prec[key].get(lk, [])
            if not data:
                continue
            lags = np.array([r[0] for r in data], float)
            prs = np.array([r[1] for r in data], float)
            order = np.argsort(lags)
            uq = np.unique(lags[order])
            m = [prs[lags == u].mean() for u in uq]
            ax.plot(uq, m, "o-", ms=3, lw=1.2, color=color, label=lab)
        cl = prec["clean"].get(lk, [])
        if cl:
            prs = np.array([r[1] for r in cl], float)
            ax.axhspan(prs.mean() - prs.std(), prs.mean() + prs.std(), color=GOOD, alpha=0.15)
            ax.axhline(prs.mean(), color=GOOD, ls="--", lw=1.2, label="clean generations")
        ax.axvline(0, color=MUTED, ls=":", lw=1)
        ax.set_title(f"layer {lk}")
        ax.set_xlabel("lag vs collapse onset (tokens)")
        ax.legend()
    axes[0].set_ylabel("participation ratio (window 16)")
    fig.suptitle("Spectral precursor: windowed PR around the collapse onset", color=INK)
    _src(axes[-1], "data: results/rcda2_probe.json → precursor")
    _save(fig, out)
    return True


# ─────────────────────────────────────────────── fig10: token attribution ───
def fig10(rcda_dir: str, out: str) -> bool:
    probe = _load(os.path.join(rcda_dir, "results", "rcda2_probe.json"))
    if not probe or "token_attribution" not in probe:
        return False
    attr = probe["token_attribution"]
    lk = sorted(attr.keys())[0]
    fig, ax = plt.subplots(figsize=(9, 5.4), constrained_layout=True)
    b = attr[lk]["boosted"][:10]
    s = attr[lk]["suppressed"][:10]
    toks = [t for t, _ in b] + [t for t, _ in s]
    vals = [v / max(abs(v) for _, v in b + s) for _, v in b + s]
    colors = [COLLAPSE] * len(b) + [BULK] * len(s)
    ypos = np.arange(len(toks))[::-1]
    ax.barh(ypos, vals, color=colors)
    ax.set_yticks(ypos, [repr(t)[:22] for t in toks], fontsize=9)
    # tokens come from the model's multilingual generations: Noto Sans SC
    # covers both the Latin and the CJK glyphs, DejaVu only the Latin ones
    for lbl in ax.get_yticklabels():
        lbl.set_fontfamily("Noto Sans SC")
    ax.tick_params(axis="y", labelcolor=INK)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel(r"relative $E\,\Delta h$ score (tied embeddings)")
    ax.set_title(f"Token attribution of the behavioral loop direction, {lk}")
    ax.grid(axis="x")
    _src(ax, "data: results/rcda2_probe.json → token_attribution")
    _save(fig, out)
    return True


def _auc_curve(pos, neg):
    allv = np.unique(np.concatenate([pos, neg]))
    thrs = np.concatenate([[allv[0] - 1e-9], allv])
    tpr, fpr = [], []
    for thr in thrs:
        tpr.append(float(np.mean(pos >= thr)))
        fpr.append(float(np.mean(neg >= thr)))
    order = np.argsort(fpr)
    return np.array(fpr)[order], np.array(tpr)[order]


def _auc_mw(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    sa = allv[order]
    i = 0
    while i < len(sa):
        j = i
        while j + 1 < len(sa) and sa[j + 1] == sa[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    rp = ranks[: len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _pooled_scores(sent, det, w, horizon=4):
    calib = sent["calib"]
    pos, neg = [], []
    for r in det:
        c = r["collapse"]
        for t in range(w - 1, len(r["ids"])):
            zs = []
            for l in (15, 16, 17):
                v = r["pr_traces"][f"W{w}_L{l}"][t]
                if v is not None:
                    cb = calib[f"W{w}_L{l}"]
                    zs.append((cb["mu"] - v) / max(cb["sigma"], 1e-9))
            if not zs:
                continue
            s = max(zs)
            if c is not None and t < c <= t + horizon:
                pos.append(s)
            elif c is None or t + horizon < c:
                neg.append(s)
    return np.array(pos), np.array(neg)


# ─────────────────────────────────────────────── fig11: sentinel ────────────
def fig11(rcda_dir: str, out: str) -> bool:
    sent = _load(os.path.join(rcda_dir, "results", "rcda3_sentinel.json"))
    gens = _jl(os.path.join(rcda_dir, "results", "rcda3_sentinel_generations.jsonl"))
    bench = _load(os.path.join(rcda_dir, "results", "rcda_benchmark.json"))
    if not sent or not gens or not bench:
        return False
    det = [g for g in gens if g.get("arm") == "detect" and "pr_traces" in g]
    if not det:
        return False
    fig, axs = plt.subplots(1, 4, figsize=(19, 4.2), constrained_layout=True)

    ax = axs[0]
    looping = [r for r in det if r["collapse"] is not None]
    early = next(r for r in looping if r["collapse"] <= 8)
    late = next(r for r in looping if r["collapse"] >= 16)
    clean = next((r for r in det if r["collapse"] is None and len(r["ids"]) >= 14), None)
    theta8 = sent["calib"]["W8_L15"]["theta"]
    for r, col, lab in [
        (early, COLLAPSE, f"early collapse (c={early['collapse']})"),
        (late, BULK, f"late collapse (c={late['collapse']})"),
    ]:
        tr = r["pr_traces"]["W8_L15"]
        ax.plot(range(len(tr)), tr, color=col, lw=1.4, label=lab)
        ax.axvline(r["collapse"], color=col, ls=":", lw=1)
        if r["alarm_steps"]:
            a = r["alarm_steps"][0]
            ax.scatter(
                [a], [tr[a] if tr[a] is not None else theta8], marker="v", color=col, zorder=5
            )
    if clean:
        tr = clean["pr_traces"]["W8_L15"]
        ax.plot(
            range(len(tr)),
            tr,
            color=MUTED,
            lw=1.2,
            label=f"clean gen ({clean['id'].split('_')[-1]})",
        )
    ax.axhline(theta8, color=INK, ls="--", lw=1, label=r"$\theta$ (q10 healthy)")
    ax.set_xlabel("generated token")
    ax.set_ylabel("PR-8, layer 15")
    ax.set_title("(a) online PR traces")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)

    ax = axs[1]
    deltas = []
    for r in looping:
        if r["alarm_steps"]:
            deltas.append(r["alarm_steps"][0] - r["collapse"])
    bins = np.arange(min(deltas) - 0.5, max(deltas) + 1.5, 1.0)
    ax.hist(deltas, bins=bins, color=BULK, alpha=0.85)
    ax.axvline(0, color=INK, ls="--", lw=1)
    ax.text(
        0.02,
        0.95,
        f"lead < 0: {sum(1 for d in deltas if d < 0)}/{len(deltas)}\n"
        f"median lead {np.median([d for d in deltas if d < 0]):.0f} tok",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        color=INK,
    )
    ax.set_xlabel("first alarm − collapse onset (tokens)")
    ax.set_ylabel("# generations")
    ax.set_title(f"(b) alarm timing, TPR={sent['detection']['tpr_any_alarm']:.2f}")
    ax.grid(alpha=0.3)

    ax = axs[2]
    for w, hz, col, lw in [
        (8, 4, BULK, 1.8),
        (16, 12, COLLAPSE, 1.8),
        (8, 12, BULK, 0.9),
        (16, 4, COLLAPSE, 0.9),
    ]:
        pos, neg = _pooled_scores(sent, det, w, hz)
        fpr, tpr = _auc_curve(pos, neg)
        a = _auc_mw(pos, neg)
        ax.plot(fpr, tpr, color=col, lw=lw, label=f"W={w}, h={hz}: AUC={a:.3f}")
    ax.plot([0, 1], [0, 1], "--", color=MUTED, lw=0.8)
    ax.set_xlabel("FPR (steps far from collapse)")
    ax.set_ylabel("TPR (collapse within horizon)")
    ax.set_title("(c) loop prediction: window-matched horizons")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axs[3]
    arms = sent["rescue_arms"]
    names = list(arms)
    rates = [arms[a]["rescue_rate"] for a in names]
    waste = [arms[a]["loop_waste_mean"] for a in names]
    base_waste = sent["detection"]["loop_token_waste_mean"]
    x = np.arange(len(names))
    ax.bar(x - 0.2, rates, 0.4, color=GOOD, label="rescue rate")
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.bar(x + 0.2, waste, 0.4, color=COLLAPSE, label="loop tokens (mean)")
    ax2.axhline(base_waste, color=COLLAPSE, ls="--", lw=1, label=f"no rescue: {base_waste:.0f}")
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("rescue rate")
    ax2.set_ylabel("loop tokens per gen", color=COLLAPSE)
    ax2.tick_params(axis="y", labelcolor=COLLAPSE)
    ax.set_title(f"(d) rescue arms (winner: {sent['winner']})")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left")
    ax.grid(alpha=0.3)
    fig.suptitle(
        "RCDA-3 spectral sentinel: online detection and rescue of repetition loops", color=INK
    )
    _src(axs[0], "data: results/rcda3_sentinel{.json,_generations.jsonl}")
    _save(fig, out, dpi=160)
    return True


# ─────────────────────────────────────────────── fig12: head surgery ────────
def fig12(rcda_dir: str, out: str) -> bool:
    heads = _load(os.path.join(rcda_dir, "results", "rcda3_heads.json"))
    if not heads:
        return False
    prof = heads["head_profiles"]
    conds = heads["conditions"]
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.4), constrained_layout=True)

    ax = axs[0]
    w = 0.26
    for i, (l, col) in enumerate([("15", BULK), ("16", SPIKE), ("18", EDGE)]):
        conc = np.array(prof[l]["concentration"])
        xs = np.arange(14) + (i - 1) * w
        ax.bar(xs, conc, w, color=col, label=f"L{l}", alpha=0.9)
        ax.scatter(
            [prof[l]["top_head"] + (i - 1) * w],
            [conc[prof[l]["top_head"]]],
            marker="v",
            color=INK,
            zorder=5,
        )
    ax.axhline(1.0, color=MUTED, ls="--", lw=1, label="uniform")
    ax.set_xticks(range(14), [f"h{h}" for h in range(14)], fontsize=7)
    ax.set_ylabel("concentration (× uniform)")
    ax.set_title("(a) who writes the loop direction (o_proj)")
    ax.legend()
    ax.grid(axis="y")

    ax = axs[1]
    names = list(conds)
    hit_bad = [conds[n]["hit_bad"] for n in names]
    changed = [conds[n]["changed_bad"] for n in names]
    cols = [COLLAPSE if n == "placebo3" else BULK for n in names]
    x = np.arange(len(names))
    ax.bar(x, hit_bad, 0.55, color=cols)
    ax.axhline(
        heads["baseline"]["hit_bad"],
        color=INK,
        ls="--",
        lw=1,
        label=f"baseline {heads['baseline']['hit_bad']:.2f}",
    )
    for i, (hb, ch) in enumerate(zip(hit_bad, changed)):
        ax.text(i, hb + 0.02, f"{hb:.2f}\n({ch} chg)", ha="center", fontsize=7, color=INK)
    ax.set_xticks(x, names, fontsize=8)
    ax.set_ylabel("defect rate, 8 bad cal prompts")
    ax.set_title("(b) head ablation, calibration dose")
    ax.legend()
    ax.grid(axis="y")

    ax = axs[2]
    kl = [conds[n]["kl"] for n in names]
    rel = [conds[n]["rel_frobenius"] for n in names]
    tw = [",".join(conds[n]["tw_verdicts"].values()) for n in names]
    x = np.arange(len(names))
    ax.bar(x - 0.2, kl, 0.4, color=EDGE, label="KL(base||edited)")
    ax.set_yscale("log")
    ax.set_xticks(x, [f"{n}\ntw={t}\nrel={r:.1e}" for n, t, r in zip(names, tw, rel)], fontsize=6.5)
    ax.set_ylabel("KL on control texts")
    ax.set_title("(c) cost of head ablation")
    ax.legend()
    ax.grid(axis="y")
    fig.suptitle("RCDA-3 head surgery: is the concentration causal?", color=INK)
    _src(axs[0], "data: results/rcda3_heads.json")
    _save(fig, out, dpi=160)
    return True


# ─────────────────────────────────────────────── fig13: therapy arms ────────
def fig13(rcda_dir: str, out: str) -> bool:
    sent = _load(os.path.join(rcda_dir, "results", "rcda3_sentinel.json"))
    bench = _load(os.path.join(rcda_dir, "results", "rcda_benchmark.json"))
    heads = _load(os.path.join(rcda_dir, "results", "rcda3_heads.json"))
    if not sent or not bench:
        return False
    bm = sent["benchmark"]
    arms_rho = {
        "A\nbaseline": bm["A_baseline"]["rho"],
        "B\npermanent edit": bench["summary"]["rho_edited"],
        f"C\nsentinel({sent['winner']})": bm["C_sentinel"]["rho"],
    }
    if heads and heads.get("decision_go") and heads.get("arm_d"):
        arms_rho["D\nhead combo"] = heads["arm_d"]["rho"]

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.4), constrained_layout=True)
    ax = axs[0]
    x = np.arange(len(arms_rho))
    cols = [MUTED, EDGE, GOOD, SPIKE][: len(arms_rho)]
    ax.bar(x, list(arms_rho.values()), 0.55, color=cols)
    ax.axhline(
        bench["summary"]["rho_baseline"],
        color=INK,
        ls="--",
        lw=1,
        label=f"baseline rho = {bench['summary']['rho_baseline']:.2f}",
    )
    for i, v in enumerate(arms_rho.values()):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(x, list(arms_rho), fontsize=8)
    ax.set_ylabel("defect rate rho (20 bad prompts)")
    ax.set_title("(a) therapy arms vs baseline")
    ax.legend()
    ax.grid(axis="y")

    ax = axs[1]
    waste = {
        "A": float(np.mean([r["loop_tokens"] for r in bm["A_baseline"]["rows"]])),
        "C": float(np.mean([r["loop_tokens"] for r in bm["C_sentinel"]["rows"]])),
    }
    if heads and heads.get("arm_d"):
        waste["D"] = float(
            np.mean(
                [
                    (r["n_tokens"] - r["collapse"]) if r["collapse"] is not None else 0
                    for r in heads["arm_d"]["rows"]
                ]
            )
        )
    waste["T_edited"] = bm["T_edited_transfer"]["waste"]
    names = list(waste)
    x = np.arange(len(names))
    ax.bar(x, [waste[n] for n in names], 0.55, color=[MUTED, GOOD, SPIKE, EDGE][: len(names)])
    for i, n in enumerate(names):
        ax.text(i, waste[n] + 0.5, f"{waste[n]:.0f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(x, names)
    ax.set_ylabel("loop tokens per looping gen")
    ax.set_title("(b) time wasted inside the loop")
    ax.grid(axis="y")

    ax = axs[2]
    fp = bm["FP_good"]
    labels = [
        "FP changed texts\n(20 good prompts)",
        "FP loops created\n(good prompts)",
        "transfer TPR\n(edited model)",
    ]
    vals = [
        fp["changed"] / 20,
        fp["looping"] / 20,
        bm["T_edited_transfer"]["tpr"] / max(bm["T_edited_transfer"]["n_loop"], 1),
    ]
    ax.bar(np.arange(3), vals, 0.5, color=[EDGE, SPIKE, BULK])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(np.arange(3), labels, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("(c) sentinel side effects / robustness")
    ax.grid(axis="y")
    fig.suptitle("RCDA-3: vaccination (online sentinel) vs cure (weight edits)", color=INK)
    _src(axs[0], "data: results/rcda3_sentinel.json + rcda_benchmark.json")
    _save(fig, out, dpi=160)
    return True


# ─────────────────────────────────────────────── fig14: RCDA-4 ──────────────
def fig14(rcda_dir: str, out: str) -> bool:
    res = os.path.join(rcda_dir, "results")
    sec = _load(os.path.join(res, "partC_secret.json"))
    heads = _load(os.path.join(res, "rcda3_heads.json"))
    trc = _load(os.path.join(res, "partC_partC_transcript.json"))
    base = _load(os.path.join(res, "partA_base_summary.json"))
    edited = _load(os.path.join(res, "partA_edited_summary.json"))
    if not (sec and heads and trc and base and edited):
        return False
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.4), constrained_layout=True)

    ok_a = False
    try:
        import re as _re

        agent_pick = json.loads(_re.search(r"\{[\s\S]*?\}", trc["final"]).group(0))
        inv = {v: k for k, v in sec["mapping"].items()}
        ax = axs[0]
        w = 0.27
        for i, lay in enumerate([15, 16, 18]):
            prof = heads["head_profiles"][str(lay)]["concentration"]
            lab = inv[str(lay)]
            x = np.arange(14) + (i - 1) * w
            ax.bar(x, prof, w, color=[BULK, SPIKE, EDGE][i], label=f"L{lay} ({lab})", alpha=0.9)
            pick = agent_pick[lab][0]
            ax.scatter([pick + (i - 1) * w], [prof[pick]], marker="v", color=INK, zorder=5)
        ax.set_xticks(range(14), [f"h{h}" for h in range(14)], fontsize=7)
        ax.set_ylabel("concentration (× uniform)")
        ax.set_title("(a) Part C: agent's top pick (▼) vs measured profile")
        ax.legend()
        ax.grid(axis="y")
        ok_a = True
    except Exception:
        if not ok_a:
            axs[0].set_title("(a) Part C: transcript unavailable")

    ax = axs[1]
    keys = [
        ("rho_bad_ru", "ρ loop (bad RU)"),
        ("finished_bad_ru", "finished (bad RU)"),
        ("rho_gsm8k_en", "ρ loop (GSM8K)"),
        ("finished_gsm8k_en", "finished (GSM8K)"),
    ]
    x = np.arange(len(keys))
    bv = [base[k] for k, _ in keys]
    ev = [edited[k] for k, _ in keys]
    ax.bar(x - 0.2, bv, 0.4, color=MUTED, label="base")
    ax.bar(x + 0.2, ev, 0.4, color=COLLAPSE, label="MIT edited")
    for i, (b, e) in enumerate(zip(bv, ev)):
        ax.text(i - 0.2, b + 0.02, f"{b:.2f}", ha="center", fontsize=8, color=INK)
        ax.text(i + 0.2, e + 0.02, f"{e:.2f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x, [l for _, l in keys], fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_title("(b) Part A: independent check of the MIT edit")
    ax.legend()
    ax.grid(axis="y")

    ax = axs[2]
    drew = False
    for tag, col in [("blind", MUTED), ("aided", SPIKE)]:
        tr = _load(os.path.join(res, f"partB_partB_{tag}_transcript.json"))
        if not tr:
            continue
        xs, ys, zs = [], [], []
        for e in tr.get("transcript", []):
            if "cell" not in e or "out" not in e:
                continue
            try:
                d = json.loads(e["out"][e["out"].index("{") : e["out"].rindex("}") + 1])
                if "rho_mini" in d:
                    xs.append(d.get("bench_calls_used", len(xs) + 1))
                    ys.append(d["rho_mini"])
                    zs.append(d.get("mean_loop_share", 0))
            except Exception:
                continue
        if xs:
            ax.plot(xs, ys, "o-", color=col, label=f"{tag} ρ_mini")
            ax.plot(xs, zs, "s--", color=col, alpha=0.6, label=f"{tag} mean loop_share")
            drew = True
    ax.set_xlabel("benchmark() calls used")
    ax.set_ylim(0, 1.05)
    ax.set_title("(c) Part B: GLM-programmer trajectories")
    if drew:
        ax.legend()
    ax.grid(alpha=0.3)
    fig.suptitle(
        "RCDA-4 weight-space programming: reading (C), verifying (A), writing (B)", color=INK
    )
    _src(axs[0], "data: results/partA_*, partB_*, partC_*, rcda3_heads.json")
    _save(fig, out, dpi=160)
    return True


# ─────────────────────────────────────────────── fig15: RCDA-5 ──────────────
def fig15(rcda_dir: str, out: str) -> bool:
    res = os.path.join(rcda_dir, "results")
    ev = {
        t: _load(os.path.join(res, f"part5_eval_{t}.json")) for t in ("base", "surgical", "trained")
    }
    log = _load(os.path.join(res, "training_log.json"))
    if not all(ev.values()) or not log:
        return False
    arms = ["base", "surgical\n(RCDA-2)", "trained\n(SGD, 150 steps)"]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), constrained_layout=True)

    ax = axes[0]
    x = np.arange(3)
    w = 0.36
    rb = [ev[t]["rho_bad"] for t in ("base", "surgical", "trained")]
    rg = [ev[t]["rho_good"] for t in ("base", "surgical", "trained")]
    ax.bar(x - w / 2, rb, w, label="ρ_bad (loops on bad)", color=COLLAPSE)
    ax.bar(x + w / 2, rg, w, label="ρ_good (loops on good)", color=MUTED)
    for xi, (b, g) in enumerate(zip(rb, rg)):
        ax.text(xi - w / 2, b + 0.02, f"{b:.2f}", ha="center", fontsize=9, color=INK)
        ax.text(xi + w / 2, g + 0.02, f"{g:.2f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(x, arms, fontsize=8.5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("share of looping generations")
    ax.set_title("(a) Fresh measurement: 20 bad + 20 good, greedy 96")
    ax.legend(loc="upper left")

    ax = axes[1]
    steps = [r["step"] for r in log["steps"]]
    pen = [r["pen_norm"] for r in log["steps"]]
    mse = [r["mse"] for r in log["steps"]]
    ax.plot(steps, pen, color=EDGE, label="penalty pen_norm (v-projection)²")
    ax.plot(steps, mse, color=BULK, label="MSE anchor (good tail)")
    ax.axhline(1.0, color=BULK, ls=":", lw=1, label="anchor cap")
    bx = [0, 75, 150]
    brho = [b["rho_mini"] for b in log["bench"]]
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(bx, brho, "o-", color=COLLAPSE, label="ρ_mini (bench)")
    for xi, yi in zip(bx, brho):
        ax2.annotate(
            f"{yi:.2f}",
            (xi, yi),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=8,
            color=COLLAPSE,
        )
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel("ρ_mini (8 bad, greedy)", color=COLLAPSE)
    ax2.tick_params(axis="y", labelcolor=COLLAPSE)
    ax.set_xlabel("training step")
    ax.set_ylabel("loss term value")
    ax.set_title("(b) Training: the formula in the loss (lr 3e-5, o_proj L15/16/18)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center left", bbox_to_anchor=(0.42, 0.62))

    ax = axes[2]
    cosv = {
        "surgical\n(RCDA-2)": [1.0, 1.0, 1.0],
        "trained\n(SGD)": [0.7249, 0.6383, 0.2919],
        "agent aided\n(RCDA-4 B)": [0.0539, 0.0539, 0.0539],
    }
    tw_status = ["pass×3", "pass×3", "violation"]
    x = np.arange(3)
    w = 0.26
    cols = {"surgical\n(RCDA-2)": GOOD, "trained\n(SGD)": BULK, "agent aided\n(RCDA-4 B)": COLLAPSE}
    for i, (name, vals) in enumerate(cosv.items()):
        ax.bar(x + (i - 1) * w, vals, w, label=name, color=cols[name])
    ax.set_xticks(x, ["L15", "L16", "L18"])
    ax.set_ylabel("|cos(u1 of the edit, v)|")
    ax.set_ylim(0, 1.15)
    ax.set_title("(c) Spectral purity of the edit: direction vs v")
    for i, s in enumerate(tw_status):
        ax.text(
            x[1] + (i - 1) * w, 1.03, s, ha="center", fontsize=7.5, color=cols[list(cosv.keys())[i]]
        )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=3)
    fig.suptitle(
        "RCDA-5: teach the model the formula — SGD vs surgery (MNN-Pocket-0.3B)", color=INK
    )
    _src(axes[0], "data: results/part5_eval_*.json + training_log.json")
    _save(fig, out, dpi=160)
    return True


# ─────────────────────────────── Graphic/ additions (top-level kit) ─────────
def graphic_additions() -> None:
    r5 = os.path.join(MNN, "R CDA - 5", "Research", "RCDA")
    outdir = os.path.join(MNN, "Graphic")

    probe = _load(os.path.join(r5, "results", "rcda2_probe.json"))
    if probe:
        layers = probe["layers"]
        li = [d["layer"] for d in layers]
        top = probe.get("top_layers", [])
        fig, ax = plt.subplots(figsize=(9.6, 5.2), constrained_layout=True)
        ax.semilogy(
            li,
            [d["lambda_max"] for d in layers],
            "o-",
            ms=4,
            lw=1.4,
            color=BULK,
            label=r"$\lambda_{max}$ (behavioral cov)",
        )
        ax.semilogy(
            li,
            [d["lambda_plus_mp"] for d in layers],
            "--",
            lw=1.4,
            color=MUTED,
            label=r"$\lambda_+$ (MP edge)",
        )
        for cd in top:
            ax.axvline(cd, color=SPIKE, ls="--", lw=1.0, alpha=0.85)
        for cd in top:
            ax.annotate(
                f"L{cd}",
                (cd, ax.get_ylim()[1] * 0.4),
                ha="center",
                fontsize=8,
                color=SPIKE,
                rotation=90,
            )
        ax.set_xlabel("transformer block (0-27)")
        ax.set_ylabel(r"eigenvalue $\lambda$")
        ax.set_title(
            "Where the repetition loop lives: λ_max vs the MP edge "
            "(amber = writer layers 15/16/17/18)"
        )
        ax.legend(loc="upper left")
        _src(ax, "data: R CDA - 5/Research/RCDA/results/rcda2_probe.json")
        _save(fig, os.path.join(outdir, "aurora_layer_profile.png"), dpi=200)

    b1 = _load(os.path.join(r5, "results", "rcda_benchmark.json"))
    b2 = _load(os.path.join(r5, "results", "rcda2_benchmark.json"))
    if b1 and b2:
        fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.2), constrained_layout=True)
        ax = axs[0]
        rounds = ["round 1", "round 2"]
        base = [b1["summary"]["rho_baseline"], b2["summary"]["rho_baseline"]]
        edit = [b1["summary"]["rho_edited"], b2["summary"]["rho_edited"]]
        x = np.arange(2)
        ax.bar(x - 0.18, base, 0.36, color=MUTED, label="baseline")
        ax.bar(x + 0.18, edit, 0.36, color=EDGE, label="edited (columns)")
        for i in range(2):
            ax.text(i - 0.18, base[i] + 0.02, f"{base[i]:.2f}", ha="center", fontsize=9, color=INK)
            ax.text(i + 0.18, edit[i] + 0.02, f"{edit[i]:.2f}", ha="center", fontsize=9, color=INK)
        ax.set_xticks(x, rounds)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("defect rate ρ (20 bad prompts)")
        ax.set_title("Surgical edits: ρ before/after")
        ax.legend()
        ax.grid(axis="y")

        ax = axs[1]
        ppl = [
            ("baseline", b2["summary"]["ppl_baseline"]),
            ("edited v2", b2["summary"]["ppl_edited"]),
        ]
        ax.bar([0, 1], [p for _, p in ppl], 0.5, color=[MUTED, GOOD])
        for i, (_, p) in enumerate(ppl):
            ax.text(i, p + 40, f"{p:.0f}", ha="center", fontsize=9, color=INK)
        ax.set_xticks([0, 1], [n for n, _ in ppl])
        ax.set_ylabel("PPL, control texts")
        ax.set_title("Abilities intact: PPL parity (TW-passed edits)")
        ax.grid(axis="y")
        fig.suptitle("MNN-Pocket-0.3B intervention benchmark (greedy, max_new 96)", color=INK)
        _src(axs[0], "data: R CDA - 5/Research/RCDA/results/rcda{,2}_benchmark.json")
        _save(fig, os.path.join(outdir, "aurora_benchmark_summary.png"), dpi=200)


# ─────────────────────────────── branded README summary charts ──────────────
def branded_charts() -> None:
    r5 = os.path.join(MNN, "R CDA - 5", "Research", "RCDA")
    res = os.path.join(r5, "results")
    outdir = os.path.join(MNN, "assets", "charts")
    os.makedirs(outdir, exist_ok=True)

    log = _load(os.path.join(res, "training_log.json"))
    if log:
        steps = [r["step"] for r in log["steps"]]
        pen = [r["pen_norm"] for r in log["steps"]]
        mse = [r["mse"] for r in log["steps"]]
        fig, ax = plt.subplots(figsize=(10.4, 4.6), constrained_layout=True)
        ax.plot(steps, pen, color=EDGE, lw=1.6, label="loop penalty pen_norm = (vᵀh/‖v‖)²")
        ax.plot(steps, mse, color=BULK, lw=1.6, label="abilities anchor (MSE)")
        ax.axhline(1.0, color=BULK, ls=":", lw=1.2)
        ax.text(2, 1.05, "anchor cap = 1.0", fontsize=8, color=BULK)
        bx = [b.get("step", 0) for b in log["bench"]]
        brho = [b["rho_mini"] for b in log["bench"]]
        bls = [b["mean_ls"] for b in log["bench"]]
        ax2 = ax.twinx()
        ax2.spines["right"].set_visible(True)
        ax2.plot(bx, brho, "o-", color=COLLAPSE, lw=2, ms=7, label="ρ_mini")
        ax2.plot(bx, bls, "s--", color=COLLAPSE, lw=1.2, alpha=0.7, label="mean loop_share")
        for xi, yi in zip(bx, brho):
            ax2.annotate(
                f"{yi:.2f}",
                (xi, yi),
                textcoords="offset points",
                xytext=(0, 9),
                ha="center",
                fontsize=9,
                color=COLLAPSE,
            )
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("loop rates (8 bad, greedy 96)", color=COLLAPSE)
        ax2.tick_params(axis="y", labelcolor=COLLAPSE)
        ax.set_xlabel("training step (Adam, lr 3e-5)")
        ax.set_ylabel("loss terms")
        ax.set_ylim(0, max(pen) * 1.1)
        ax.set_title("RCDA-5 · the formula as a loss: the loop melts, the abilities anchor holds")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="center right")
        _src(
            ax,
            "data: R CDA - 5/Research/RCDA/results/training_log.json "
            "(150 steps, o_proj L15/16/18, 0.8% params)",
        )
        _save(fig, os.path.join(outdir, "chart-training-dynamics.png"), dpi=200)

    ev = {
        t: _load(os.path.join(res, f"part5_eval_{t}.json")) for t in ("base", "surgical", "trained")
    }
    ppl = _load(os.path.join(res, "ppl_check.json"))
    if all(ev.values()) and ppl:
        fig, axs = plt.subplots(1, 3, figsize=(12.6, 4.0), constrained_layout=True)
        names = ["base", "surgical\n(RCDA-2)", "trained\n(SGD-150)"]
        x = np.arange(3)
        ax = axs[0]
        rb = [ev[t]["rho_bad"] for t in ("base", "surgical", "trained")]
        cols = [MUTED, EDGE, GOOD]
        ax.bar(x, rb, 0.55, color=cols)
        for i, v in enumerate(rb):
            ax.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=10, color=INK)
        ax.set_xticks(x, names, fontsize=8.5)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("ρ_bad (20 bad, greedy 96)")
        ax.set_title("(a) the defect rate")
        ax.grid(axis="y")

        ax = axs[1]
        lw_ = [ev[t]["loop_waste_mean"] for t in ("base", "surgical", "trained")]
        ax.bar(x, lw_, 0.55, color=cols)
        for i, v in enumerate(lw_):
            ax.text(i, v + 1.5, f"{v:.0f}", ha="center", fontsize=10, color=INK)
        ax.set_xticks(x, names, fontsize=8.5)
        ax.set_ylabel("loop tokens per gen (mean)")
        ax.set_title("(b) time wasted in the loop")
        ax.grid(axis="y")

        ax = axs[2]
        kl = [max(ev[t]["kl_ctrl"], 1e-4) for t in ("base", "surgical", "trained")]
        pm = [ppl[t]["ppl_control_mean"] for t in ("base", "surgical", "trained")]
        ax.bar(x - 0.19, kl, 0.38, color=EDGE, label="KL(base‖arm), log")
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1.5)
        for i, v in enumerate(kl):
            ax.text(
                i - 0.19,
                v * 1.25,
                f"{v:.2f}" if v > 1e-3 else "0",
                ha="center",
                fontsize=8,
                color=INK,
            )
        ax.set_xticks(x, names, fontsize=8.5)
        ax.set_ylabel("KL(base‖arm)")
        ax2 = ax.twinx()
        ax2.spines["right"].set_visible(True)
        ax2.bar(x + 0.19, pm, 0.38, color=GOOD, alpha=0.85, label="PPL ctrl")
        ax2.set_ylim(4700, 4980)
        for i, v in enumerate(pm):
            ax2.text(i + 0.19, v + 8, f"{v:.0f}", ha="center", fontsize=8, color=INK)
        ax2.set_ylabel("PPL, control texts")
        ax.set_title("(c) the price: drift vs abilities")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7.5)
        ax.grid(axis="y")
        fig.suptitle("RCDA-5 · three arms, one evaluator: surgery vs gradient training", color=INK)
        _src(axs[0], "data: results/part5_eval_*.json + ppl_check.json")
        _save(fig, os.path.join(outdir, "chart-three-arms.png"), dpi=200)

    dose = _load(os.path.join(res, "rcda2_dose.json"))
    if dose:
        dd = dose["dose_response"]
        al = [d["alpha"] for d in dd]
        fig, ax = plt.subplots(figsize=(9.6, 4.4), constrained_layout=True)
        ax.plot(
            al,
            [d["hit_bad"] for d in dd],
            "o-",
            color=COLLAPSE,
            lw=1.8,
            label="hit_bad (calibration defect rate)",
        )
        ax.plot(
            al,
            [d["hit_good"] for d in dd],
            "o-",
            color=BULK,
            lw=1.8,
            label="hit_good (side effects)",
        )
        ax.plot(al, [d["kl"] for d in dd], "s--", color=EDGE, lw=1.4, label="KL(base‖edited)")
        ax.set_xlabel(r"ablation strength $\alpha$ (dose of the surgical edit)")
        ax.set_ylabel("rate / KL")
        ax.set_title("RCDA-2 · dose-response: why one-direction ablation stalls")
        ax.legend()
        _src(
            ax, f"data: results/rcda2_dose.json (α* = {dose['alpha_star']:.1f}, TW-backoff at L18)"
        )
        _save(fig, os.path.join(outdir, "chart-dose-response.png"), dpi=200)

    probe = _load(os.path.join(res, "rcda2_probe.json"))
    heads = _load(os.path.join(res, "rcda3_heads.json"))
    if probe and heads:
        prof = heads["head_profiles"]
        fig, axs = plt.subplots(1, 2, figsize=(12.0, 4.2), constrained_layout=True)
        ax = axs[0]
        layers = probe["layers"]
        li = [d["layer"] for d in layers]
        zr = [d["z_diff"] for d in layers]
        ax.semilogy(li, zr, "-", color=BULK, lw=1.6)
        ax.fill_between(li, 1e-1, zr, color=BULK, alpha=0.12)
        top = probe.get("top_layers", [])
        for cd in top:
            ax.axvline(cd, color=SPIKE, ls="--", lw=1.2, alpha=0.9)
        for cd in top:
            ax.annotate(
                f"L{cd}", (cd, max(zr) * 0.55), ha="center", fontsize=9, color=SPIKE, rotation=90
            )
        ax.set_ylim(1e-1, max(zr) * 2)
        ax.set_xlabel("transformer block")
        ax.set_ylabel(r"$z_{diff}$ (Hotelling on behavioral $\Delta h$)")
        ax.set_title("(a) the loop writers: behavioral contrast by layer")
        ax.grid(axis="y")

        ax = axs[1]
        w = 0.26
        for i, (l, col) in enumerate([("15", BULK), ("16", SPIKE), ("18", EDGE)]):
            conc = np.array(prof[l]["concentration"])
            xs = np.arange(14) + (i - 1) * w
            ax.bar(xs, conc, w, color=col, alpha=0.9, label=f"L{l}")
            ax.scatter(
                [prof[l]["top_head"] + (i - 1) * w],
                [conc[prof[l]["top_head"]]],
                marker="v",
                color=INK,
                zorder=5,
            )
        ax.axhline(1.0, color=MUTED, ls="--", lw=1)
        ax.set_xticks(range(14), [f"h{h}" for h in range(14)], fontsize=7)
        ax.set_ylabel("concentration (× uniform)")
        ax.set_title("(b) down to attention heads (▼ = top head)")
        ax.legend()
        ax.grid(axis="y")
        fig.suptitle(
            "RCDA-2/3 · from spectrum to heads: localization of the repetition loop", color=INK
        )
        _src(axs[0], "data: results/rcda2_probe.json + rcda3_heads.json")
        _save(fig, os.path.join(outdir, "chart-layer-map.png"), dpi=200)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snap", type=int, choices=(2, 3, 4, 5), help="render one snapshot only")
    ap.add_argument(
        "--charts", action="store_true", help="only the branded README charts + Graphic additions"
    )
    args = ap.parse_args()

    builders = (
        ("fig5_rcds_layers", fig5),
        ("fig6_matrix_scans", fig6),
        ("fig7_behavioral_probe", fig7),
        ("fig8_dose_response", fig8),
        ("fig9_precursor_pr", fig9),
        ("fig10_token_attribution", fig10),
        ("fig11_sentinel", fig11),
        ("fig12_heads", fig12),
        ("fig13_therapy", fig13),
        ("fig14_rcda4", fig14),
        ("fig15_rcda5", fig15),
    )
    snaps = (args.snap,) if args.snap else (2, 3, 4, 5)
    for n in snaps:
        rcda_dir = os.path.join(MNN, f"R CDA - {n}", "Research", "RCDA")
        figdir = os.path.join(rcda_dir, "figures")
        print(f"== R CDA - {n} ==")
        made = 0
        for name, fn in builders:
            out = os.path.join(figdir, f"{name}.png")
            try:
                made += bool(fn(rcda_dir, out))
            except Exception as exc:
                print(f"  {name}: SKIPPED ({exc.__class__.__name__}: {exc})")
        print(f"  -> {made}/{len(builders)} figures")

    print("== Graphic additions ==")
    graphic_additions()
    if args.charts or not args.snap:
        print("== branded charts ==")
        branded_charts()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
