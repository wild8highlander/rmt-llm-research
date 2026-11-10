#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RCDA-2 stage 3 — figures + numeric digest (no model needed).

Inputs:  rcda_work/rcda2/rcda2_probe.json, rcda2_dose.json, rcda2_benchmark (json)
Outputs: fig7..fig10 (PNG, 150 dpi) + printed digest for the report.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
fm.fontManager.addfont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Noto Sans SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

fm.fontManager.addfont("/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf")

WORK = "/home/z/my-project/rcda_work"
OUT = os.path.join(WORK, "rcda2")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

probe = json.load(open(os.path.join(OUT, "rcda2_probe.json")))
dose_path = os.path.join(OUT, "rcda2_dose.json")
if not os.path.exists(dose_path):
    # stage 2 stores the dose grid inside the edit-pack meta
    er = os.path.join(OUT, "rcda2_edit_report.json")
    if os.path.exists(er):
        meta = json.load(open(er)).get("meta", {})
        dose = {"alpha_star": meta.get("alpha_star"), "note": meta.get("note"),
                "base_cal": meta.get("base_cal"),
                "dose_response": meta.get("dose_response")} if meta.get("dose_response") else None
    else:
        dose = None
else:
    dose = json.load(open(dose_path))
bench_path = os.path.join(WORK, "bench_v2/rcda_benchmark.json")
bench = json.load(open(bench_path)) if os.path.exists(bench_path) else None

# ---------------------------------------------------------------- fig7: probe
layers = probe["layers"]
li = [d["layer"] for d in layers]
top = probe["top_layers"]
fig, axes = plt.subplots(2, 1, figsize=(9, 6.8), constrained_layout=True)
axes[0].plot(li, [d["z_diff"] for d in layers], "o-", ms=3, lw=1.2,
             color="#2ca02c", label=r"$z_{diff}$ (Hotelling, behavioral $\Delta h$)")
for cd in top:
    axes[0].axvline(cd, color="#ff7f0e", ls="--", lw=0.8, alpha=0.7)
axes[0].set_yscale("log")
axes[0].set_ylabel("z-score")
axes[0].set_title("RCDA-2: behavioral contrast (loop vs clean tokens, same prompts)")
cos_l = [d for d in layers if d.get("cosine_vs_style")]
if cos_l:
    names = sorted({k for d in cos_l for k in d["cosine_vs_style"]})
    xs = np.arange(len(cos_l))
    for k in names:
        axes[1].plot(xs, [abs(d["cosine_vs_style"].get(k, np.nan)) for d in cos_l],
                     "s-", ms=4, lw=1.2, label=f"|cos| vs style {k}")
    axes[1].set_xticks(xs, [f"L{d['layer']}" for d in cos_l])
    axes[1].axhline(0.3, color="k", ls=":", lw=1.0)
    axes[1].text(0, 0.31, " 0.3", fontsize=8)
    axes[1].set_ylabel("|cosine|")
    axes[1].set_title("Overlap of the round-1 style direction with the round-2 behavioral direction")
    axes[1].legend(fontsize=8)
for ax in axes:
    ax.grid(alpha=0.3)
fig.savefig(os.path.join(FIG, "fig7_behavioral_probe.png"), dpi=150)
plt.close(fig)

# ------------------------------------------------------- fig8: dose-response
if dose:
    dd = dose["dose_response"]
    al = [d["alpha"] for d in dd]
    fig, ax1 = plt.subplots(figsize=(9, 4.6), constrained_layout=True)
    ax1.plot(al, [d["kl"] for d in dd], "o-", color="#1f77b4", label="mean KL(base||edited)")
    ax1.set_xlabel(r"ablation strength $\alpha$")
    ax1.set_ylabel("KL", color="#1f77b4")
    ax2 = ax1.twinx()
    w = 0.03
    ax2.bar([a - w for a in al], [d["changed"]["bad"] for d in dd], width=2*w,
            color="#d62728", alpha=0.7, label="changed bad trajectories")
    ax2.bar([a + w for a in al], [d["changed"]["good"] for d in dd], width=2*w,
            color="#ff7f0e", alpha=0.7, label="changed good trajectories")
    ax2.plot(al, [d["rep4_bad"] for d in dd], "s--", color="#2ca02c",
             label="mean RepRate-4 (bad)")
    ax2.axhline(dose["base_cal"]["rep4_bad"], color="#2ca02c", ls=":", lw=1)
    ax2.set_ylabel("trajectories changed / rep4", color="#d62728")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
    ax1.set_title("Dose-response of the behavioral RCDA edit (calibration subset)")
    ax1.grid(alpha=0.3)
    fig.savefig(os.path.join(FIG, "fig8_dose_response.png"), dpi=150)
    plt.close(fig)

# ----------------------------------------------------------- fig9: precursor
prec = probe["precursor"]
sel_layers = sorted(prec["loop"].keys(), key=int)
fig, axes = plt.subplots(1, len(sel_layers), figsize=(4.2 * len(sel_layers), 4.2),
                         constrained_layout=True, sharey=True)
if len(sel_layers) == 1:
    axes = [axes]
for ax, lk in zip(axes, sel_layers):
    for key, color in (("pre", "#1f77b4"), ("loop", "#d62728")):
        data = prec[key].get(lk, [])
        if not data:
            continue
        lags = np.array([r[0] for r in data], float)
        prs = np.array([r[1] for r in data], float)
        order = np.argsort(lags)
        uq = np.unique(lags[order])
        m = [prs[lags == u].mean() for u in uq]
        ax.plot(uq, m, "o-", ms=3, lw=1.2, color=color,
                label=("looping regime" if key == "loop" else "pre-collapse"))
    cl = prec["clean"].get(lk, [])
    if cl:
        prs = np.array([r[1] for r in cl], float)
        ax.axhspan(prs.mean() - prs.std(), prs.mean() + prs.std(),
                   color="#2ca02c", alpha=0.15)
        ax.axhline(prs.mean(), color="#2ca02c", ls="--", lw=1.2,
                   label="clean generations")
    ax.axvline(0, color="k", ls=":", lw=1)
    ax.set_title(f"layer {lk}", fontsize=10)
    ax.set_xlabel("lag vs collapse onset (tokens)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
axes[0].set_ylabel("participation ratio (window 16)")
fig.suptitle("Spectral precursor: windowed PR around the collapse onset", fontsize=11)
fig.savefig(os.path.join(FIG, "fig9_precursor_pr.png"), dpi=150)
plt.close(fig)

# ------------------------------------------------------ fig10: attribution
attr = probe["token_attribution"]
lk = sorted(attr.keys())[0]
fig, ax = plt.subplots(figsize=(9, 5.4), constrained_layout=True)
b = attr[lk]["boosted"][:10]
s = attr[lk]["suppressed"][:10]
toks = [t for t, _ in b] + [t for t, _ in s]
vals = [v / max(abs(v) for _, v in b + s) for _, v in b + s]
colors = ["#d62728"] * len(b) + ["#1f77b4"] * len(s)
ypos = np.arange(len(toks))[::-1]
ax.barh(ypos, vals, color=colors)
ax.set_yticks(ypos, [repr(t)[:22] for t in toks], fontsize=9)
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel(r"relative $E\,\Delta h$ score (tied embeddings)")
ax.set_title(f"Token attribution of the behavioral loop direction, {lk}")
ax.grid(alpha=0.3, axis="x")
fig.savefig(os.path.join(FIG, "fig10_token_attribution.png"), dpi=150)
plt.close(fig)

# ------------------------------------------------------------------- digest
print("=== RCDA-2 digest ===")
print("top layers:", probe["top_layers"])
for d in sorted(layers, key=lambda x: -x["z_diff"])[:4]:
    print(f"  L{d['layer']}: z_diff={d['z_diff']:.1f} cos_style={d['cosine_vs_style']}")
if dose:
    print("alpha*:", dose["alpha_star"], "|", dose["note"])
    for r in dose["dose_response"]:
        print(f"  a={r['alpha']}: KL={r['kl']:.4f} changed={r['changed']} "
              f"rep4_bad={r['rep4_bad']:.3f} hit_bad={r['hit_bad']:.2f}")
if bench:
    s = bench["summary"]
    print("benchmark: rho", s["rho_baseline"], "->", s["rho_edited"],
          "| fpr", s["fpr_baseline"], "->", s["fpr_edited"],
          "| changed:", s.get("n_changed_generations"))
print("figures saved to", FIG)
