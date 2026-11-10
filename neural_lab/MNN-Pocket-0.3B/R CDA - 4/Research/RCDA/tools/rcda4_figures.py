#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rcda4_figures.py — fig14 for RCDA-4: (a) Part C head-reading vs measured
concentration, (b) Part A base-vs-edited bars, (c) Part B agent trajectories.
All panels guarded: missing data -> panel skipped, figure still produced.
"""
import json
import os

import matplotlib.font_manager as fm

fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf')
fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
import numpy as np  # noqa: E402

R4 = "/home/z/my-project/rcda_work/rcda4"
FIG = "/home/z/my-project/rcda_work/rcda3/figures"
os.makedirs(FIG, exist_ok=True)

fig, axs = plt.subplots(1, 3, figsize=(16, 4.4), constrained_layout=True)

# (a) Part C: measured concentration vs agent's ranking
try:
    sec = json.load(open(f"{R4}/partC_secret.json"))
    heads = json.load(open(f"{R4}/rcda3_heads.json"))["head_profiles"]
    tr = json.load(open(f"{R4}/partC/partC_transcript.json"))
    import re as _re
    ans = json.loads(_re.search(r"\{[\s\S]*?\}", tr["final"]).group(0))
    inv = {v: k for k, v in sec["mapping"].items()}  # layer -> label
    ax = axs[0]
    w = 0.27
    for i, lay in enumerate([15, 16, 18]):
        prof = heads[str(lay)]["concentration"]
        lab = inv[str(lay)]
        x = np.arange(14) + (i - 1) * w
        ax.bar(x, prof, w, color=["#1f77b4", "#ff7f0e", "#2ca02c"][i],
               label=f"L{lay} ({lab})")
        pick = ans[lab][0]
        ax.scatter([pick + (i - 1) * w], [prof[pick]], marker="v",
                   color="k", zorder=5)
    ax.set_xticks(range(14))
    ax.set_xticklabels([f"h{h}" for h in range(14)], fontsize=7)
    ax.set_ylabel("concentration (x uniform)")
    ax.set_title("(a) Part C: agent's top pick (▼) vs measured profile")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")
except Exception as e:
    axs[0].set_title(f"(a) Part C: no data ({e.__class__.__name__})")

# (b) Part A: base vs edited
try:
    base = json.load(open(f"{R4}/partA_base_summary.json"))
    edited = json.load(open(f"{R4}/partA_edited_summary.json"))
    ax = axs[1]
    keys = [("rho_bad_ru", "ρ loop (bad RU)"), ("finished_bad_ru", "finished (bad RU)"),
            ("rho_gsm8k_en", "ρ loop (GSM8K)"), ("finished_gsm8k_en", "finished (GSM8K)")]
    x = np.arange(len(keys))
    bv = [base[k] for k, _ in keys]
    ev = [edited[k] for k, _ in keys]
    ax.bar(x - 0.2, bv, 0.4, color="#7f7f7f", label="base")
    ax.bar(x + 0.2, ev, 0.4, color="#d62728", label="MIT edited")
    for i, (b, e) in enumerate(zip(bv, ev)):
        ax.text(i - 0.2, b + 0.02, f"{b:.2f}", ha="center", fontsize=8)
        ax.text(i + 0.2, e + 0.02, f"{e:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([l for _, l in keys], fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_title("(b) Part A: independent check of the MIT edit")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
except Exception as e:
    axs[1].set_title(f"(b) Part A: no data ({e.__class__.__name__})")

# (c) Part B: agent trajectories (rho_mini / mean_loop_share per benchmark call)
try:
    ax = axs[2]
    for tag, col in [("blind", "#7f7f7f"), ("aided", "#ff7f0e")]:
        tr = json.load(open(f"{R4}/partB/partB_{tag}_transcript.json"))
        xs, ys, zs = [], [], []
        for e in tr["transcript"]:
            if "cell" not in e or "rho_mini" not in e["out"]:
                continue
            try:
                d = json.loads(e["out"][e["out"].index("{"):e["out"].rindex("}") + 1])
                if "rho_mini" in d:
                    xs.append(d.get("bench_calls_used", len(xs) + 1))
                    ys.append(d["rho_mini"])
                    zs.append(d.get("mean_loop_share", 0))
            except Exception:
                pass
        if xs:
            ax.plot(xs, ys, "o-", color=col, label=f"{tag} ρ_mini")
            ax.plot(xs, zs, "s--", color=col, alpha=0.6,
                    label=f"{tag} mean loop_share")
    ax.set_xlabel("benchmark() calls used")
    ax.set_ylim(0, 1.05)
    ax.set_title("(c) Part B: GLM-programmer trajectories")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
except Exception as e:
    axs[2].set_title(f"(c) Part B: no data ({e.__class__.__name__})")

fig.suptitle("RCDA-4 weight-space programming: reading (C), verifying (A), writing (B)")
fig.savefig(os.path.join(FIG, "fig14_rcda4.png"), dpi=150)
plt.close(fig)
print("fig14 saved")
