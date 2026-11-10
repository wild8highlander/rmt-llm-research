#!/usr/bin/env python3
"""RCDA-5 figure: fig15_rcda5.png — three arms of curing the loop."""
import json

import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf')
fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

R5 = "/home/z/my-project/rcda_work/rcda5"
FIG = "/home/z/my-project/rcda_work/rcda3/figures/fig15_rcda5.png"

arms = ["base", "surgical\n(RCDA-2)", "trained\n(SGD, 150 steps)"]
ev = {t: json.load(open(f"{R5}/part5_eval_{t}.json"))
      for t in ("base", "surgical", "trained")}
log = json.load(open(f"{R5}/training_log.json"))

fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), constrained_layout=True)

# (a) loop rates
ax = axes[0]
x = np.arange(3)
w = 0.36
rb = [ev[t]["rho_bad"] for t in ("base", "surgical", "trained")]
rg = [ev[t]["rho_good"] for t in ("base", "surgical", "trained")]
ax.bar(x - w / 2, rb, w, label="ρ_bad (loops on bad)", color="#c0392b")
ax.bar(x + w / 2, rg, w, label="ρ_good (loops on good)", color="#7f8c8d")
for xi, (b, g) in enumerate(zip(rb, rg)):
    ax.text(xi - w / 2, b + 0.02, f"{b:.2f}", ha="center", fontsize=9)
    ax.text(xi + w / 2, g + 0.02, f"{g:.2f}", ha="center", fontsize=9)
ax.set_xticks(x, arms, fontsize=8.5)
ax.set_ylim(0, 1.0)
ax.set_ylabel("share of looping generations")
ax.set_title("(a) Fresh measurement: 20 bad + 20 good, greedy 96")
ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(0.0, 1.0))

# (b) training curves
ax = axes[1]
steps = [r["step"] for r in log["steps"]]
pen = [r["pen_norm"] for r in log["steps"]]
mse = [r["mse"] for r in log["steps"]]
ax.plot(steps, pen, color="#2c3e50", label="penalty pen_norm (v-projection)²")
ax.plot(steps, mse, color="#2980b9", label="MSE anchor (good tail)")
ax.axhline(1.0, color="#2980b9", ls=":", lw=1, label="anchor cap")
bx = [0, 75, 150]
brho = [b["rho_mini"] for b in log["bench"]]
ax2 = ax.twinx()
ax2.plot(bx, brho, "o-", color="#c0392b", label="ρ_mini (bench)")
for xi, yi in zip(bx, brho):
    ax2.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points",
                 xytext=(0, 6), ha="center", fontsize=8, color="#c0392b")
ax2.set_ylim(0, 1.0)
ax2.set_ylabel("ρ_mini (8 bad, greedy)", color="#c0392b")
ax.set_xlabel("training step")
ax.set_ylabel("loss term value")
ax.set_title("(b) Training: the formula in the loss (lr 3e-5, o_proj L15/16/18)")
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="center left",
          bbox_to_anchor=(0.42, 0.62))

# (c) spectral comparison
ax = axes[2]
labels = ["surgical\n(RCDA-2)", "trained\n(SGD)", "agent aided\n(RCDA-4 B)"]
cosv = {
    "surgical\n(RCDA-2)": [1.0, 1.0, 1.0],
    "trained\n(SGD)": [0.7249, 0.6383, 0.2919],
    "agent aided\n(RCDA-4 B)": [0.0539, 0.0539, 0.0539],
}
tw_status = ["pass×3", "pass×3", "violation"]
x = np.arange(3)
w = 0.26
cols = {"surgical\n(RCDA-2)": "#27ae60", "trained\n(SGD)": "#2980b9",
        "agent aided\n(RCDA-4 B)": "#c0392b"}
for i, (name, vals) in enumerate(cosv.items()):
    ax.bar(x + (i - 1) * w, vals, w, label=name, color=cols[name])
ax.set_xticks(x, ["L15", "L16", "L18"])
ax.set_ylabel("|cos(u1 of the edit, v)|")
ax.set_ylim(0, 1.15)
ax.set_title("(c) Spectral purity of the edit: direction vs v")
for i, s in enumerate(tw_status):
    ax.text(x[1] + (i - 1) * w, 1.03, s, ha="center", fontsize=7.5,
            color=cols[list(cosv.keys())[i]])
ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=3)

fig.suptitle("RCDA-5: teach the model the formula — SGD vs surgery (MNN-Pocket-0.3B)",
             fontsize=11)
fig.savefig(FIG, dpi=160)
print("saved", FIG)
