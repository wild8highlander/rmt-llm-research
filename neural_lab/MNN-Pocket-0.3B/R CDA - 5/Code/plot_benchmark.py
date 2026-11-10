# -*- coding: utf-8 -*-
"""plot_benchmark.py — summary chart for the RMT-intervention benchmark.

Reads results.json from the benchmark directory and draws 3 panels:
  Distinct-4 (higher is better) / Repetition Rate-4 (lower is better) /
  PPL on a log scale (lower is better), across configurations;
the dashed line marks the baseline level.

Usage: python plot_benchmark.py [benchmark_dir]
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

out_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
with open(os.path.join(out_dir, "results.json"), encoding="utf-8") as f:
    data = json.load(f)

runs = {r["config"]["name"]: r for r in data["runs"]}

LABELS = {
    "baseline": "baseline",
    "clip": "clip",
    "dampen_a0.9": "dampen\nα=0.9",
    "dampen_a0.5": "dampen\nα=0.5",
    "dampen_a0.1": "dampen\nα=0.1",
    "project_k16": "project\ntop-16",
    "random_abl": "random\n(ablation)",
}

def metrics(name):
    r = runs[name]
    rows = r["rows"]
    d4 = np.mean([x["distinct4"] for x in rows if x["distinct4"] is not None])
    rr = np.mean([x["rep4"] for x in rows if x["rep4"] is not None])
    ppls = [p for p in r["ppl"] if p is not None and np.isfinite(p)]
    ppl = np.mean(ppls) if ppls else np.nan
    return d4, rr, ppl

names = [n for n in LABELS if n in runs]
vals = {n: metrics(n) for n in names}
base_ppl = vals["baseline"][2]

fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.4), constrained_layout=True)
panels = [
    ("Distinct-4 (higher is better)", 0, "#2a7fb8"),
    ("Repetition Rate-4 (lower is better)", 1, "#c25b2e"),
    ("Perplexity, log scale (lower is better)", 2, "#3a8f5f"),
]
x = np.arange(len(names))
for ax, (title, idx, color) in zip(axes, panels):
    v = [vals[n][idx] for n in names]
    ax.bar(x, v, color=color, alpha=0.85, width=0.62)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[n] for n in names], fontsize=8.5)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    if idx == 0:
        ax.axhline(vals["baseline"][0], ls="--", lw=1, color="#444")
        ax.set_ylim(0, 1)
        for xi, vi in zip(x, v):
            ax.text(xi, vi + 0.02, f"{vi:.2f}", ha="center", fontsize=8.5)
    elif idx == 1:
        ax.axhline(vals["baseline"][1], ls="--", lw=1, color="#444")
        ax.set_ylim(0, 1)
        for xi, vi in zip(x, v):
            ax.text(xi, vi + 0.02, f"{vi:.2f}", ha="center", fontsize=8.5)
    else:
        ax.set_yscale("log")
        ax.axhline(base_ppl, ls="--", lw=1, color="#444")
        for xi, vi in zip(x, v):
            if np.isfinite(vi):
                ax.text(xi, vi * 1.15, f"{vi:.0f}", ha="center", fontsize=8.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

fig.suptitle("RMT interventions on MNN-Pocket-0.3B: greedy generation, 50 tokens, "
             f"{data['meta']['n_prompts']} prompts, use_cache=False", fontsize=12)
fig.savefig(os.path.join(out_dir, "benchmark_summary.png"), dpi=150)
print("saved:", os.path.join(out_dir, "benchmark_summary.png"))

from PIL import Image
im = Image.open(os.path.join(out_dir, "benchmark_summary.png")).convert("L")
h = im.histogram()
print(im.size, "non-white ratio:", round(1 - h[255] / sum(h), 4))
