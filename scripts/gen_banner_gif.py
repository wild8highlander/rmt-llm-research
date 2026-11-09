#!/usr/bin/env python3
"""Render the animated README banner: the BBP phase transition, live.

Every number on screen comes from ``src/rmt_llm`` itself — the bulk is
sampled with ``marchenko_pastur.mp_sample``, its density is drawn with
``mp_density``, and the outlier follows ``bbp_transition.bbp_lambda_max``.
Nothing here is hand-drawn decoration: the GIF is a running demo of the
package, which is exactly the claim the repository makes.

Usage::

    python scripts/gen_banner_gif.py            # writes assets/charts/bbp_transition_live.gif

Requires matplotlib + Pillow (both in the ``dev`` extra).
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

fm.fontManager.addfont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")

from rmt_llm.bbp_transition import bbp_critical_theta, bbp_lambda_max
from rmt_llm.marchenko_pastur import mp_bounds, mp_density, mp_sample

# --- scene parameters (q = 0.1, sigma2 = 1: the default activation regime)
Q, SIGMA2 = 0.1, 1.0
N, T = 480, 4800                     # n/t = q
THETA_C = bbp_critical_theta(Q)      # sqrt(q) * sigma2 = 0.3162...
LAMBDA_PLUS = (1 + np.sqrt(Q)) ** 2 * SIGMA2   # right MP edge
THETA_GRID = np.linspace(0.05, 1.4, 96)

# static physics, computed once
rng = np.random.default_rng(7)
bulk = mp_sample(N, T, SIGMA2, rng=rng)
lam = np.linspace(0.0, LAMBDA_PLUS * 1.6, 400)
rho = mp_density(lam, Q, SIGMA2)
lmax_curve = np.array([bbp_lambda_max(t, Q, SIGMA2) for t in THETA_GRID])

# --- Black Aurora palette (matches assets/branding)
BG, PANEL = "#0b1020", "#111a2e"
TEXT, MUTED = "#e6edf7", "#8fa3c8"
AURORA, CYAN, VIOLET, GOLD = "#39ff88", "#46e6e6", "#8a7bff", "#ffcb47"

plt.rcParams.update({
    "text.color": TEXT, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": TEXT,
})

fig, (ax_s, ax_t) = plt.subplots(
    1, 2, figsize=(7.2, 4.05), dpi=100, constrained_layout=True,
    gridspec_kw={"width_ratios": [1.45, 1.0]})
fig.set_facecolor(BG)
for ax in (ax_s, ax_t):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_alpha(0.4)

# left: spectrum panel
ax_s.hist(bulk, bins=60, density=True, color=CYAN, alpha=0.35,
          edgecolor=CYAN, linewidth=0.2, zorder=1)
ax_s.plot(lam, rho, color=AURORA, lw=2, zorder=2)
ax_s.axvline(LAMBDA_PLUS, color=MUTED, lw=1, ls="--", zorder=2)
XMAX = LAMBDA_PLUS * 1.6
edge_lbl = ax_s.text(LAMBDA_PLUS + 0.03, 0.9, "MP edge\n$\\lambda_+$",
                     color=MUTED, fontsize=7, va="top")
spike_v, = ax_s.plot([], [], color=GOLD, lw=1.4, ls=":", zorder=3)
spike, = ax_s.plot([], [], marker="v", color=GOLD, markersize=9, zorder=4)
out_lbl = ax_s.text(XMAX * 0.99, 1.30, "", color=GOLD, fontsize=8,
                    ha="right", va="center")
state_lbl = ax_s.text(0.03, 0.97, "", transform=ax_s.transAxes, fontsize=10,
                      fontweight="bold", va="top")
ax_s.set_xlim(0, LAMBDA_PLUS * 1.6)
ax_s.set_ylim(0, 1.55)
ax_s.set_xlabel("eigenvalue $\\lambda$")
ax_s.set_title("spectral snapshot", fontsize=10, pad=8)

# right: lambda_max(theta) trajectory panel
ax_t.axhline(LAMBDA_PLUS, color=MUTED, lw=1, ls="--")
ax_t.text(0.02, LAMBDA_PLUS + 0.7, "bulk edge $\\lambda_+$", color=MUTED,
          fontsize=7)
ax_t.axvline(THETA_C, color=VIOLET, lw=1, ls="--")
ax_t.axvspan(THETA_C, THETA_GRID[-1], color=VIOLET, alpha=0.10)
ax_t.text(THETA_C * 1.1, 0.95, "$\\theta_c$", color=VIOLET, fontsize=9)
ax_t.plot(THETA_GRID, lmax_curve, color=AURORA, lw=1.8, zorder=2)
marker, = ax_t.plot([], [], "o", color=GOLD, markersize=8, zorder=3)
trace, = ax_t.plot([], [], color=GOLD, lw=1.2, alpha=0.5, zorder=2)
ax_t.set_xlabel("signal strength $\\theta$")
ax_t.set_ylabel("$\\lambda_{\\max}(\\theta)$")
ax_t.set_xlim(THETA_GRID[0], THETA_GRID[-1])
ax_t.set_ylim(0, max(lmax_curve) * 1.12)
ax_t.set_title("outlier trajectory", fontsize=10, pad=8)

foot = fig.text(0.01, 0.005,
                "rendered live by src/rmt_llm: mp_sample · mp_density · "
                "bbp_lambda_max  |  wild8highlander/rmt-llm-research",
                color=MUTED, fontsize=6, ha="left")

def draw(k: int):
    theta = float(THETA_GRID[k])
    lmax = float(lmax_curve[k])
    super_ = theta > THETA_C
    y0, y1 = ax_s.get_ylim()
    lmax_vis = min(lmax, XMAX * 0.97)
    spike_v.set_data([lmax_vis, lmax_vis], [0, y1])
    spike.set_data([lmax_vis], [1.42])
    out_lbl.set_text("")
    if lmax > XMAX * 0.97:
        out_lbl.set_text("$\\lambda_{\\max}$=%.1f $\\rightarrow$" % lmax)
    state_lbl.set_text(
        "$\\theta=%.2f$  %s  $\\theta_c=%.2f$\n%s" % (
            theta, ">" if super_ else "<", THETA_C,
            "SUPERCITICAL — outlier escaped" if super_
            else "subcritical — outlier hidden in bulk"))
    state_lbl.set_color(GOLD if super_ else MUTED)
    marker.set_data([theta], [lmax])
    trace.set_data(THETA_GRID[:k + 1], lmax_curve[:k + 1])
    return spike_v, spike, out_lbl, state_lbl, marker, trace

anim = FuncAnimation(fig, draw, frames=len(THETA_GRID), blit=False)
out = os.path.join(REPO, "assets", "charts", "bbp_transition_live.gif")
anim.save(out, writer=PillowWriter(fps=24))
print("wrote", out)
