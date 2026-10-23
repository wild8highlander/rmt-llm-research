#!/usr/bin/env python3
"""Experiment 6 -- Layer operator calculus (Step 2).

Verifies the derived results of chapter 2:

  1. RMSNorm trace conservation: after Norm(x) = sqrt(N) x/||x|| every
     column has norm sqrt(N), hence Tr C = N exactly (deviation ~ 1e-13);
  2. GELU moment closed forms (Price/Stein) vs Monte Carlo;
  3. the bivariate SiLU kernel (Gauss-Hermite) vs Monte Carlo;
  4. the full GELU-layer covariance transform  Cov(GELU(Wx))  predicted by
     the pairwise kernels vs the empirical covariance of the layer output.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from _style import AUR, INK, MUTED, PANEL, banner, save
from rmt_fp import layer_ops as lo

RNG = np.random.default_rng(707)


def main() -> None:
    # -- 1. RMSNorm ------------------------------------------------------------
    dev = lo.rmsnorm_trace_check(256, 64)
    print(f"1. RMSNorm trace conservation: |Tr C - N| = {dev:.2e}")

    # -- 2. GELU moments ---------------------------------------------------------
    print("2. GELU moments (closed form vs MC):")
    gam2s = [0.25, 1.0, 2.5]
    mc_mean_errs, mc_var_errs = [], []
    for g2 in gam2s:
        m, v2, var = lo.gelu_moments(g2)
        x = RNG.standard_normal(400_000) * np.sqrt(g2)
        g = lo.gelu(x)
        mc_mean_errs.append(abs(g.mean() - m))
        mc_var_errs.append(abs(g.var() - var))
        print(f"   gamma^2={g2:4.2f}: E={m:+.5f} (mc {g.mean():+.5f}), "
              f"Var={var:.5f} (mc {g.var():.5f})")

    # -- 3. SiLU bivariate kernel ------------------------------------------------
    print("3. SiLU kernel E[SiLU(u)SiLU(v)] (Gauss-Hermite vs MC):")
    ker_errs = []
    for var_u, var_v, rho in [(1.0, 1.0, 0.5), (2.0, 0.7, -0.3), (0.5, 0.5, 0.9)]:
        quad = lo.bivariate_kernel(lambda u, v: lo.silu(u) * lo.silu(v),
                                   var_u, var_v, rho)
        n_mc = 2_000_000
        z1 = RNG.standard_normal(n_mc)
        z2 = rho * z1 + np.sqrt(1 - rho**2) * RNG.standard_normal(n_mc)
        mc = np.mean(lo.silu(np.sqrt(var_u) * z1) * lo.silu(np.sqrt(var_v) * z2))
        ker_errs.append(abs(quad - mc))
        print(f"   var=({var_u},{var_v}), rho={rho:+.1f}: quad={quad:.5f} (mc {mc:.5f})")

    # -- 4. full GELU-layer covariance -------------------------------------------
    print("4. GELU-layer covariance transform (kernel prediction vs empirical):")
    k, n_in, n_samp = 10, 150, 400_000
    w = RNG.standard_normal((k, n_in)) / np.sqrt(n_in)
    # random correlated input covariance (low effective rank + isotropic floor)
    a = RNG.standard_normal((n_in, 40))
    c_in = a @ a.T / 40 + 0.3 * np.eye(n_in)
    c_pred = lo.gelu_covariance(w, c_in)
    l_mat = np.linalg.cholesky(c_in + 1e-12 * np.eye(n_in))
    x = l_mat @ RNG.standard_normal((n_in, n_samp))
    y = lo.gelu(w @ x)
    c_emp = np.cov(y)
    rel = float(np.linalg.norm(c_pred - c_emp) / np.linalg.norm(c_emp))
    print(f"   relative Frobenius deviation: {rel:.4f} (MC noise at {n_samp} samples)")

    # -- figure -------------------------------------------------------------------
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.1), constrained_layout=True)
    banner(fig, "Layer-operator calculus: RMSNorm, GELU/SiLU kernels (Step 2)",
           "closed forms from Price/Stein vs Monte Carlo; Gaussian assumption stated explicitly")

    ax = axes[0]
    g2s = np.linspace(0.05, 4.0, 60)
    means = np.array([lo.gelu_moments(g)[0] for g in g2s])
    vars_ = np.array([lo.gelu_moments(g)[2] for g in g2s])
    mc_m, mc_v = [], []
    for g in g2s[::6]:
        x = RNG.standard_normal(300_000) * np.sqrt(g)
        gval = lo.gelu(x)
        mc_m.append(gval.mean())
        mc_v.append(gval.var())
    ax.plot(g2s, means, color=AUR[0], lw=2.0, label=r"$E[\mathrm{GELU}]$")
    ax.plot(g2s, vars_, color=AUR[2], lw=2.0, label=r"$\mathrm{Var}[\mathrm{GELU}]$")
    ax.plot(g2s[::6], mc_m, "o", ms=5, mfc=PANEL, mec=AUR[0], mew=1.4)
    ax.plot(g2s[::6], mc_v, "s", ms=5, mfc=PANEL, mec=AUR[2], mew=1.4)
    ax.set_xlabel(r"$\gamma^2 = \mathrm{Var}(x)$")
    ax.set_title("GELU moment closed forms")
    ax.legend(loc="lower right")

    ax = axes[1]
    rhos = np.linspace(-0.95, 0.95, 39)
    for var_u, var_v, c in [(1.0, 1.0, AUR[0]), (2.0, 0.7, AUR[2]), (0.5, 1.6, AUR[3])]:
        vals = [lo.bivariate_kernel(lambda u, v: lo.silu(u) * lo.silu(v),
                                    var_u, var_v, float(r)) for r in rhos]
        ax.plot(rhos, vals, lw=2.0, color=c,
                label=f"var=({var_u}, {var_v})")
    pts = [(1.0, 1.0, 0.5), (2.0, 0.7, -0.3), (0.5, 0.5, 0.9)]
    mc_pts = []
    for var_u, var_v, rho in pts:
        n_mc = 1_500_000
        z1 = RNG.standard_normal(n_mc)
        z2 = rho * z1 + np.sqrt(1 - rho**2) * RNG.standard_normal(n_mc)
        mc_pts.append(np.mean(lo.silu(np.sqrt(var_u) * z1) * lo.silu(np.sqrt(var_v) * z2)))
    ax.plot([p[2] for p in pts], mc_pts, "o", ms=7, mfc=PANEL, mec=INK, mew=1.4,
            label="MC check")
    ax.set_xlabel(r"correlation $\rho$")
    ax.set_ylabel(r"$E[\mathrm{SiLU}(u)\mathrm{SiLU}(v)]$")
    ax.set_title("SiLU bivariate kernel")
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[2]
    im = ax.imshow(c_pred - c_emp, cmap="RdBu_r", vmin=-0.02, vmax=0.02)
    ax.set_title(f"GELU covariance transform, rel. dev {rel:.3f}")
    ax.set_xlabel("output unit")
    ax.set_ylabel("output unit")
    fig.colorbar(im, ax=ax, shrink=0.85)

    save(fig, "layer_ops_verification.png")


if __name__ == "__main__":
    main()
