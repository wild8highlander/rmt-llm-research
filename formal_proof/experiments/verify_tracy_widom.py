#!/usr/bin/env python3
"""Experiment 3 -- Tracy-Widom edge fluctuations: beta=1 vs beta=2.

Verifies (Step 13) that after the Johnstone centering/scaling

    mu_NT = sigma^2 (sqrt(T-1) + sqrt(N))^2 / T,
    s_NT  = sigma^2 (sqrt(T-1) + sqrt(N)) (1/sqrt(T-1) + 1/sqrt(N))^{1/3} / T,

the largest eigenvalue of a REAL Wishart matrix converges to the F_1 law
(beta = 1 -- the class relevant for real activations of language models),
while the COMPLEX Wishart converges to F_2 (beta = 2).  Reference samples
of F_1 / F_2 are produced from GOE / GUE largest eigenvalues standardized
by their asymptotic edge, so the comparison needs no external tables.

Calibration anchors (Bornemann tables / Wikipedia):
    F1: mean -1.2065335745820, std 1.267981..., skew 0.29346452408
    F2: mean -1.7710868074110, std 0.901775...,  skew 0.224084203610
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy import stats

from _style import AUR, INK, MUTED, PANEL, banner, save
from rmt_fp import tracy_widom as tw

TRIALS = 2000
N_DIMS, N_SAMPLES = 220, 440


def main() -> None:
    rng = np.random.default_rng(404)
    print(f"Tracy-Widom check: N={N_DIMS}, T={N_SAMPLES}, {TRIALS} trials per ensemble")

    # -- real Wishart, standardized by Johnstone --------------------------------
    mu_j, s_j = tw.johnstone_center_scale(N_SAMPLES, N_DIMS)
    vals = np.empty(TRIALS)
    for i in range(TRIALS):
        x = rng.standard_normal((N_DIMS, N_SAMPLES))
        vals[i] = np.linalg.eigvalsh(x @ x.T / N_SAMPLES)[-1]
    real_std = (vals - mu_j) / s_j

    # -- complex Wishart ---------------------------------------------------------
    vals_c = np.empty(TRIALS)
    for i in range(TRIALS):
        x = (rng.standard_normal((N_DIMS, N_SAMPLES))
             + 1j * rng.standard_normal((N_DIMS, N_SAMPLES))) / np.sqrt(2.0)
        vals_c[i] = np.linalg.eigvalsh(x @ x.T.conj() / N_SAMPLES)[-1]
    cplx_std = (vals_c - mu_j) / s_j

    # -- GOE / GUE reference (large-p TW generators) ------------------------------
    p_ref = 300
    ref1 = tw.tw_reference_sample(1, 1400, p=p_ref, rng=rng)
    ref2 = tw.tw_reference_sample(2, 1400, p=p_ref, rng=rng)

    ks_1 = stats.ks_2samp(real_std, ref1)
    ks_2 = stats.ks_2samp(cplx_std, ref2)
    ks_cross = stats.ks_2samp(real_std, cplx_std)
    print(f"  real   Wishart: mean={real_std.mean():+.4f} std={real_std.std():.4f} "
          f"skew={stats.skew(real_std):+.4f}   (F1: -1.2065 / 1.2678 / 0.2935)")
    print(f"  complex Wishart: mean={cplx_std.mean():+.4f} std={cplx_std.std():.4f} "
          f"skew={stats.skew(cplx_std):+.4f}   (F2: -1.7711 / 0.9018 / 0.2241)")
    print(f"  KS real-vs-GOE(F1):  stat={ks_1.statistic:.4f} p={ks_1.pvalue:.3f}")
    print(f"  KS complex-vs-GUE(F2): stat={ks_2.statistic:.4f} p={ks_2.pvalue:.3f}")
    print(f"  KS real-vs-complex:    stat={ks_cross.statistic:.4f} "
          f"p={ks_cross.pvalue:.2e}  (beta classes differ)")

    # -- figure -------------------------------------------------------------------
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    banner(fig, "Tracy-Widom edge universality with the Johnstone normalization (Step 13)",
           f"N = {N_DIMS}, T = {N_SAMPLES}, {TRIALS} trials; GOE/GUE references at p = {p_ref}")

    ax = axes[0]
    bins = np.linspace(-5.2, 2.6, 70)
    ax.hist(real_std, bins=bins, density=True, color=AUR[0], alpha=0.6,
            label="real Wishart (Johnstone-std)")
    ax.hist(ref1, bins=bins, density=True, histtype="step", color=INK, lw=1.8,
            label="GOE reference $\\to F_1$")
    ax.set_xlabel("standardized $\\lambda_{max}$")
    ax.set_ylabel("density")
    ax.set_title(f"$\\beta=1$: KS = {ks_1.statistic:.3f}, p = {ks_1.pvalue:.2f}")
    ax.legend(loc="upper left")

    ax = axes[1]
    ax.hist(cplx_std, bins=bins, density=True, color=AUR[3], alpha=0.6,
            label="complex Wishart (Johnstone-std)")
    ax.hist(ref2, bins=bins, density=True, histtype="step", color=INK, lw=1.8,
            label="GUE reference $\\to F_2$")
    ax.set_xlabel("standardized $\\lambda_{max}$")
    ax.set_ylabel("density")
    ax.set_title(f"$\\beta=2$: KS = {ks_2.statistic:.3f}, p = {ks_2.pvalue:.2f}")
    ax.legend(loc="upper left")

    save(fig, "tracy_widom_verification.png")


if __name__ == "__main__":
    main()
