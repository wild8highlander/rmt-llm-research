#!/usr/bin/env python3
"""Experiment 1 -- Marchenko-Pastur bulk: density, edges, Stieltjes transform.

Verifies (Steps 4-5):
  1. the empirical spectrum of C = XX^T / T follows the MP density with the
     derived edges lambda_pm = sigma^2 (1 +- sqrt(q))^2;
  2. the closed-form Stieltjes transform (physical root of the Silverstein
     quadratic) matches the numerically computed resolvent, both on the real
     axis above the edge and in the complex upper half-plane;
  3. the Silverstein fixed-point equation holds for the closed form.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from _style import AUR, INK, MUTED, PANEL, banner, save
from rmt_fp import mp

RNG = np.random.default_rng(101)


def empirical_spectrum(n_dims: int, n_samples: int, sigma2: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_dims, n_samples)) * np.sqrt(sigma2)
    return np.linalg.eigvalsh(x @ x.T / n_samples)


def main() -> None:
    q, sigma2 = 0.35, 1.7
    n_dims, n_samples = 900, 2571
    lamm, lamp = mp.mp_edges(q, sigma2)
    print(f"MP check: q={q}, sigma2={sigma2}, N={n_dims}, T={n_samples}")
    print(f"  derived edges: lambda_-={lamm:.4f}, lambda_+={lamp:.4f}")

    # -- 1. density vs spectrum ------------------------------------------------
    eigs = empirical_spectrum(n_dims, n_samples, sigma2, seed=11)
    grid = np.linspace(0.01, lamp * 1.25, 800)
    rho = mp.mp_density(grid, q, sigma2)
    inside = eigs[(eigs > lamm) & (eigs < lamp)]
    edge_dev = abs(eigs.max() - lamp) / lamp
    print(f"  max eigenvalue {eigs.max():.4f} vs lambda_+ {lamp:.4f} (rel dev {edge_dev:.3f})")

    # KS statistic against the theoretical CDF (continuous part + atom)
    xs = np.sort(eigs)
    cdf = mp.mp_cdf(xs, q, sigma2)
    ks = float(np.max(np.abs(cdf - (np.arange(1, xs.size + 1) - 0.5) / xs.size)))
    print(f"  KS distance vs theoretical CDF: {ks:.4f} (finite-N; decreases as N grows)")

    # -- 2. Stieltjes transform vs numerical resolvent --------------------------
    zs_real = np.array([lamp * f for f in (1.05, 1.15, 1.35, 1.8)])
    zs_cplx = np.array([2.0 + 0.8j, 3.5 + 1.2j, 5.0 + 0.5j])
    trials = 40
    g_num_r = np.zeros(zs_real.size)
    g_num_c = np.zeros(zs_cplx.size, dtype=complex)
    for _ in range(trials):
        rng = np.random.default_rng(RNG.integers(1 << 31))
        x = rng.standard_normal((n_dims, n_samples)) * np.sqrt(sigma2)
        c = x @ x.T / n_samples
        for k, z in enumerate(zs_real):
            zc = complex(z, 0.0)
            g_num_r[k] += np.trace(np.linalg.inv(zc * np.eye(n_dims) - c)).real / n_dims
        for k, z in enumerate(zs_cplx):
            g_num_c[k] += np.trace(np.linalg.inv(z * np.eye(n_dims) - c)) / n_dims
    g_num_r /= trials
    g_num_c /= trials
    g_th_r = mp.mp_resolvent_G(zs_real, q, sigma2).real
    g_th_c = mp.mp_resolvent_G(zs_cplx, q, sigma2)
    err_r = np.abs(g_num_r - g_th_r).max()
    err_c = float(np.abs(g_num_c - g_th_c).max())
    print(f"  resolvent max |error| (real axis, {trials} trials): {err_r:.2e}")
    print(f"  resolvent max |error| (complex plane): {err_c:.2e}")

    # -- 3. Silverstein residual -------------------------------------------------
    ztest = np.array([1.2 + 0.9j, 3.0 + 0.3j, 6.0 + 2.0j])
    res = np.abs(mp.silverstein_residual(ztest, q, sigma2)).max()
    print(f"  Silverstein fixed-point residual: {res:.2e}")

    # -- figure -------------------------------------------------------------------
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    banner(fig, "Marchenko-Pastur bulk from first principles (Steps 4-5)",
           "N = 900, T = 2571, q = 0.35, sigma^2 = 1.7  -  closed-form Stieltjes vs direct resolvent")

    ax = axes[0]
    ax.hist(eigs, bins=90, density=True, color=AUR[2], alpha=0.55,
            label="empirical spectrum of C = XX$^T$/T")
    ax.plot(grid, rho, color=AUR[0], lw=2.2, label=r"derived MP density $\rho(\lambda)$")
    for edge, c in [(lamm, AUR[3]), (lamp, AUR[1])]:
        ax.axvline(edge, color=c, lw=1.4, ls="--")
    ax.text(lamp + 0.06, ax.get_ylim()[1] * 0.02 + 0.02, r"$\lambda_+$", color=AUR[1])
    ax.text(lamm - 0.28, ax.get_ylim()[1] * 0.02 + 0.02, r"$\lambda_-$", color=AUR[3])
    ax.set_xlim(0, lamp * 1.3)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("density")
    ax.set_title(f"bulk density, KS = {ks:.3f}")
    ax.legend(loc="upper right")

    ax = axes[1]
    zz = np.linspace(lamp * 1.02, lamp * 2.2, 400)
    g_curve = mp.mp_resolvent_G(zz, q, sigma2).real
    ax.plot(zz, g_curve, color=AUR[0], lw=2.2, label=r"$G(z)=\frac{1}{N}\mathrm{Tr}(zI-C)^{-1}$ (derived)")
    ax.plot(zs_real, g_num_r, "o", ms=7, mfc=PANEL, mec=AUR[1], mew=1.6,
            label="direct resolvent (40 trials)")
    ax.plot(1 / zz, color=MUTED, lw=1, ls=":", label=r"$1/z$")
    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$G(z)$")
    ax.set_title(f"Stieltjes transform, max err = {err_r:.1e}")
    ax.legend(loc="upper right")

    save(fig, "mp_law_verification.png")


if __name__ == "__main__":
    main()
