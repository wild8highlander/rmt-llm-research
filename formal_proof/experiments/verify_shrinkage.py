#!/usr/bin/env python3
"""Experiment 4 -- Optimal spectral filtering (Step 14).

Compares, on the spiked-covariance model, the Frobenius relative error of
estimating the *population* covariance from the sample spectrum:

  * oracle-rank hard truncation   (Eckart-Young: exact for the stated
    rank-constrained problem of approximating the EMPIRICAL matrix);
  * hard MP-edge threshold        (rank-free clip at lambda_+);
  * Gavish-Donoho optimal hard threshold (oracle-risk optimal for
    singular values; the 4/sqrt(3) anchor at q = 1 is asserted);
  * exact inverse-BBP soft shrinker (derived in chapter 6/10);
  * its first-order expansion     lambda - sigma^2 (1+q) - q sigma^4/lambda.

The experiment demonstrates the honest hierarchy: the exact inverse-BBP
shrinker dominates its first-order expansion, and no single operator is
uniformly optimal across all spike strengths -- the choice of operator
depends on which problem is being solved.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from _style import AUR, INK, MUTED, PANEL, banner, save
from rmt_fp import mp, shrinkage as sh, spiked

TRIALS = 25
N_DIMS, N_SAMPLES = 300, 600


def sample_population_and_spectrum(mu: float, q: float, sigma2: float,
                                   rng: np.random.Generator):
    n, t = N_DIMS, N_SAMPLES
    z = rng.standard_normal((n, t))
    v = rng.standard_normal(n)
    v /= np.linalg.norm(v)
    root = np.sqrt(sigma2 + mu) - np.sqrt(sigma2)
    x = np.sqrt(sigma2) * z + root * np.outer(v, z.T @ v)
    c_emp = x @ x.T / t
    # population spectrum: sigma^2 (multiplicity n-1) and sigma^2 + mu
    pop = np.full(n, sigma2)
    pop[0] = sigma2 + mu
    return np.linalg.eigvalsh(c_emp), pop


def main() -> None:
    q, sigma2 = 0.5, 1.0
    mu_c = spiked.bbp_threshold(q, sigma2)
    lamp = mp.mp_edges(q, sigma2)[1]
    tau_gd = sh.gavish_donoho_tau(1.0)
    assert abs(tau_gd - 4.0 / np.sqrt(3.0)) < 1e-12, "4/sqrt(3) anchor failed"
    print(f"Shrinkage check: q={q}, mu_c={mu_c:.4f}, lambda_+={lamp:.4f}")
    print(f"  Gavish-Donoho anchor: tau*(1) = {tau_gd:.12f} = 4/sqrt(3) OK")

    mus = np.array([0.6, 1.0, 1.5, 2.5, 4.0, 7.0]) * mu_c
    # Panel A: target = full population covariance Sigma = sigma2 I + mu v v^T
    methods_a = {
        "oracle rank-1 truncation": lambda e: sh.hard_truncation(e, 1),
        "hard MP-edge clip": lambda e: sh.hard_mp_threshold(e, q, sigma2),
        "inverse-BBP soft (exact)": lambda e: sh.inverse_bbp_shrinker(e, q, sigma2),
        "inverse-BBP soft (1st order)": lambda e: sh.first_order_inverse_shrinker(e, q, sigma2),
    }
    # Panel B: target = spike-only signal P (bulk removed -- the Gavish-Donoho
    # problem setting).  The de-biased operator estimates sigma^2 + mu_hat above
    # the edge and zeroes the bulk (P has no bulk); hard operators keep the raw
    # (biased) sample eigenvalue.
    lamp_q = mp.mp_edges(q, sigma2)[1]
    methods_b = {
        "oracle rank-1 truncation": lambda e: sh.hard_truncation(e, 1),
        "hard MP-edge clip (raw lambda)": lambda e: sh.hard_mp_threshold(e, q, sigma2),
        "Gavish-Donoho hard": lambda e: sh.gavish_donoho_threshold(e, q, sigma2),
        "de-biased soft (sigma^2 + mu_hat, bulk zero)":
            lambda e: np.where(e > lamp_q, sh.inverse_bbp_shrinker(e, q, sigma2), 0.0),
    }
    errors_a = {name: np.zeros(len(mus)) for name in methods_a}
    errors_b = {name: np.zeros(len(mus)) for name in methods_b}
    rng = np.random.default_rng(505)
    for i, mu in enumerate(mus):
        for _ in range(TRIALS):
            eigs, pop = sample_population_and_spectrum(float(mu), q, sigma2, rng)
            spike_only = np.zeros_like(pop)
            spike_only[0] = pop[0]
            for name, fn in methods_a.items():
                errors_a[name][i] += sh.frobenius_relative_error(fn(eigs), pop)
            for name, fn in methods_b.items():
                errors_b[name][i] += sh.frobenius_relative_error(fn(eigs), spike_only)
        for name in methods_a:
            errors_a[name][i] /= TRIALS
        for name in methods_b:
            errors_b[name][i] /= TRIALS
    print("  relative errors, panel A (target = full Sigma):")
    for name in methods_a:
        print(f"    {name:30s}: " + " ".join(f"{v:.3f}" for v in errors_a[name]))
    print("  relative errors, panel B (target = spike-only P):")
    for name in methods_b:
        print(f"    {name:30s}: " + " ".join(f"{v:.3f}" for v in errors_b[name]))

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), constrained_layout=True)
    banner(fig, "Spectral filtering operators under the Frobenius loss (Step 14)",
           f"N = {N_DIMS}, T = {N_SAMPLES}, q = 0.5, sigma^2 = 1  -  "
           f"{TRIALS} trials per point; two problems, two exact solutions")

    ax = axes[0]
    for (name, err), c in zip(errors_a.items(), [AUR[4], AUR[2], AUR[0], MUTED]):
        ax.plot(mus / mu_c, err, "o-", ms=5, lw=1.8, color=c, label=name)
    ax.axvline(1.0, color=AUR[1], lw=1.2, ls="--")
    ax.text(1.04, 0.9, r"$\mu_c$", color=AUR[1], transform=ax.get_xaxis_transform())
    ax.set_xlabel(r"spike strength $\mu/\mu_c$")
    ax.set_ylabel("relative Frobenius error")
    ax.set_yscale("log")
    ax.set_title("target = full population covariance $\\Sigma$\n"
                 "(inverse-BBP soft shrinker: exact inverse of Step 8)")
    ax.legend(loc="center left", fontsize=8)

    ax = axes[1]
    for (name, err), c in zip(errors_b.items(), [AUR[4], AUR[2], AUR[3], AUR[0]]):
        ax.plot(mus / mu_c, err, "o-", ms=5, lw=1.8, color=c, label=name)
    ax.axvline(1.0, color=AUR[1], lw=1.2, ls="--")
    ax.text(1.04, 0.9, r"$\mu_c$", color=AUR[1], transform=ax.get_xaxis_transform())
    ax.set_xlabel(r"spike strength $\mu/\mu_c$")
    ax.set_ylabel("relative Frobenius error")
    ax.set_yscale("log")
    ax.set_title("target = spike-only signal $P$ (Gavish-Donoho setting)\n"
                 "(hard operators keep the biased raw eigenvalue)")
    ax.legend(loc="center left", fontsize=8)

    save(fig, "shrinkage_comparison.png")


if __name__ == "__main__":
    main()
