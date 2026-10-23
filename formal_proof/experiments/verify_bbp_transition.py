#!/usr/bin/env python3
"""Experiment 2 -- BBP phase transition: threshold, outliers, overlaps.

Verifies (Steps 7-8) on the spiked-covariance ensemble

    population  Sigma = sigma^2 I + mu v v^T,
    sample      C = (1/T) Sigma^{1/2} Z Z^T Sigma^{1/2},  q = N/T:

  1. the largest sample eigenvalue sticks to the bulk edge lambda_+ while
     mu <= mu_c = sigma^2 sqrt(q) and detaches onto the trajectory
     lambda_out(mu) = (sigma^2 + mu)(1 + q sigma^2 / mu) above mu_c;
  2. the squared eigenvector overlap follows
     |<v_hat, v>|^2 = (1 - q sigma^4/mu^2)/(1 + q sigma^2/mu);
  3. an independent numeric root of the spiked-resolvent pole equation
     coincides with the closed-form outlier formula.

Also renders the BBP-detachment animation (assets/animations).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from _style import ANIM_DIR, AUR, INK, MUTED, PANEL, banner, save
from rmt_fp import mp, spiked

TRIALS = 60
N_DIMS, N_SAMPLES = 400, 1000


def sample_spiked(mu: float, q: float, sigma2: float, rng: np.random.Generator,
                  n_dims: int = N_DIMS, n_samples: int = N_SAMPLES):
    """One draw of the spiked-covariance model; returns top eigenpair + overlap."""
    z = rng.standard_normal((n_dims, n_samples))
    v = rng.standard_normal(n_dims)
    v /= np.linalg.norm(v)
    root = np.sqrt(sigma2 + mu) - np.sqrt(sigma2)
    x = np.sqrt(sigma2) * z + root * np.outer(v, z.T @ v)
    c = x @ x.T / n_samples
    w, vecs = np.linalg.eigh(c)
    return w[-1], float(abs(vecs[:, -1] @ v) ** 2)


def main() -> None:
    q, sigma2 = 0.4, 1.0
    mu_c = spiked.bbp_threshold(q, sigma2)
    lamm, lamp = mp.mp_edges(q, sigma2)
    print(f"BBP check: q={q}, sigma2={sigma2}, mu_c={mu_c:.4f}, lambda_+={lamp:.4f}")

    mus = np.linspace(0.15 * mu_c, 3.2 * mu_c, 13)
    emp_lmax = np.zeros_like(mus)
    emp_ov = np.zeros_like(mus)
    emp_std = np.zeros_like(mus)
    rng = np.random.default_rng(202)
    for i, mu in enumerate(mus):
        vals, ovs = zip(*(sample_spiked(float(mu), q, sigma2, rng) for _ in range(TRIALS)))
        emp_lmax[i] = np.mean(vals)
        emp_ov[i] = np.mean(ovs)
        emp_std[i] = np.std(vals)
        th_l = spiked.bbp_outlier(mu, q, sigma2)
        th_o = spiked.bbp_overlap(mu, q, sigma2)
        print(f"  mu={mu:.3f}: lmax={emp_lmax[i]:.4f} (th {th_l:.4f})  "
              f"overlap={emp_ov[i]:.4f} (th {th_o:.4f})")

    # pole-equation cross-check
    pole_err = max(
        abs(spiked.bbp_outlier_pole_equation(float(m), q, sigma2)
            - float(spiked.bbp_outlier(m, q, sigma2)))
        for m in mus[mus > mu_c]
    )
    print(f"  pole-equation vs closed form: max |diff| = {pole_err:.2e}")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    banner(fig, "Baik-Ben Arous-Peche transition from first principles (Steps 7-8)",
           f"N = {N_DIMS}, T = {N_SAMPLES}, q = 0.4, sigma^2 = 1  -  {TRIALS} trials per point")

    ax = axes[0]
    mu_grid = np.linspace(0, mus[-1], 400)
    ax.plot(mu_grid, spiked.bbp_outlier(mu_grid, q, sigma2), color=AUR[0], lw=2.2,
            label=r"derived $\lambda_{out}(\mu)$")
    ax.plot(mu_grid, np.full_like(mu_grid, lamp), color=MUTED, lw=1, ls=":",
            label=r"bulk edge $\lambda_+$")
    ax.errorbar(mus, emp_lmax, yerr=emp_std, fmt="o", ms=6, mfc=PANEL, mec=AUR[1],
                mew=1.5, capsize=2, label="empirical")
    ax.axvline(mu_c, color=AUR[3], lw=1.4, ls="--")
    ax.text(mu_c * 1.04, lamp - 0.45, r"$\mu_c=\sigma^2\sqrt{q}$", color=AUR[3])
    ax.set_xlabel(r"spike strength $\mu$")
    ax.set_ylabel(r"largest eigenvalue")
    ax.set_title("spectral transition at " r"$\mu_c$")
    ax.legend(loc="upper left")

    ax = axes[1]
    ax.plot(mu_grid[mu_grid > mu_c], spiked.bbp_overlap(mu_grid[mu_grid > mu_c], q, sigma2),
            color=AUR[0], lw=2.2, label=r"derived $|\langle\hat v, v\rangle|^2$")
    ax.plot(mus, emp_ov, "o", ms=6, mfc=PANEL, mec=AUR[2], mew=1.5, label="empirical")
    ax.axvline(mu_c, color=AUR[3], lw=1.4, ls="--")
    ax.set_xlabel(r"spike strength $\mu$")
    ax.set_ylabel("squared overlap")
    ax.set_title("eigenvector alignment")
    ax.legend(loc="upper left")

    save(fig, "bbp_transition_verification.png")
    make_animation(q, sigma2, mu_c, lamp)


def make_animation(q: float, sigma2: float, mu_c: float, lamp: float) -> None:
    """GIF: the outlier detaching from the bulk as mu crosses mu_c."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    frames = 48
    mus = mu_c * np.linspace(0.35, 2.6, frames)
    rng = np.random.default_rng(303)
    n_dims, n_samples = 220, 550

    fig, ax = plt.subplots(figsize=(7.4, 4.3), constrained_layout=True)
    fig.patch.set_facecolor("#050510")
    ax.set_facecolor("#0a0a1c")

    hist_art = {}
    text = ax.text(0.02, 0.95, "", transform=ax.transAxes, color=INK, fontsize=11,
                   verticalalignment="top")

    def update(f: int):
        mu = float(mus[f])
        vals = np.array([sample_spiked(mu, q, sigma2, rng, n_dims, n_samples)[0]
                         for _ in range(24)])
        for art in hist_art.values():
            art.remove()
        hist_art.clear()
        n_b, bins, patches = ax.hist(vals, bins=60, range=(0, lamp * 2.6), density=True,
                                     color=AUR[2], alpha=0.65)
        hist_art["patches"] = patches
        grid = np.linspace(0.02, lamp * 2.6, 500)
        hist_art["mp"], = ax.plot(grid, mp.mp_density(grid, q, sigma2), color=MUTED,
                                  lw=1.2, ls="--")
        lo = float(spiked.bbp_outlier(mu, q, sigma2))
        hist_art["out"], = ax.plot([lo], [0.02], "v", ms=11, color=AUR[3])
        hist_art["edge"] = ax.axvline(lamp, color=AUR[1], lw=1.1, ls=":")
        status = "DETACHED" if mu > mu_c else "in bulk"
        text.set_text(rf"$\mu = {mu:.2f}$   $\mu_c = {mu_c:.2f}$" f"   ->  {status}")
        return []

    ax.set_xlim(0, lamp * 2.6)
    ax.set_ylim(0, 1.15 / np.sqrt(lamp))
    ax.set_xlabel(r"$\lambda$", color=INK)
    ax.set_ylabel("density", color=INK)
    ax.tick_params(colors=MUTED)
    for s in ax.spines.values():
        s.set_color("#1c2340")
    title = ax.set_title("BBP transition: the semantic spike leaves the bulk (Step 7)",
                         color=INK, fontsize=11)

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    ANIM_DIR.mkdir(parents=True, exist_ok=True)
    out = ANIM_DIR / "bbp_transition.gif"
    anim.save(out, writer=PillowWriter(fps=8), savefig_kwargs={"facecolor": "#050510"})
    plt.close(fig)
    print(f"  saved {out.relative_to(ANIM_DIR.parents[2])}")


if __name__ == "__main__":
    main()
