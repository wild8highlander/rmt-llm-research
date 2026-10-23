#!/usr/bin/env python3
"""Experiment 5 -- Entropy response of the decoder to MP-bulk noise (Step 11).

Verifies the derived second-order theorem (chapter 8): for near-uniform
softmax baselines and logit noise  eps = W x_noise / tau  with
x_noise ~ N(0, Sigma_noise),

    E[H(softmax(z0 + eps))] = H0 - (1/2) Tr( (diag(p0) - p0 p0^T) A )
                              + O(tau^{-3}),   A = W Sigma W^T / tau^2,

which at the exact uniform baseline with isotropic decoder rows reduces to

    E[H] = H0 - (1 - 1/K) s2bar / (2 tau^2),   s2bar = (1/K) Tr(W Sigma W^T).

The Monte-Carlo checker sweeps the temperature and overlays the derived
curves.  The mixture entropy H(E[p]) is plotted for contrast: it grows
toward log K, illustrating that the sign of the correction depends on
which entropy is meant -- the plan's phenomenological Step 11 formula
corresponds to neither reading exactly and is therefore not claimed here.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from _style import AUR, INK, MUTED, PANEL, banner, save
from rmt_fp import entropy as ent
from rmt_fp import mp

K_VOCAB = 64
N_DIMS = 200
N_NOISE = 400


def mp_noise_covariance(n_dims: int, n_samples: int, sigma2: float,
                        rng: np.random.Generator) -> np.ndarray:
    """A covariance matrix whose spectrum follows the MP bulk."""
    x = rng.standard_normal((n_dims, n_samples)) * np.sqrt(sigma2)
    c = x @ x.T / n_samples
    w, v = np.linalg.eigh(c)
    w = np.maximum(w, 1e-12)
    return (v * w) @ v.T


def main() -> None:
    rng = np.random.default_rng(606)
    q = N_DIMS / N_NOISE
    sigma2_eff = 1.0 / (1.0 - q / 2.0)  # normalize mean eigenvalue to ~1
    sigma = mp_noise_covariance(N_DIMS, N_NOISE, sigma2_eff, rng)
    w_rows = rng.standard_normal((K_VOCAB, N_DIMS)) / np.sqrt(N_DIMS)
    z0 = np.zeros(K_VOCAB)
    h0 = np.log(K_VOCAB)
    s2bar = float(np.trace(w_rows @ sigma @ w_rows.T) / K_VOCAB)
    pred_iso = ent.entropy_response_isotropic(w_rows, sigma, 1.0)  # per 1/tau^2 unit
    print(f"Entropy check: K={K_VOCAB}, N={N_DIMS}, q={q:.3f}, s2bar={s2bar:.4f}")
    print(f"  derived slope (isotropic form): -delta H per 1/tau^2 = "
          f"{-pred_iso:.5f} = (1-1/K) s2bar / 2")

    taus = np.linspace(1.6, 6.0, 9)
    mc_h, mc_mix = [], []
    n_trials = 6000
    for tau in taus:
        mc_h.append(ent.mc_conditional_entropy(z0, w_rows, sigma, tau, n_trials, rng))
        mc_mix.append(ent.mc_mixture_entropy(z0, w_rows, sigma, tau, 12000, rng))
        print(f"  tau={tau:.2f}: E[H]={mc_h[-1]:.4f}  H(mixture)={mc_mix[-1]:.4f}")

    x_slope = 1.0 / taus**2
    A_full = w_rows @ sigma @ w_rows.T
    pred_exact = np.array([
        h0 + ent.entropy_logit_response(np.full(K_VOCAB, 1.0 / K_VOCAB), A_full / tau**2)
        for tau in taus
    ])

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.6, 4.6), constrained_layout=True)
    banner(fig, "Second-order entropy response to bulk noise (Step 11, derived)",
           f"K = {K_VOCAB} tokens, N = {N_DIMS}, MP bulk with q = {q:.2f}; "
           f"{n_trials} MC trials per point")
    ax.plot(x_slope, mc_h, "o", ms=6, mfc=PANEL, mec=AUR[0], mew=1.6,
            label=r"MC: $E[H(\mathrm{softmax}(z_0 + W\xi/\tau))]$")
    ax.plot(x_slope, pred_exact, color=AUR[0], lw=2.0,
            label=r"derived: $H_0 - \frac{1}{2}\mathrm{Tr}[(\mathrm{diag}\,p - pp^T)A]$")
    ax.plot(x_slope, h0 + x_slope * pred_iso, color=MUTED, lw=1.2, ls="--",
            label=r"isotropic closed form: $H_0 - (1-1/K)\,\bar\sigma^2/2\tau^2$")
    ax.plot(x_slope, mc_mix, "s", ms=5, mfc=PANEL, mec=AUR[3], mew=1.5,
            label=r"MC: mixture entropy $H(E[p])$ (nearly flat, $\mathcal{O}(\tau^{-4})$)")
    ax.axhline(h0, color=INK, lw=0.8, ls=":")
    ax.text(0.30, h0 + 0.004, r"$H_0 = \log K$", color=INK, fontsize=9)
    ax.set_xlabel(r"$1/\tau^2$")
    ax.set_ylabel("entropy (nats)")
    ax.set_title("conditional entropy decreases at second order;\n"
                 "the mixture entropy is flat at this order -- the reading matters")
    ax.legend(loc="upper left", fontsize=8.5)
    save(fig, "entropy_response.png")


if __name__ == "__main__":
    main()
