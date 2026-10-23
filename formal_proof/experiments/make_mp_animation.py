#!/usr/bin/env python3
"""Animation: empirical spectra converging to the Marchenko-Pastur law."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from _style import ANIM_DIR, AUR, INK, MUTED
from rmt_fp import mp


def main() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    q, sigma2 = 0.35, 1.7
    lamm, lamp = mp.mp_edges(q, sigma2)
    sizes = [(60, 171), (120, 343), (240, 686), (450, 1286), (750, 2143), (900, 2571)]
    frames = 42
    rng = np.random.default_rng(808)

    spectra = []
    for n, t in sizes:
        x = rng.standard_normal((n, t)) * np.sqrt(sigma2)
        spectra.append(np.linalg.eigvalsh(x @ x.T / t))

    fig, ax = plt.subplots(figsize=(7.6, 4.4), constrained_layout=True)
    fig.patch.set_facecolor("#050510")
    ax.set_facecolor("#0a0a1c")
    grid = np.linspace(0.02, lamp * 1.3, 500)
    ax.plot(grid, mp.mp_density(grid, q, sigma2), color=AUR[0], lw=2.2,
            label="derived MP density")
    for edge, c in [(lamm, AUR[3]), (lamp, AUR[1])]:
        ax.axvline(edge, color=c, lw=1.2, ls="--", alpha=0.8)
    title = ax.set_title("", color=INK, fontsize=11)
    ax.set_xlim(0, lamp * 1.3)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(r"$\lambda$", color=INK)
    ax.set_ylabel("density", color=INK)
    ax.tick_params(colors=MUTED)
    for s in ax.spines.values():
        s.set_color("#1c2340")
    ax.legend(loc="upper right")

    def update(f: int):
        for p in list(ax.patches):
            p.remove()
        idx = min(int(f / frames * len(sizes)), len(sizes) - 1)
        n, t = sizes[idx]
        ax.hist(spectra[idx], bins=70, range=(0, lamp * 1.3), density=True,
                color=AUR[2], alpha=0.55)
        title.set_text(f"N = {n},  T = {t}   (q = {q})  ->  Marchenko-Pastur")
        return []

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    ANIM_DIR.mkdir(parents=True, exist_ok=True)
    out = ANIM_DIR / "mp_convergence.gif"
    anim.save(out, writer=PillowWriter(fps=7), savefig_kwargs={"facecolor": "#050510"})
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
