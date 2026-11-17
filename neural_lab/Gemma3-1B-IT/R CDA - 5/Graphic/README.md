# <sub>📊</sub> Graphic — the figures of the Gemma3-1B-IT RCDA-5 transfer test

All figures are re-renders of this folder's `results/` JSONs (dark aurora
style, mirrors the repository's figure language). Source renders live in
[`../Research/RCDA/figures/`](../Research/RCDA/figures/) and are copied here.

| Figure | What it shows |
|--------|---------------|
| [`figG1_probe_layers.png`](figG1_probe_layers.png) | the per-layer behavioral separation profile (z_diff, loop vs pre-loop states): a clean unimodal writer band **L10–L12** (z_diff 205–220) — the analog of Pocket's L15/16/18 |
| [`fig15_gemma_training.png`](fig15_gemma_training.png) | training curves of the SGD arm: `pen_raw` (violet) and the dimensionless `pen_norm` (cyan, 1.0 → 0.0084), ρ_mini diamonds at steps 0/75/150, and the abilities anchor MSE (amber) against its hard cap 1.0 (crimson) |
| [`figG2_spectral_purity.png`](figG2_spectral_purity.png) | singular spectrum of the edit ΔW per writer layer vs the MP edge of the v-orthogonal residual bulk — why the TW verdict is **violation ×3** (the MNN SGD edit passed ×3; this is the transfer failure, visible) |
| [`figG3_gemma_eval.png`](figG3_gemma_eval.png) | the three-arm summary: ρ_bad 0.60 → 1.00 (SGD) / 0.95 (Adam), loop waste 147 → 180 / 178 tokens, KL 0.22 / 0.02, PPL parity 10.9 → 10.2 / 11.0, and cos(u₁, v) per writer layer |

Re-render: `python3 Research/RCDA/tools/rcda5_figures.py --work <GEMMA_WORK>
--results <GEMMA_WORK>/rcda5_final --graphic Graphic/`.
