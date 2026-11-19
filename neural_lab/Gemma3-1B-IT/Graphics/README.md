# 📊 Graphics — the real-Gemma round figures

<p align="center">
  <img src="https://img.shields.io/badge/figures-12-00D4FF" alt="figures badge">
  <img src="https://img.shields.io/badge/rounds-RCDA--5%20%C2%B7%20RCDA--6%20(real%20model)-B026FF" alt="rounds badge">
  <img src="https://img.shields.io/badge/source-%60JSON%20Results%2F%60%20JSONs-38E1C0" alt="source badge">
  <a href="../LlamaFactory/figures/"><img src="https://img.shields.io/badge/next%20gallery-LlamaFactory%2Ffigures-FFB020" alt="next badge"></a>
  <a href="../README.md"><img src="https://img.shields.io/badge/%E2%86%90-Gemma3--1B--IT-FFB020" alt="parent"></a>
</p>

The rendered record of the real-model rounds — every figure is drawn from
the committed JSONs of [`../JSON Results/`](../JSON%20Results/README.md)
by the `rcda*_figures.py` renderers of [`../Script's/`](../Script's/README.md).
Project palette: near-black field, cyan MP bulk, amber BBP spike, violet
structure, crimson defect.

| File | What it shows |
|------|---------------|
| [`figG1_probe_layers.png`](figG1_probe_layers.png) | per-layer z_diff of the real-model probe — the writer band L10–12 (RCDA-5) |
| [`figG2_spectral_purity.png`](figG2_spectral_purity.png) | TW spectral control of the RCDA-5 edit — violation ×3, structured bulk |
| [`figG3_gemma_eval.png`](figG3_gemma_eval.png) | the RCDA-5 arms vs base: ρ_bad, loop_share, waste — the honest negative |
| [`fig15_gemma_training.png`](fig15_gemma_training.png) | training dynamics: pen_norm and anchor MSE over 150 steps |
| [`figC1_probe_phase.png`](figC1_probe_phase.png) | the phase probe: early vs deep separability per layer |
| [`figC2_band.png`](figC2_band.png) | the RCDA-6 writer band (o_proj + down_proj, L8–L15) |
| [`figC3_training.png`](figC3_training.png) | RCDA-6 phase-matched penalty descent |
| [`figC4_eval_arms.png`](figC4_eval_arms.png) | RCDA-6 arm vs base vs RCDA-5 — the first partial repair |
| [`figC5_migration.png`](figC5_migration.png) | **the migration figure**: hard-12 improves while mild-8 breaks |
| `*_v5.png` / `*_v6.png` | the per-round generations of the same figures (RCDA-5 / RCDA-6 captures) |

The successor gallery — the RCDA-8 LlamaFactory round (formula arm vs
data arm, 9 figures + 6 animations from the real model) — lives at
[`../LlamaFactory/figures/`](../LlamaFactory/README.md).
