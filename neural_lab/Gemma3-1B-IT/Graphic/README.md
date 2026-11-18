# 🎨 Graphic — the result gallery

<p align="center">
  <img src="https://img.shields.io/badge/figures-fig05%E2%80%93fig16-00D4FF" alt="figures badge">
  <img src="https://img.shields.io/badge/animations-2%20GIFs-B026FF" alt="anims badge">
  <img src="https://img.shields.io/badge/regenerate-one%20command-38E1C0" alt="regen badge">
  <a href="../README.md"><img src="https://img.shields.io/badge/%E2%86%90-Gemma3--1B--IT-FFB020" alt="parent"></a>
</p>

Every image in this folder is rendered from the committed result JSONs by
[`make_graphic.py`](make_graphic.py) — the same rule as the rest of the
repository: **nothing is hand-typed**. Palette is the project identity:
near-black field, cyan MP bulk, amber BBP spike, violet structure, crimson
defect.

| File | What it shows |
|------|---------------|
| [`fig05_probe_zdiff_profile.png`](fig05_probe_zdiff_profile.png) | per-layer z_diff; writers L6/7/8 in crimson; θ_v vs θ_c overlay |
| [`fig06_matrix_scan.png`](fig06_matrix_scan.png) | λ_v/λ₊ per o_proj — the distributed-defect verdict |
| [`fig07_channel_decomposition.png`](fig07_channel_decomposition.png) | carried vs MLP-write vs attention-write v-energy (log scale) |
| [`fig08_training_pen_descent.png`](fig08_training_pen_descent.png) | pen_norm over 150 steps, all variants, smoothed + raw |
| [`fig09_rho_mini_melt.png`](fig09_rho_mini_melt.png) | mini-benchmark loop_share trajectories |
| [`fig10_headline_defect.png`](fig10_headline_defect.png) | loop_share / waste / ρ_bad across the four arms |
| [`fig11_abilities_preserved.png`](fig11_abilities_preserved.png) | KL, PPL and penalty-descent bars |
| [`fig12_cosine_alignment.png`](fig12_cosine_alignment.png) | cos(u₁ of the edit, v) per variant per tensor |
| [`fig13_spectral_purity.png`](fig13_spectral_purity.png) | relative spectral gate: bulk KS of edited vs base matrices |
| [`fig14_leverage_chart.png`](fig14_leverage_chart.png) | descent % vs KL — the efficiency frontier of the three levers |
| [`fig15_ppl_parity.png`](fig15_ppl_parity.png) | PPL parity, per control text and mean |
| [`fig16_overview_dashboard.png`](fig16_overview_dashboard.png) | the one-glance dashboard |
| [`anims/training-loop.gif`](anims/training-loop.gif) | 30-frame animation of the penalty melting (122 KB) |
| [`anims/defect-melt.gif`](anims/defect-melt.gif) | the defect state morphing across the arms (25 KB) |

```bash
python make_graphic.py     # rebuilds all of the above from results/*.json
```

Image budgets follow the repository rules: PNG ≤ 400 KB, GIF ≤ 2 MB — the
largest file here is 166 KB.

---

<p align="center">
  <sub><a href="../README.md">← kit root</a></sub>
</p>
