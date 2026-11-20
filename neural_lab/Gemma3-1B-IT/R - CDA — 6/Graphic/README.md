# Graphic/ — the RCDA-6 figures

Mirrored from [`../Research/RCDA/figures/`](../Research/RCDA/figures/)
(regenerate with `python3 Research/RCDA/tools/rcda6_figures.py`). All values
are the measured record of this experiment, not illustrations.

| Figure | What it shows |
|--------|----------------|
| `figC1_probe_phase.png` | the phase probe: z_early vs z_deep per layer (the deep phase separates 2.5–3.5× better) and cos(v_early, v_deep) — why a single collapse-point direction was not enough |
| `figC2_band.png` | the trainable set: the collapse-probe z_diff profile, the L8–L15 band and the 16 edited tensors vs RCDA-5's 3 |
| `figC3_training.png` | the 150-step record: pen_norm collapse, the anchor riding the hard cap, the deep-phase share of the penalty positions |
| `figC4_eval_arms.png` | the full benchmark across four states: base / RCDA-5 SGD / RCDA-5 Adam / RCDA-6 (ρ_bad, loop share, loop waste) |
| `figC5_migration.png` | per-prompt loop-share migration base → RCDA-6: hard loopers relieve (2 fully repaired), mild prompts break — the defect migrates |
| `figC6_spectral.png` | the spectral record of the 16-tensor edit: rel_Frobenius, TW z_resid, cos(u₁, v_early/v_deep) |

Parent experiment: [`../Research/RCDA/RCDA6_REPORT.md`](../Research/RCDA/RCDA6_REPORT.md).
