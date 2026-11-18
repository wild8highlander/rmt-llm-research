<p align="center">
  <img src="../../assets/logos/gemma-rcda-avatar.svg" width="48" alt="R CDA - 5 avatar">
</p>

# <sub>🔥</sub> R CDA — 5 · the formula as a loss (Gemma3-1B-IT)

<p align="center">
  <img src="https://img.shields.io/badge/snapshot-canonical%20round-FF3B6B" alt="snapshot badge">
  <img src="https://img.shields.io/badge/pen_norm-−71.5%25-38E1C0" alt="pen badge">
  <img src="https://img.shields.io/badge/ρ_bad-0.94%20%E2%86%92%200.88-00D4FF" alt="rho badge">
  <img src="https://img.shields.io/badge/KL_control-0.002-38E1C0" alt="kl badge">
  <a href="../README.md"><img src="https://img.shields.io/badge/%E2%86%90-Gemma3--1B--IT-FFB020" alt="parent"></a>
</p>

The baseline round of the Gemma program — the direct analogue of the sibling
kit's RCDA-5: the repository's formula `(I − P_v)` baked into gradient training
as `pen = (vᵀh/‖v‖)²` on the writer-layer states (L6/L7/L8, o_proj only,
1.42% of the parameters), plus the hard abilities anchor
`β·relu(mse − cap)` on the trailing 32 logit positions. 150 Adam steps,
batch 1, one evaluator.

**Result.** The penalty melts **−71.5%** (0.964 → 0.275 normalised); ρ_bad
0.94 → **0.88** and loop_share −2.1% on the strict full protocol (16 bad
prompts × greedy 96), **−7.4%** on the 64-token mini protocol; the abilities
are untouched (**KL 0.002, PPL 45.7 → 45.6**) and the edit is rank-structured
and clean under the relative spectral gate (**pass ×3**). The honest finding:
on this stand-in the loop *bends* before it breaks — the carry analysis
([`rcda6_channel_decomposition.json`](Research/RCDA/results/rcda6_channel_decomposition.json))
explains why, and is exactly what launches RCDA-6.

## What is inside

| Path | Contents |
|------|----------|
| [`Research/RCDA/RCDA5_REPORT.md`](Research/RCDA/RCDA5_REPORT.md) | the round report — method, run, results, the carry finding |
| [`Research/RCDA/RCDA5_PLAN.md`](Research/RCDA/RCDA5_PLAN.md) | the pre-registered plan |
| [`Research/RCDA/results/`](Research/RCDA/results) | `probe_report.json` · `rcds5_vectors.npz` · `rcda5_training_log.json` · `rcda5_trained_state.pt` · `rcda5_deltas.npz` · `part5_eval_base.json` · `part5_eval_trained.json` · `ppl_check.json` · `tw_relative.json` · `rcda6_channel_decomposition.json` · `summary_all_variants.json` (the canonical summary all tables read) |
| [`Research/RCDA/figures/`](Research/RCDA/figures) | the round's figure copies |

**Start here:** [RCDA-5 report](Research/RCDA/RCDA5_REPORT.md) → then
[`../R CDA - 6/README.md`](../R%20CDA%20-%206/README.md) and
[`../R CDA - 7/README.md`](../R%20CDA%20-%207/README.md).

---

<p align="center">
  <sub><a href="../README.md">← kit root</a> · <a href="Research/RCDA/RCDA5_REPORT.md">the round-5 report →</a></sub>
</p>
