<p align="center">
  <img src="../../assets/logos/gemma-rcda-avatar.svg" width="48" alt="R CDA - 6 avatar">
</p>

# <sub>🧩</sub> R CDA — 6 · extend the trainable tensors to down_proj

<p align="center">
  <img src="https://img.shields.io/badge/snapshot-round%206-FFB020" alt="snapshot badge">
  <img src="https://img.shields.io/badge/trainable-o_proj%20%2B%20down_proj%20%C3%97%20L6%2F7%2F8-FFB020" alt="tensors badge">
  <img src="https://img.shields.io/badge/MLP%20write%20share-68.6%25-00D4FF" alt="share badge">
  <img src="https://img.shields.io/badge/cos(u1%2Cv)-0.74%E2%80%930.89-B026FF" alt="cos badge">
  <a href="../README.md"><img src="https://img.shields.io/badge/%E2%86%90-Gemma3--1B--IT-FFB020" alt="parent"></a>
</p>

The round the sibling kit's report pre-registered: *"the residual ρ_bad is not
zero … the RCDA-6 hypothesis is to add receiver layers (down the stack) to the
penalty or extend the trainable tensors to down_proj"*. This folder runs the
second half of the hypothesis: the trainable set doubles
(`o_proj + down_proj` of the three writers, 7.11% of the parameters,
drift-matched lr 2e-5), the penalty reads the same writer-layer states, and a
dedicated **channel decomposition** first proves the premise — on the stand-in
the loop is *carried* by the residual stream (~500–700 v-units) and the MLP
channel writes **68.6%** of it (83% at deep-phase L8), so an o_proj-only
penalty structurally under-treats the defect.

**Result.** The penalty descends **−62.6%**, loop metrics land between the
base and RCDA-5 at the matched drift budget (loop_share 0.8815, ρ_bad 0.9375),
abilities hold (KL 0.0094, PPL 46.5), the relative TW gate passes ×6 — and the
scientific payoff: the trained edits align with the measured behavioural
direction, **|cos(u₁, v)| = 0.74–0.89**. The sibling kit's core claim — *SGD
converges to a spectrally clean edit pointing along v* — replicates on a
second architecture.

## What is inside

| Path | Contents |
|------|----------|
| [`Research/RCDA/RCDA6_REPORT.md`](Research/RCDA/RCDA6_REPORT.md) | the round report incl. the decomposition argument |
| [`Research/RCDA/RCDA6_PLAN.md`](Research/RCDA/RCDA6_PLAN.md) | the pre-registered plan |
| [`Research/RCDA/results/`](Research/RCDA/results) | `rcda6_training_log.json` · `rcda6_trained_state.pt` · `rcda6_deltas.npz` · `part6_eval_trained.json` · `ppl_check.json` · `tw_relative.json` |
| [`Research/RCDA/figures/`](Research/RCDA/figures) | the round's figure copies |

Run (shared toolkit lives in [`../Code/`](../Code/README.md)):
`python ../Code/gemma_rcds_train.py --variant rcda6 --model google/gemma-3-1b-it --steps 150 --lr 2e-5 --anchor_scope good --refresh_every 32`

---

<p align="center">
  <sub><a href="../README.md">← kit root</a> · <a href="Research/RCDA/RCDA6_REPORT.md">the round-6 report →</a> · <a href="../R%20CDA%20-%207/README.md">R CDA - 7 →</a></sub>
</p>
