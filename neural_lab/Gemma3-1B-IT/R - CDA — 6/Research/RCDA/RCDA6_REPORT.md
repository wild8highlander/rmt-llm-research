# RCDA-6: extend the trainable tensors to down_proj — the hypothesis round

Written: November 18, 2026. The round the sibling kit's RCDA-5 report
pre-registered: *"the residual ρ_bad is not zero … the RCDA-6 hypothesis is to
add receiver layers (down the stack) to the penalty or extend the trainable
tensors to down_proj."* This folder runs the second half of the hypothesis and
first *proves the premise* with a dedicated measurement.

## 1. The premise, measured: the loop is carried and MLP-written

Before training, the write channels are decomposed at the three writers
(`rcda6_channel_decomposition.py`, one forward pass per bad prompt with hooks
on `o_proj`, `down_proj` and the block outputs):

| layer | attention o_proj write | MLP down_proj write | carried by the residual stream |
| --- | --- | --- | --- |
| L6 | 2.24 | 2.79 | 538.8 |
| L7 | 2.21 | 2.94 | 740.6 |
| L8 | 1.53 | 7.32 | 565.6 |
| **Σ** | **5.98** | **13.05** | ~1800 |

Readings:

- **MLP share of the write energy: 68.6%** (83% at deep-phase L8). An
  o_proj-only penalty structurally under-treats the defect — the RCDA-5
  result (−71.5% penalty, only a bend in behaviour) is what that ceiling looks
  like from inside.
- The carried v-energy is ~100× a single write. The defect lives in the
  residual stream; the writers shape it, they do not own it. This is the
  quantitative restatement of the sibling kit's "distributed defect" verdict,
  and it turns the RCDA-6 hypothesis from a guess into a measured target.

## 2. Setup

Everything except the named levers is RCDA-5's protocol (same data, same loss
form, same evaluator, same 150 Adam steps):

- **Trainable set (the lever):** `o_proj + down_proj` of L6/L7/L8 — 6 tensors,
  8,847,360 parameters = **7.11%** of the model;
- **drift-matched lr:** 2e-5 (8.85M params × 2e-5 ≈ 1.77M × 1e-4 — the
  per-step displacement budget is matched so the tensor-set comparison stays
  clean);
- **anchor:** β=10, cap=1.0, scope `good` (unchanged);
- **penalty points:** the same writer states as RCDA-5 (unchanged) — only the
  actuator set moves.

## 3. Training and results

| bench | ρ_mini | mean loop_share |
| --- | --- | --- |
| step 0 | 0.875 | 0.865 |
| step 75 | 0.875 | 0.824 |
| step 150 | 0.875 | **0.824 (−4.7%)** |

pen_norm **0.972 → 0.364 (−62.6%)**. The final measurement:

| metric | base | RCDA-5 | **RCDA-6** |
| --- | --- | --- | --- |
| ρ_bad | 0.94 | 0.88 | 0.9375 |
| mean loop_share (bad) | 0.894 | **0.876** | 0.8815 |
| loop_waste mean | 85.9 | **84.1** | 84.6 |
| KL(base‖arm), control | 0 | **0.0021** | 0.0094 |
| PPL control | 45.7 | **45.6** | 46.5 |
| TW (relative gate) | — | pass ×3 | **pass ×6** |
| cos(u₁ of the edit, v) | — | ≤0.09 | **0.74–0.89** |

Three conclusions.

1. **At the matched drift budget the extended actuator does not beat the
   focused one on behaviour.** RCDA-5's loop metrics remain the best of the
   program at 96 tokens. The honest reading: doubling the actuator set at a
   per-tensor halved lr spreads the displacement; the behavioural melt is
   budget-bound, not actuator-bound. The premise is true (the MLP channel
   *does* carry the writes) but the carry itself is ~100× the writes —
   closing the write channels does not close the carry within a 150-step
   drift budget.
2. **The spectral conclusion of the sibling kit replicates.** The trained
   edits align with the measured behavioural direction:
   **|cos(u₁, v)| = 0.743 / 0.851 / 0.891** across the writer layers — versus
   ≤0.09 for RCDA-5's single-channel edit. *The model finds a spectrally
   clean edit of the same class on its own* — now shown on a second
   architecture and a doubled tensor set, with the relative TW gate passing
   on all six tensors (bulk KS ratio ≤ 1.001).
3. **The abilities stay held** (KL 0.0094, PPL +1.8%) — the tightened
   discipline carries over even with 4× the trainable mass.

## 4. What this round sets up

The efficiency frontier ([`fig14`](../../../Graphics/fig14_leverage_chart.png))
shows the descent bought per unit of drift. The next lever named by the
program — **RCDA-7: deep-phase-only penalty + tightened anchor** —
concentrates the same actuator set on the single point where the MLP share
peaks (L8, 83%) and halves the tolerated drift. See
[`../R CDA - 7/Research/RCDA/RCDA7_REPORT.md`](../../../R%20-%20CDA%20%E2%80%94%207/Research/RCDA/RCDA7_REPORT.md).

## 5. Reproduction and artifacts

```bash
python Code/rcda6_channel_decomposition.py        # the premise, measured
python Code/gemma_rcds_train.py --variant rcda6 --steps 150 --lr 2e-5 \
    --anchor_scope good --refresh_every 32
python Code/gemma_benchmark.py --round 6 --tag trained \
    --state "R CDA - 6/Research/RCDA/results/rcda6_trained_state.pt"
```

Artifacts: `rcda6_training_log.json`, `rcda6_trained_state.pt`,
`rcda6_deltas.npz`, `part6_eval_trained.json`, `ppl_check.json`,
`tw_relative.json` (in `results/`); the decomposition JSON lives canonically
in `R CDA - 5/Research/RCDA/results/rcda6_channel_decomposition.json`.
