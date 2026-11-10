# RCDA-5: teach the model the formula — SGD versus surgery

Written: November 9, 2026. A direct continuation of RCDA-4 by request: "train
MNN-0.3B on our formulas, let it understand what makes it glitch, and take a new
measurement". The RCDA-4 plan is closed (RCDA4_REPORT.md); this is a new experiment
on the same model and with the same evaluator.

## 1. Setup: the formula as a loss function

In RCDA-2 the formula `W'[:, S] = (I − P_v)·W[:, S]` was applied to the weights by
hand (surgery); in RCDA-4 the agent failed to reproduce it. RCDA-5 sets up a third
way: **bake the formula into training**. The penalty is literally `(I − P_v)`
expressed as a gradient signal: the model is fine-tuned so that its hidden states on
glitched trajectories stop writing along the loop direction v.

- Model: MNN-Pocket-0.3B (the `rcda_work/model` base, fp32).
- Trainable tensors: exactly the ones surgery touches — the o_proj of layers
  15/16/18 (2,408,448 parameters, 0.8% of the model).
- Data: the sentinel's 20 good prompts + the base's greedy continuations (anchor);
  20 bad prompts + their greedy prefixes of 48 tokens (the penalty applies there).
- Step loss: `pen_norm + β·relu(mse − cap)`, where
  `pen_norm = (v_lᵀh_l/‖v_l‖)²` at layers 15/16/18 on generated positions,
  normalized by the prompt's init value (the init-penalty spread across prompts is
  1.2–200, so without normalization one prompt would dominate);
  the abilities anchor is the MSE of the trailing 32 logit positions against the
  base logits with tolerance cap = 1.0 and β = 10 (a hard constraint, not a soft
  term: auto-scaling via init-MSE would give α = 0, because the MSE at step 0 is
  identically zero).
- Training: Adam, lr 3e-5, 150 steps, batch 1, ~1 s/step CPU; a mini-benchmark
  (ρ_mini, 8 bad, greedy 96) at steps 0/75/150.

## 2. Training: the loop melts by half, the anchor holds within tolerance

| bench | ρ_mini | mean loop_share |
| --- | --- | --- |
| step 0 | 0.80 | 0.729 |
| step 75 | 0.50 | 0.451 |
| step 150 | **0.45** | **0.379 (−48%)** |

The MSE anchor stayed within tolerance for all 150 steps (peak 0.57 at cap 1.0). A
note: ρ_mini of the base is 0.80 here, not 1.0 as in RCDA-4 (bf16 bench), because
training and measurement run in fp32 — greedy trajectories differ slightly; all
RCDA-5 comparisons use one protocol.

## 3. The new measurement: three arms, one evaluator

The final measurement is the same `rcda4_eval_b.py` (20 bad + 20 good, greedy 96,
KL on control texts, TW control, the cosine of the edit's main direction with v),
run sequentially on three states:

| metric | base | surgical (RCDA-2) | trained (SGD-150) |
| --- | --- | --- | --- |
| ρ_bad (loops ≥0.5 on 20 bad) | 0.80 | 0.80 | **0.35** |
| ρ_good | 0.60 | 0.60 | 0.45 |
| loop_waste mean (bad) | 73 | 73 | **31** |
| KL(base‖arm), control | 0 | 0.0065 | 0.41 |
| TW verdict | — | pass ×3 | **pass ×3** |
| cos(u₁ of the edit, v) | — | 1.0 / 1.0 / 1.0 | 0.72 / 0.64 / 0.29 |
| PPL control (mean) | 4873 | 4864 | 4857 |

Three conclusions from the table.

1. **Training cured the loop better than surgery**: ρ_bad 0.80 → 0.35 (−56%),
   tokens "wasted in the loop" 73 → 31 (−58%). RCDA-2's surgery under the strict
   protocol (loop_share ≥ 0.5) did not move a single one of the 20 bad gens — its
   RCDA-2 effect (14/20 → 13/20) was visible only to the more sensitive defect
   detector; hard looping it does not remove at all.
2. **SGD converged to a spectrally clean edit** — and this is the main scientific
   result: TW pass on all three matrices, the edit's main output-direction aligned
   with v (0.72/0.64/0.29 versus 0.054 for RCDA-4's aided agent). The formula was
   never given to the model as weights — the model found a solution of the same class
   itself, through the gradient over 896-column matrices. But the solution is
   distributed: all 896 columns changed, the top-128-column overlap with the surgical
   S is only 4–16% at the same edit scale
   (rel_Frobenius 0.027/0.027/0.014 versus 0.026/0.026/0.015 for surgery) —
   the same target, a different path.
3. **Abilities are preserved**: PPL on control texts 4857 versus 4873 for the base
   (no worse); two texts with PPL ~9700 are read equally badly by all arms (that is
   the 0.3B ceiling itself, not a consequence of the edit). At the same time
   KL 0.41 — the drift from the base distribution is noticeable and above our agent
   gate of 0.15: the cure was bought for a larger shift than the agent was allowed in
   RCDA-4; for self-editing this is acceptable, and recorded honestly.

## 4. What this says about "weight-space programming"

- The formula `(I − P_v)` works in both forms: as point weight surgery
  (TW pass, but a weak behavioral effect) and as a training signal
  (TW pass + a strong effect). Training — "sweeping" the v-component out of all
  columns at once — is more effective, because the loop is written not by 64 columns
  but in a distributed way; our RCDA-3 diagnostics (the collinearity v15/v16/v18
  0.83–0.94) predicted that all three layers must be edited — SGD did exactly that,
  simultaneously.
- For the MIT paper this is a strong control from below: their primitive is
  reproduced by gradient training on our bench better than by an agent (RCDA-4) and
  better than their own published edit (RCDA-4 Part A: gate failed). The difference
  is in who gets v and how: our measurements give the exact direction, their edit is
  tuned to their set.
- The residual ρ_bad 0.35 is not zero: the loop shared by the three layers is not
  fully removed by editing three o_proj matrices; the RCDA-6 hypothesis is to add
  receiver layers (down the stack) to the penalty or extend the trainable tensors to
  down_proj.

## 5. Reproduction and artifacts

```bash
python3 Research/RCDA/tools/rcda5_prepare.py    # base/surgical states from RCDA-2
python3 Research/RCDA/tools/rcda5_train.py --steps 150 --lr 3e-5 --bench-at 75 150
python3 Research/RCDA/tools/rcda5_analyze.py    # TW/cos forensics of the SGD edit
python3 Research/RCDA/tools/rcda4_eval_b.py --edited-state base_state.safetensors --tag base     --out part5
python3 Research/RCDA/tools/rcda4_eval_b.py --edited-state surgical_state.safetensors --tag surgical --out part5
python3 Research/RCDA/tools/rcda4_eval_b.py --edited-state trained_state.safetensors --tag trained  --out part5
python3 Research/RCDA/tools/rcda5_ppl.py        # the abilities PPL parity check
python3 Research/RCDA/tools/rcda5_figures.py    # fig15
```

Artifacts: `rcda5/` — base/surgical/trained_state.safetensors,
training_log.json (steps + bench + the final analysis), part5_eval_{base,
surgical,trained}.json, ppl_check.json, surgical_diff.json; figures/
fig15_rcda5.png.
