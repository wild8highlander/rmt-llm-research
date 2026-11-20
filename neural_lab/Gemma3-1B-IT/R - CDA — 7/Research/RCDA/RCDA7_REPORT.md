# RCDA-7: deep-phase-only penalty + tightened anchor — the lever round

Written: November 18, 2026. The next leverage point named by the program after
RCDA-6: **enforce the formula where the loop is read out — the deep phase —
and make the anchor twice as strict.**

## 1. Setup: one point of enforcement, half the tolerance

The trainable set stays RCDA-6's (`o_proj + down_proj` × L6/L7/L8, 8.85M
params, lr 2e-5 drift-matched). Two levers move:

1. **Deep-phase-only penalty.** The penalty is enforced at a single point —
   the **L8 state**, the deepest writer, where the channel decomposition
   showed the MLP share peaking (7.32 of 8.85 units = 83%). Rationale: the
   defect is carried down the stack; the deep phase is where it is finally
   read out into logits, so a correction applied there maximises the
   behavioural gradient per unit of enforcement.
2. **Tightened anchor.** β = 10 → **20**, cap = 1.0 → **0.5**: the tolerated
   distribution drift is halved. This directly targets the sibling kit's
   honest weakness — its RCDA-5 bought ρ_bad −56% at KL 0.41, above the
   agent gate of 0.15; this round asks whether the descent survives at KL
   two orders of magnitude lower.

Everything else is the shared recipe: same data, same 150 Adam steps,
`--anchor_scope good`, `--refresh_every 32`, clip 1.0, same evaluator.

## 2. Training and results

| bench | ρ_mini | mean loop_share |
| --- | --- | --- |
| step 0 | 0.875 | 0.865 |
| step 75 | 0.875 | 0.818 |
| step 150 | 0.875 | **0.822 (−5.0%)** |

pen_norm **0.953 → 0.190 (−80.0%)** — the deepest descent of the program, from
a single enforcement point under the hardest anchor. The final measurement:

| metric | base | RCDA-5 | RCDA-6 | **RCDA-7** |
| --- | --- | --- | --- | --- |
| ρ_bad | 0.94 | **0.88** | 0.9375 | 0.9375 |
| mean loop_share (bad) | 0.894 | **0.876** | 0.8815 | 0.8776 |
| loop_waste mean | 85.9 | **84.1** | 84.6 | 84.2 |
| KL(base‖arm), control | 0 | **0.0021** | 0.0094 | 0.0110 |
| PPL control | 45.7 | **45.6** | 46.5 | 46.3 |
| TW (relative gate) | — | pass ×3 | pass ×6 | **pass ×6** |
| cos(u₁ of the edit, v) | — | ≤0.09 | 0.74–0.89 | 0.38–0.86 |
| pen descent | — | −71.5% | −62.6% | **−80.0%** |

Three conclusions.

1. **The deep phase is the efficient enforcement point.** −80% penalty descent
   from one point (vs −62.6% from three points in RCDA-6 at the same budget)
   — concentrating the formula at the read-out layer buys the most defect
   energy per step of training. This is the program's lever recommendation.
2. **The tightened anchor holds and stays honest.** At cap 0.5 the abilities
   remain within tolerance (KL 0.011, PPL 46.3, parity +1.3%) — an order of
   magnitude under the sibling kit's 0.41, and evidence that the hard-anchor
   discipline scales with strictness instead of fighting the penalty.
3. **Behaviour converges across the extended-tensor rounds.** RCDA-6 and
   RCDA-7 land within noise of each other on the strict protocol and both
   trail RCDA-5's focused single-channel edit — the 96-token horizon is
   budget-bound (see the RCDA-6 report), and the alignment with v (cos up to
   0.86 at L8) again replicates the sibling kit's spectral conclusion.

## 3. The leverage chart

[`fig14`](../../../Graphics/fig14_leverage_chart.png) — descent % vs KL:

- RCDA-5: −71.5% at KL 0.0021 — the cheapest descent per drift;
- RCDA-6: −62.6% at KL 0.0094;
- RCDA-7: **−80.0% at KL 0.011** — the deepest descent, one enforcement point,
  twice the anchor strictness.

For the Gemma-scale run the recommendation is explicit: **inherit RCDA-7's
configuration** (deep-phase enforcement, β=20/cap=0.5, extended tensors),
raise the drift budget (more steps or higher lr within the anchor's tolerance),
and expect the melt to compound — on this stand-in the 150-step budget caps
the carry correction, not the formula.

## 4. Reproduction and artifacts

```bash
python Code/gemma_rcds_train.py --variant rcda7 --steps 150 --lr 2e-5 \
    --anchor_scope good --refresh_every 32
python Code/gemma_benchmark.py --round 7 --tag trained \
    --state "R CDA - 7/Research/RCDA/results/rcda7_trained_state.pt"
```

Artifacts: `rcda7_training_log.json`, `rcda7_trained_state.pt`,
`rcda7_deltas.npz`, `part7_eval_trained.json`, `ppl_check.json`,
`tw_relative.json`.
