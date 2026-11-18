# RCDA-6 plan — extend the trainable tensors to down_proj

Written: November 18, 2026 (pre-registered before the round's training run).

## Question

The sibling kit's RCDA-5 report named this round: *add receiver layers or
extend the trainable tensors to down_proj.* On the Gemma stand-in the
question splits in two:

1. is the premise true here — does the MLP channel write a material share of
   the loop direction? (a measurement, not a guess);
2. does doubling the actuator set buy behavioural melt at a matched drift
   budget?

## Pre-registered decisions

1. **Premise measurement first**: one forward pass per bad prompt with hooks
   on `o_proj`, `down_proj` and the block outputs; the v-energies of the
   three channels are committed before training
   (`rcda6_channel_decomposition.json`). If the MLP share were small, the
   round would be called off and RCDA-7 re-scoped.
2. **Trainable set (the only lever vs RCDA-5)**: `o_proj + down_proj` of the
   same writers L6/L7/L8 — 6 tensors, 8,847,360 params (7.11%).
3. **Drift matching**: lr 2e-5 so that `n_params × lr` matches RCDA-5's
   1.77M × 1e-4; every other knob identical (β=10, cap=1.0, scope `good`,
   refresh 32, clip 1.0, 150 Adam steps).
4. **Evaluator**: unchanged, one protocol for all arms.

## Success criteria

- premise: MLP write share reported and material (>50% would confirm the
  actuator extension is aimed correctly);
- behaviour: ρ_bad / loop_share / loop_waste vs RCDA-5 at the matched budget —
  reported regardless of which arm wins;
- spectral: relative TW gate on all six tensors; cos(u₁, v) per tensor —
  the alignment claim is the round's primary scientific readout;
- abilities: KL and PPL within the anchor tolerance.

## Outcome

Premise confirmed (MLP 68.6% of writes, 83% at L8). Behaviour at the matched
budget: loop metrics between base and RCDA-5 (the carry dominates the writes
~100× — closing write channels does not close the carry in 150 steps).
Spectral: TW rel pass ×6; **cos(u₁, v) = 0.74–0.89** — the sibling kit's
alignment conclusion replicates on a doubled tensor set. Abilities held
(KL 0.0094, PPL +1.8%). Full details:
[`RCDA6_REPORT.md`](RCDA6_REPORT.md).
