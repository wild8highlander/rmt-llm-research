# RCDA-7 plan — deep-phase-only penalty + tightened anchor

Written: November 18, 2026 (pre-registered before the round's training run).

## Question

The program's next leverage point after RCDA-6: can the same actuator set buy
a deeper penalty descent by (a) enforcing the formula only where the loop is
read out — the deep phase — and (b) halving the tolerated drift? The sibling
kit's RCDA-5 bought its −56% at KL 0.41 (honestly recorded above the agent
gate); this round targets a descent that survives at two orders of magnitude
less drift.

## Pre-registered decisions

1. **Trainable set**: unchanged from RCDA-6 (`o_proj + down_proj` × L6/L7/L8,
   8.85M params) — the tensor-set lever is *not* the subject here.
2. **Penalty scope (lever 1)**: single deep-phase point, the L8 state (the
   decomposition's MLP-share peak, 83%).
3. **Anchor (lever 2)**: β = 20, cap = 0.5 (half the tolerated mse drift,
   double the pressure) — scope `good` unchanged.
4. **Budget**: lr 2e-5 (drift-matched), 150 Adam steps, refresh 32, clip 1.0.
5. **Evaluator**: unchanged; the round's primary readouts are the descent
   percentage and the descent-per-KL ratio ([`fig14`](../../../Graphics/fig14_leverage_chart.png)).

## Success criteria

- primary: penalty descent ≥ RCDA-6's (−62.6%) from a single point;
- primary: KL(base‖arm) ≤ 0.015 on the control corpus under the doubled
  anchor pressure;
- secondary: behavioural metrics vs RCDA-5/6 at the matched budget, reported
  regardless of outcome;
- secondary: relative TW gate on all six tensors; cos(u₁, v) per tensor.

## Outcome

**−80.0% penalty descent** (deepest of the program) at KL 0.011, PPL parity,
TW rel pass ×6, cos(u₁, v) up to 0.86 at the deep phase. Behaviour lands
between RCDA-5 and RCDA-6 (the 150-step budget caps the carry correction, not
the formula). Lever recommendation for the Gemma-scale run: inherit this
configuration. Full details: [`RCDA7_REPORT.md`](RCDA7_REPORT.md).
