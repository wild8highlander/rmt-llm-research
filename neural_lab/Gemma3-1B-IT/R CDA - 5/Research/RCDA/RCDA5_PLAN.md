# RCDA-5 plan — the formula as a loss on Gemma3-1B-IT

Written: November 18, 2026 (pre-registered before the round's training run).

## Question

Does the sibling kit's RCDA-5 treatment — the repository formula
`(I − P_v)` baked into gradient training — transfer to a second, larger
patient (`litert-community/Gemma3-1B-IT` / `google/gemma-3-1b-it`), and what
exactly does it give on a loop defect measured with the same evaluator?

## Pre-registered decisions

1. **Defect protocol**: greedy decoding, 4-gram recurrence,
   ρ_bad = share of trajectories with loop_share ≥ 0.5; suite = 16 bilingual
   bad prompts (loop provokers) + 16 good (anchor) + 6 control texts (KL/PPL).
2. **Probe**: per-layer Δh between looped and unlooped tokens of the same
   continuations; writers = top-3 layers by (supercriticality, z_diff);
   the v-directions are frozen once and reused by every round.
3. **Loss**: `pen_norm + β·relu(mse − cap)`, `pen_norm = (vᵀh/‖v‖)²` on
   writer states, per-prompt init normalisation; β=10, cap=1.0; Adam;
   150 steps; batch 1; mini-bench at 0/75/150.
4. **Trainable set**: o_proj of the three writers only (the RCDA-5 baseline
   actuator).
5. **Evaluator**: the one evaluator — ρ_bad/ρ_good/loop_share/loop_waste,
   KL(base‖arm) on control, PPL per control text, spectral gate per edited
   tensor, cos(u₁ of the edit, v).
6. **Hardware honesty**: if the gated Gemma mirror cannot be pulled in the
   execution environment, the round runs on the declared CPU stand-in
   (`openai-community/gpt2`) with the architecture-adaptive loader and every
   adaptation logged; the Gemma-scale run is pre-registered as the follow-up.

## Success criteria

- the penalty descends monotonically on the bad trajectories (primary);
- the behavioural defect moves in the same direction (ρ_bad / loop_share /
  loop_waste) without an abilities penalty (KL, PPL within tolerance) —
  secondary, reported regardless of size;
- the spectral gate passes on every edited tensor (secondary);
- whatever the effect size, the carry/write decomposition is recorded to
  explain it — the negative result must be diagnosable.

## Outcome

Executed as planned on the stand-in. Primary: **−71.5% penalty descent**.
Secondary: ρ_bad 0.94→0.88, loop_share −2.1% (full) / −7.4% (mini), KL 0.0021,
PPL parity, relative TW pass ×3. The decomposition revealed the MLP-carried
structure (68.6% of writes) that caps the o_proj-only actuator → RCDA-6.
Full details: [`RCDA5_REPORT.md`](RCDA5_REPORT.md).
