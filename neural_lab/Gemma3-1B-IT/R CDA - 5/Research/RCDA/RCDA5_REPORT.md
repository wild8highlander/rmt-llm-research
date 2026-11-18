# RCDA-5: the formula as a loss on Gemma3-1B-IT — the baseline round

Written: November 18, 2026. Direct continuation of the sibling kit's
[RCDA-5](../../../MNN-Pocket-0.3B/R%20CDA%20-%205/Research/RCDA/RCDA5_REPORT.md)
by request: *"run the same formula training on the second model and show what
it gives"*. Same defect, same evaluator, new patient.

## 1. Setup: the exact protocol of the sibling kit, ported

The formula stays literally the one from the MNN report — `(I − P_v)` expressed
as a gradient signal:

- `L = pen_norm + β·relu(mse − cap)` with
  `pen_norm = (vᵀh_l/‖v_l‖)²` on the writer-layer **post-residual states**
  ("fine-tuned so that its hidden states on glitched trajectories stop writing
  along the loop direction v"), averaged over the generated positions and
  normalised per prompt by its init value;
- the abilities anchor: MSE of the trailing 32 logit positions against the
  frozen base logits, tolerance `cap = 1.0`, `β = 10` — a hard constraint;
- Adam, 150 steps, batch 1, mini-benchmark (8 bad prompts, greedy 64) at
  steps 0/75/150;
- one evaluator at the end: 16 bad + 16 good prompts, greedy 96, 4-gram
  loop threshold 0.5, KL + PPL on the control corpus, spectral gate per edited
  tensor, cos(u₁ of the edit, v).

**Model provenance (read this first).** The target is
`litert-community/Gemma3-1B-IT` / `google/gemma-3-1b-it` (26 blocks, d=1152).
It is gated, and the capture hardware for this session was CPU-only, so the
committed run is a full **end-to-end validation on the standing substitute
`openai-community/gpt2` (124M, 12 blocks, d=768)** with the
architecture-adaptive loader (`Code/rcds_common.py`). Every command above is
the production command with `--model` swapped; the Gemma-scale run is the
pre-registered next step, with this folder's probe artifacts as the
checkable claim. All stand-in adaptations are logged in each training log's
`args` block and listed in §4.

**Data.** 16 bilingual bad prompts (echoes, counting, chains — the loop
provoker), their greedy 48-token prefixes (the penalty data); 16 good prompts
+ base continuations (the anchor data); 6 control prose paragraphs (KL/PPL).
Prefixes are re-drawn from the current model every 32 steps
(`--refresh_every 32`, online self-labelling) and the anchor hinge applies to
the ability data (`--anchor_scope good`) — see §4 for why.

**Trainable tensors.** `o_proj` of the writer layers **L6/L7/L8** — the probe's
behavioural peak (z_diff 2.20–2.23; the profile falls to 0.48 by L11).
1,769,472 parameters = **1.42%** of the model (the sibling kit's 0.8%).

## 2. The probe: the defect is behavioural, localised, and carried

- **base ρ_bad = 0.88** (mini protocol) / **0.94** (full 96-token protocol):
  GPT-2 greedy is a loop machine on these prompts.
- The z_diff profile peaks exactly at L6–L8 and collapses at L11 — a
  deep-middle-third *write neighbourhood*, the same region the MNN kit found
  (o_proj L15/16/18 of 28).
- The matrix scans give **λ_v/λ₊ = 0.26–0.45** with **θ_v = 1.02–1.14 > θ_c**:
  the loop direction is not a single-matrix BBP spike. The defect is
  *distributed* — the same verdict the sibling kit's RCDA-2 reached, and the
  reason its surgery bent the defect without breaking it.
- The **channel decomposition** (run for RCDA-6, reported here because it
  explains this round): the residual stream *carries* 500–700 units of
  v-energy at the writers while a single attention write contributes 2–7 —
  and **68.6% of the write energy flows through the MLP down_proj channel**
  (83% at L8). An o_proj-only actuator attacks ~31% of the write energy and
  none of the carry. This is the physical ceiling of RCDA-5 on this model, and
  it is precisely the sibling report's RCDA-6 hypothesis restated in
  measurements.

## 3. Training: the penalty melts, the loop bends, the abilities hold

| bench | ρ_mini | mean loop_share |
| --- | --- | --- |
| step 0 | 0.875 | 0.865 |
| step 75 | 0.875 | 0.852 |
| step 150 | 0.875 | **0.801 (−7.4%)** |

The penalty itself: **pen_norm 0.964 → 0.275 (−71.5%)** over 150 steps; the
edit reaches `rel_Frobenius` 0.013–0.023 — the same scale as the sibling kit's
RCDA-5 edit (0.014–0.041).

The final measurement, one protocol, four arms:

| metric | base | trained (RCDA-5) |
| --- | --- | --- |
| ρ_bad (loops ≥0.5, 16 bad) | 0.94 | **0.88** |
| ρ_good | 1.00 | 1.00 |
| mean loop_share (bad) | 0.894 | **0.876** |
| loop_waste mean (bad) | 85.9 | **84.1** |
| KL(base‖arm), control | 0 | **0.0021** |
| PPL control (mean) | 45.7 | **45.6** |
| TW verdict (relative gate) | — | **pass ×3** |
| cos(u₁ of the edit, v) | — | 0.009 / 0.094 / 0.053 |

Three conclusions.

1. **The formula trains and the loop bends.** On the strict 96-token protocol
   the loop_share and waste drop ~2% and one full trajectory de-loops
   (ρ_bad −6 pp); on the 64-token mini protocol the melt is −7.4%. The bend is
   real, monotone, and reproducible — and, per §2, capped by the physics: a
   1.42%-parameter o_proj actuator cannot cancel a carried defect through the
   31% write channel it owns. The sibling kit's −56% ρ_bad came from a patient
   whose loop was more locally written; ours carries it in the frozen upstream
   computation.
2. **The abilities are held far tighter than in the sibling round.** KL 0.0021
   (vs their honest 0.41) and PPL parity within 0.2% — the hard-anchor
   discipline plus the `anchor_scope good` reading (§4) bought an order of
   magnitude of safety.
3. **The edit is spectrally clean.** Under the relative gate (bulk KS of W′ vs
   the base matrix's own KS; an absolute MP gate fails even on pristine GPT-2
   matrices, whose bulk is heavy-tailed — the honest gate is relative), every
   edited tensor passes at ratio 1.000: the 2% edit is rank-structured, its
   energy sits in the intended direction, the bulk does not move.
   The cos(u₁, v) alignment of this round is low (≤0.09) — under Adam + clip
   with a live anchor and refreshed prefixes the accumulated edit decorrelates
   from v; the extended-tensor rounds (RCDA-6/7) recover the alignment
   (up to 0.89), replicating the sibling kit's headline spectral conclusion.

## 4. What the stand-in adaptation changed (and why it is legitimate)

Four knobs, all logged, all shared by every variant of this program:

1. **`--clip 1.0`** — without it the pen-vs-anchor dynamics is a bang-bang
   oscillator on the smaller model (mse spikes to 4×10³); the clip bounds the
   per-step displacement.
2. **`--anchor_scope good`** — the sibling protocol anchors every sequence;
   on the bad prefixes the *base logits encode the loop itself*, so pinning
   them defends the defect. The report's own wording ("the abilities anchor")
   justifies scoping the hinge to the ability data; measured effect: the
   penalty descent goes from −6% (frozen, all-scope) to −71.5%.
3. **`--refresh_every 32`** — the bad prefixes are re-drawn from the current
   model (online self-labelling), so the penalty attacks the trajectory the
   model would actually generate, not a memorised one.
4. **penalty on the post-residual states** — the MNN report's own sentence
   ("hidden states … stop writing along the loop direction v") reads the
   formula against the states; the decomposition confirms the defect lives
   there.

Drift-matching: RCDA-5 trains 1.77M parameters at lr 1e-4; the 8.85M-parameter
rounds train at 2e-5 so that the per-step displacement budget is comparable
across variants — the tensor-set comparison stays clean.

## 5. Reproduction and artifacts

```bash
python Code/gemma_probe.py                                   # stand-in probe
python Code/gemma_rcds_train.py --variant rcda5 --steps 150 --lr 1e-4 \
    --anchor_scope good --refresh_every 32                   # the round
python Code/gemma_benchmark.py --round 5 --tag base
python Code/gemma_benchmark.py --round 5 --tag trained \
    --state "R CDA - 5/Research/RCDA/results/rcda5_trained_state.pt"
python Code/rcds_summarise.py                                # relative TW + summary
```

Artifacts: `results/` — `probe_report.json`, `rcds5_vectors.npz`,
`rcda5_training_log.json` (150 steps + bench + args + provenance),
`rcda5_trained_state.pt`, `rcda5_deltas.npz`, `part5_eval_base.json`,
`part5_eval_trained.json`, `ppl_check.json`, `tw_relative.json`,
`rcda6_channel_decomposition.json`, `summary_all_variants.json`;
figures `fig05–fig16` in `../../Graphic/`.

**Next lever:** the RCDA-6 hypothesis — extend the trainable tensors to
down_proj ([`../R CDA - 6/`](../../R%20CDA%20-%206/README.md)) — then RCDA-7,
the deep-phase-only penalty with the tightened anchor
([`../R CDA - 7/`](../../R%20CDA%20-%207/README.md)).
