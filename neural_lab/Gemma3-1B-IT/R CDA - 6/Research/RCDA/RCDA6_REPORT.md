# RCDA-6 for Gemma3-1B-IT: the expanded trainable set, phase-conditioned

Written: November 16, 2026. RCDA-6 executes the hypothesis stated at the end
of the RCDA-5 report:

> "the loop is written jointly by attention output *and* MLP down_proj across
> a wider band (L8–L15 all show z_diff > 130), and the probe direction must be
> conditioned on the loop *phase*, not only the collapse point."

Both halves are implemented as pre-registered (`RCDA6_PLAN.md`), run on the
same model, the same frozen sentinel sets and the same evaluator as RCDA-5 —
only the trainable set and the penalty target change. **For the first time on
Gemma3-1B-IT the formula-as-loss moved the hard-loop attractor: the 12
prompts that hard-loop at base dropped from loop share 0.827 to 0.620
(−25 %), two of them stopped looping entirely — while RCDA-5's edit pushed
the same prompts up to 0.908.** The defect did not disappear: it migrated to
the mild prompts, and the edit costs abilities drift. The numbers are
recorded exactly as measured.

## 1. Setup: two changes against RCDA-5

- **Model**: `google/gemma-3-1b-it` (= the
  [litert-community/Gemma3-1B-IT](https://huggingface.co/litert-community/Gemma3-1B-IT)
  weights; trained from the public mirror `unsloth/gemma-3-1b-it`,
  checksum-verified, Gemma license applies). Protocol unchanged: raw
  completion mode, greedy 192, 4-gram loop labels, threshold 0.5.
- **Change 1 — the expanded trainable set.** o_proj **and** down_proj of
  every band layer L8–L15: 8 × (1152×1152 + 6912×1152) =
  **73 138 176 parameters, 7.31 % of the model** — 21× the RCDA-5 set
  (3 o_proj, 3.5 M, 0.35 %). Band justification: the RCDA-5 collapse-probe
  profile has z_diff 159/154/213/220/205/182/176/194 on L8…L15 — all above
  the 130 mark named in the hypothesis (L10–L12 stay the peak; L6/L7 sit
  just above threshold at 153/135 and were left out of the band as
  pre-registered).
- **Change 2 — the phase-conditioned directions** (`gemma_probe_phase.py`).
  The 14 looping trajectories of the frozen RCDA-5 probe are split into
  position phases: *pre* (before the collapse), *early* (the first 2 cycles
  after it; cycle period fitted from the token tail), *deep* (beyond). 9 of
  14 trajectories are period-regular (periods 4–32) and define the
  directions; 5 irregular ones are excluded from the pools (their phase
  structure is undefined). Measured on band layers:
  **z_early 75–118 vs z_deep 247–393** — the deep phase is 2.5–3.5× more
  separable than the collapse-mixed direction of RCDA-5 (z 205–220) — and
  **cos(v_early, v_deep) = 0.48–0.89**: the two phases point in substantially
  different directions, exactly what the hypothesis predicted.
  `v_L{l}.npy := v_deep` for the evaluator (the deep phase dominates long
  loops); the RCDA-5 collapse-only direction stays available in the RCDA-5
  kit's `rcds5_vectors.npz` / `vectors_all.npz` for comparison.
- **Step loss (the final form of the training)** — same formula, better
  measured target:

  ```
  L = pen_norm + β · relu(mse − cap)
  pen_norm = pen_raw / pen0_j                       (dimensionless, init 1.0)
  pen_raw  = mean_l mean_t ( v̂_{phase(t), l}ᵀ h_t )²    over band layers l
  phase(pre) = phase(early) → v̂_early ;  phase(deep) → v̂_deep
  mse        = MSE(trailing 32 logits, base logits) on a good prompt
               + the base greedy 192-token completion
  ```

  β = 10, cap = 1.0 (hard anchor), grad-clip max-norm 1.0, SGD momentum 0.9,
  lr 3e-3 (the RCDA-5 primary-arm recipe — no re-tuning), fp32 masters over
  bf16 compute, batch 1, 150 steps (~5–8 s/step CPU). Per-prompt init
  normalization pen0_j (median init pen 5.29e6).
- **Data**: identical frozen prompts; the base greedy completions/prefixes
  are seeded from the RCDA-5 cache (same model, same decoding — byte-equal).

## 2. Training: the penalty collapses again — and the hard loops move

| bench | ρ_mini | mean loop share |
| --- | --- | --- |
| step 0 | 1.00 | 0.921 |
| step 150 | **0.875** | **0.667 (−28 %)** |

- `pen_norm` 1.0 → **0.028** (mean of the last 50 steps, −97.2 %; final step
  0.0118) — the phase-matched penalty collapsed just as in RCDA-5.
- The anchor rode the cap hard: MSE > 1.0 on **105/150 steps** (mean of last
  50: 1.385, peak 3.52) — 74 M trainable params push much harder against the
  abilities constraint than RCDA-5's 3.5 M did (RCDA-5: peak 6.16, no
  sustained riding). The penalty and the anchor fought for the whole run.
- **Protocol deviation, recorded**: the planned mini-bench at step 75 was
  dropped mid-run (CPU time budget; the step-75 weights were not
  checkpointed). Benches are recorded at steps 0 and 150, the step-150 bench
  produced by the standalone `rcda6_minibench.py` (identical protocol,
  fresh process).

## 3. The new measurement: four states, one evaluator

`gemma_benchmark.py` (20 bad + 20 good, greedy 192; KL(base‖arm) on the 5
control texts; TW control per edited tensor; cos(u₁ of the edit, v)) run on
the RCDA-6 trained state; the base/RCDA-5 numbers are the RCDA-5 report's,
same protocol, same prompts.

| metric | base | RCDA-5 SGD-150 | RCDA-5 Adam-150 | **RCDA-6 SGD-150** |
| --- | --- | --- | --- | --- |
| ρ_bad (loops ≥ 0.5 on 20 bad) | 0.60 | 1.00 | 0.95 | **0.85** |
| ρ_good | 0.10 | 0.10 | 0.05 | **0.10** |
| loop_share mean (bad) | 0.562 | 0.911 | 0.908 | **0.627** |
| loop_waste mean (bad), tok | 147.1 | 180.1 | 177.9 | **160.1** |
| KL(base‖arm), control | 0 | 0.2213 | 0.0248 | **0.3074** |
| PPL control (mean) | 10.9 | 10.2 | 11.0 | **14.1** |
| TW verdict | — | violation ×3 | violation ×3 | **violation ×16** |
| cos(u₁ of the edit, v) | — | 0.32 / 0.18 / 0.31 (vs v) | 0.08 / 0.05 / 0.14 | **0.03–0.33 vs v_deep; up to 0.66 vs v_early** |
| rel_Frobenius (edited tensors) | — | 0.026 / 0.017 / 0.008 | 0.019 / 0.019 / 0.018 | **0.0006–0.0113 (16 tensors)** |
| trainable params | — | 3 538 944 | 3 538 944 | **73 138 176** |

The per-prompt structure (the loop-share migration, `figC5`):

| subset of the 20 bad prompts | base | RCDA-5 SGD | RCDA-5 Adam | **RCDA-6** |
| --- | --- | --- | --- | --- |
| hard-12 (loop share ≥ 0.5 at base) | 0.827 | 0.908 | 0.884 | **0.620 (−25 % vs base)** |
| mild-8 (loop share < 0.5 at base) | 0.164 | 0.915 | 0.944 | **0.637** |
| prompts fully unlooped by the edit | — | 0 of 12 | 0 of 12 | **2 of 12** (`bad_g02` 0.96→0.24, `bad_g11` 0.59→0.28) |
| prompts pushed into looping | — | 8 of 8 mild broke | 8 of 8 | **7 of 8 broke** (`bad_g17` 0.08→0.89 worst) |

Five conclusions from the tables.

1. **The phase hypothesis is measured-true at the probe level.** The
   deep-phase direction separates 2.5–3.5× better than RCDA-5's
   collapse-point direction, and the phase directions genuinely differ
   (cos 0.48–0.89, lowest at L12 — the old writer peak). The RCDA-5 penalty
   was aiming at a phase-mixed direction.
2. **The expanded set + phase target moved the attractor the RCDA-5 edit
   could not touch.** On the hard-12 the RCDA-5 arms *worsened* the loops
   (0.83 → 0.91 / 0.88); RCDA-6 is the **first arm that improves over the
   base model itself** (0.83 → 0.62, −25 %; two hard loopers fully repaired;
   `bad_g02` — a top-of-the-scale looper at base 0.96 — dropped to 0.24).
   The loop *share* — the fraction of the generation swallowed by the loop —
   is the metric this edit moves.
3. **But the defect migrates instead of dying.** On the mild-8 the edit
   creates loops (0.16 → 0.64; 7 of 8 cross the 0.5 line). Net ρ_bad
   0.85 vs base 0.60: better than RCDA-5 (1.00 / 0.95, −15 %), still worse
   than doing nothing. The Gemma3 loop is a distributed, redundant circuit:
   suppressing its deepest writing channel relieves the strong attractors
   and re-opens weak ones elsewhere.
4. **Abilities are NOT free this time.** PPL parity held in RCDA-5 (10.2 /
   11.0 vs base 10.9) and breaks in RCDA-6: 14.1 (+29 %), driven by the
   hardest control text (19.7 → 30.6); KL drift 0.31 vs 0.22 (SGD) — and
   RCDA-5's Adam arm had 0.025. A 7.3 % parameter edit at lr 3e-3 over 150
   steps buys behavioral change with real drift. The hard-cap anchor held
   PPL parity only for the small edit; it is not sufficient protection for
   the big one.
5. **Still no spectrally clean solution.** TW violation ×16 — the residual
   (v-orthogonal) bulk of every edit is structured (z_resid −150…+601 in TW
   units). The edit's top directions align with the phase directions at most
   0.33 (deep) / 0.66 (early, `L11.down_proj`) — far from MNN-Pocket's
   0.72/0.64/0.29 convergence-to-the-primitive. On Gemma3, formula-training
   edits remain spectral generalists: the "spectrally clean solution of the
   same class" stays an architecture-specific property of Pocket-0.3B.

## 4. What this says about "weight-space programming"

- The **прирост chain across the three formula-trained arms** is now
  measurable end to end: RCDA-5 (o_proj writers, collapse-v) made the defect
  stronger (loop share 0.56 → 0.91); RCDA-6 (o_proj+down_proj band,
  phase-v) cut that damage by a third (→ 0.63) and — for the first time —
  repaired part of the hard attractor (−25 % on hard-12). The formula's
  behavioral leverage scales with **how well the penalized direction matches
  the defect's actual writing structure** (phase-conditioned, both writer
  types), not with trainable-parameter count alone.
- The migration pattern (hard relieved, mild broken) matches the RCDA-5
  diagnosis — the penalized direction is a *correlate* of the loop circuit,
  not its single cause — and adds the refinement that the correlate is
  **phase-structured**: the deep-phase component carries the attractor.
- Next-step candidates implied by the measurement, in order of expected
  leverage: (a) penalize ONLY the deep-phase component on mild-prompt
  trajectories (the mild prompts broke through early-phase perturbation);
  (b) lower lr / more steps with the anchor cap tightened to protect
  abilities; (c) include the receiver-side embed/lm_head columns that read
  the band's output before declaring the circuit "distributed".

## 5. Reproduction and artifacts

```bash
python3 Code/gemma_probe.py --model <gemma-3-1b-it>            # RCDA-5 probe (shared)
python3 Code/gemma_probe_phase.py --model <gemma-3-1b-it>      # stage 1: phase directions
python3 Code/gemma_rcds6_train.py  --model <gemma-3-1b-it>     # stage 2: 150-step training
    --steps 150 --lr 3e-3 --optim sgd --bench-at 999
python3 Research/RCDA/tools/rcda6_minibench.py --tag step150   # the step-150 mini-bench
python3 Code/gemma_benchmark.py    --model <gemma-3-1b-it> \
    --edited-state <work>/rcda6/trained_state.safetensors \
    --tag rcda6_trained --out part6 --probe <work>/probe_phase
python3 Research/RCDA/tools/rcda6_ppl.py                       # abilities PPL check
python3 Research/RCDA/tools/rcda6_analyze.py                   # digest + vectors
python3 Research/RCDA/tools/rcda6_figures.py                   # figC1–C6
```

Artifacts (this folder): `results/` — `training_log.json` (150 steps ×
{mse, pen_raw, pen_norm, pen_layers, deep_share} + benches at 0/150 + the
final TW/cos analysis), `probe_phase_report.json` (phase profile of all 26
layers), `part6_eval_rcda6_trained.json` + `part6_rows_rcda6_trained.jsonl`
(per-prompt rows), `part6_tw_rcda6_trained.json` (TW per 16 tensors),
`ppl_check.json`, `rcda6_spectral_digest.json`, `rcds6_vectors.npz`
(v_early/v_deep per band layer, u1, top-16 singular values of each delta);
`figures/` figC1–C6 (mirrored in `../../Graphic/`). The fine-tuned weights
ship in [`Model ZIP/`](../../Model%20ZIP/) as a checksummed self-assembling
split archive of the **delta state** (16 tensors, 140 MB bf16 → 2 volumes
≤ 95 MB) — the base checkpoint it applies to is `google/gemma-3-1b-it`
itself; see [`Docs/UNPACK_MODEL.md`](../../Docs/UNPACK_MODEL.md).

Runtime notes (2 cores / 4 GB, honest): the full program ran on CPU; the
training sessions were chunked through the 10-step atomic checkpoints
(`ckpt_masters.pt` + `ckpt_opt.pt`), the anchor logits are a fp16 memmap, and
the mini-bench/benchmark were run in fresh processes because in-process
generation slows ~6–10× under the training allocator. The in-training bench
at step 75 was dropped for the same budget reason (deviation recorded in
§2).
