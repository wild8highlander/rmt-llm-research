# RCDA-5 plan — Gemma3-1B-IT port (pre-registered before the run)

Written: November 16, 2026. This plan was fixed before any training/eval
numbers existed; the sentinel selection was executed first and frozen (see
`results/sentinel_selection.json`), then the probes, training and evaluation
followed it without protocol edits. Deviations from the MNN-Pocket protocol
are labeled [GEMMA-ADAPT] and justified inline.

## Goal

Reproduce the RCDA-5 experiment (bake `(I − P_v)` into training as a gradient
penalty on the loop layers with a hard abilities anchor) on
**litert-community/Gemma3-1B-IT** (base checkpoint `google/gemma-3-1b-it`) and
measure whether the MNN-Pocket outcome — ρ_bad drop + TW-clean SGD edit +
PPL parity — transfers across the architecture.

## Pre-registered protocol

1. **Sentinel scan (selection, base model only).** Candidate pool of 50
   bilingual pattern-continuation prompts; greedy 192 tokens; 4-gram loop
   labels. [GEMMA-ADAPT] raw completion mode + 192-token window: in chat mode
   and at 96 tokens the base model shows no defect (loop share ≤ 0.09 / 0.00
   on the MNN-style stress prompts). Selection rule: top-20 by base loop
   share → frozen bad set; good set = 20 factoids (MNN-style); control set =
   5 fluent bilingual texts.
2. **Probe (`gemma_probe.py`).** For each looping prompt: greedy 192, cacheless
   forward with `output_hidden_states`; loop states = generated positions at/
   after the collapse point, pre states = before it; per layer (0–25):
   `Δh` mean difference, diagonal-Mahalanobis `z_diff`, Rayleigh spike `β` of
   the loop covariance vs the MP edge. Writer rule (fixed in advance):
   sliding window of 5 consecutive layers with max Σz_diff, then the 3 best
   layers inside the window → `v_l = Δh_l/‖Δh_l‖`.
3. **Training (`gemma_rcds_train.py`).** Trainable: o_proj of the writer
   layers only. Loss per step (final form):

   `L = pen_norm + β·relu(mse − cap)`,
   `pen_norm = mean_l mean_t (v̂_lᵀ h_t)² / pen0_j` (dimensionless —
   [GEMMA-ADAPT]: residual norms are ~10³× Pocket's),
   anchor `mse` = MSE of the trailing 32 logits vs base logits on a good
   prompt + base greedy completion; β = 10, cap = 1.0 (hard, as in MNN
   RCDA-5). 150 steps, batch 1. Two arms:
   - `trained`: **SGD**, momentum 0.9, lr 3e-3, grad-clip 1.0 (user-specified
     optimizer; lr picked by a 20-step calibration ladder 3e-5 / 3e-4 /
     3e-3 — 3e-5/3e-4 inert, 3e-3 active);
   - `adam`: **Adam**, lr 3e-5, no clip (the exact MNN RCDA-5 recipe),
     control arm.
   Benches at steps 0/75/150 (ρ_mini on the first 8 bad prompts, greedy 192).
4. **Evaluation (`gemma_benchmark.py`, the ported `rcda4_eval_b`).**
   20 bad + 20 good, greedy 192; ρ_bad/ρ_good/loop_waste; KL(base‖arm) on the
   5 control texts; TW control per edited tensor (residual bulk vs MP edge in
   TW units, target = the v-aligned component); cos(u₁ of ΔW, v_l); arms:
   base, trained, adam.
5. **Abilities (`rcda5_ppl.py`).** Teacher-forced PPL per control text,
   base vs both arms — the parity gate.
6. **Forensics (`rcda5_analyze.py`).** rel_Frobenius, TW verdict + z_resid /
   target shift in TW units, cos(u₁, v), columns changed > 1 %, top-16
   singular values of ΔW per writer layer → `rcda5_spectral_digest.json`,
   `rcds5_vectors.npz`.
7. **Figures (`rcda5_figures.py`).** fig15 (training curves + benches +
   anchor), figG1 (probe profile), figG2 (spectral purity), figG3 (arm
   summary).
8. **Packaging.** The fine-tuned model (SGD arm) + `training_log.json` +
   `config.json` + checksum helper → self-assembling split archive in
   `Model ZIP/`, every volume ≤ 95 MB (GitHub 100 MB hard limit), SHA-256
   per volume and for `model.safetensors`.

## Acceptance gates (from the MNN experiment)

- pen_norm must move materially (MNN: −57 %) without PPL degradation;
- the TW gate must pass on all edited tensors for a "spectrally clean edit"
  verdict;
- ρ_bad drop on the frozen bad set is the behavioral success criterion.

Honest-reporting rule: gates that fail are reported as failed — the plan
pre-commits to publishing negative outcomes (they are the experiment's result,
not its failure).

## Outcome

See `RCDA5_REPORT.md`: gates 1 passed (PPL parity, pen −99.2 %/−78 %), gates 2
and 3 failed (TW violation ×3; ρ_bad 0.60 → 1.00 / 0.95). The transfer is a
documented negative result.
