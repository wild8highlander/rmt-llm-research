# RCDA-5 for Gemma3-1B-IT: teach the model the formula — the transfer test

Written: November 16, 2026. A direct port of the MNN-Pocket-0.3B RCDA-5
protocol ("train the defect out — bake `(I − P_v)` into training as a gradient
penalty") to **litert-community/Gemma3-1B-IT** (base checkpoint
`google/gemma-3-1b-it`, 1.04 B parameters, 26 layers, hidden 1152, vocab
262 144, bf16). Same methodology — spectral probe, the formula as the loss,
Tracy–Widom control, abilities anchor — with the adaptations the architecture
demanded. **The transfer did not reproduce the MNN result; both training arms
suppressed the penalized spectral signature while the behavioral defect got
stronger.** The numbers are recorded exactly as measured.

## 1. Setup: the formula as a loss function, on Gemma3

The penalty is literally `(I − P_v)` expressed as a gradient signal: the model
is fine-tuned so that its residual states on looping trajectories stop writing
along the loop direction `v`.

- **Model**: `google/gemma-3-1b-it` (the identical weights that
  [litert-community/Gemma3-1B-IT](https://huggingface.co/litert-community/Gemma3-1B-IT)
  packages for on-device use; pulled for training from the public mirror
  `unsloth/gemma-3-1b-it`, checksum-verified, Gemma license applies).
- **Protocol adaptation (documented)**: prompts are fed in **raw completion
  mode** (no chat template) — the closer analog of the MNN-Pocket protocol,
  where a plain completion model was probed greedily. In chat mode the model
  shows no defect at all (sentinel scan: loop share ≤ 0.09 on 20 stress
  prompts at 96 tokens). The loop window is **192 tokens** (`MAX_NEW`): at 96
  tokens the defect does not open on Gemma3-1B-IT.
- **Sentinel selection (pre-registered)**: 50 candidate pattern-continuation
  prompts scanned on the base model (greedy, 192 tokens, 4-gram loop labels);
  the **top-20 by base loop share** became the frozen bad set
  (`prompts/bad.jsonl`, ids `bad_g01…bad_g20`, selection record in
  `results/sentinel_selection.json`). 12/20 hard-loop at base
  → **ρ_bad(base) = 0.60**.
- **Probing (`gemma_probe.py`)**: for looping trajectories, residual states at
  generated positions inside the loop vs before the collapse; per-layer
  `Δh`, diagonal-Mahalanobis separability `z_diff`, and the Rayleigh spike
  strength `β` of the loop-state covariance. The **writer band is L10–L12**
  (z_diff 212.6 / 219.8 / 205.1 — a clean unimodal peak, the analog of Pocket's
  L15/16/18). `v_l` = normalized `Δh_l` per writer layer.
- **Trainable tensors**: exactly the tensors surgery would touch — the o_proj
  of layers **10/11/12** (3 538 944 parameters, 0.35 % of the model).
- **Step loss (the final form of the training)**:

  ```
  L = pen_norm + β · relu(mse − cap)
  pen_norm = mean_l mean_t ( v̂_lᵀ h_t )² / pen0_j        (dimensionless, init = 1.0)
  mse      = MSE(trailing 32 logits, base logits) on a good prompt
             + the base greedy 192-token completion
  v̂_l      = v_l / ‖v_l‖ measured by the probe (layers 10/11/12)
  h_t      = residual stream after layer l at generated position t
  ```

  β = 10, cap = 1.0 (a hard anchor, not a soft term), gradient clipping at
  max-norm 1.0 on the SGD arm. Per-prompt init normalization `pen0_j`
  replaces the MNN kit's `pen_ref` scaling: Gemma's residual norms are ~3
  orders of magnitude larger than Pocket's, so the penalty is kept
  dimensionless. SGD runs on fp32 master weights (exact updates), the model
  computes in bf16.
- **Two arms**: `trained` — SGD (momentum 0.9, lr 3e-3, grad-clip 1.0), 150
  steps, batch 1, ~1.3 s/step CPU (the user-specified optimizer); `adam` —
  the MNN RCDA-5 recipe control (Adam, lr 3e-5, no clip), 150 steps. Mini-
  benchmark (ρ_mini, 8 bad, greedy 192) at steps 0/75/150 in both logs.

## 2. Training: the penalty collapses, the anchor holds — and the loops do not care

| bench | SGD arm ρ_mini | SGD mean ls | Adam arm ρ_mini | Adam mean ls |
| --- | --- | --- | --- | --- |
| step 0 | 1.00 | 0.921 | 1.00 | 0.921 |
| step 75 | 1.00 | 0.894 | **0.875** | 0.846 |
| step 150 | 1.00 | 0.890 | 0.875 | 0.857 |

- **SGD arm**: `pen_norm` 1.0 → **0.0084 (−99.2 %)** — the penalized spectral
  signature vanished from the training prefixes. The anchor fought back hard:
  MSE rode the cap (max 6.16, final 0.60 vs cap 1.0; β·relu engaged
  repeatedly), i.e. the optimizer was pinned against the abilities constraint.
- **Adam arm**: `pen_norm` 1.0 → **0.2236 (−78 %)** with the anchor mostly
  satisfied (mean MSE of the last 50 steps 0.40, no sustained cap fight) —
  the closest analog of the MNN dynamics (their pen 138 → 60).
- The behavioral mini-benchmark moved only inside its noise band in both arms
  (the 8 hardest loopers of the bad set).

## 3. The new measurement: three arms, one evaluator

The evaluator is the ported `rcda4_eval_b.py`: 20 bad + 20 good prompts
(greedy 192, 4-gram loop labels, loop threshold 0.5), KL(base‖arm) on the 5
control texts, Tracy–Widom spectral control per edited tensor, and the cosine
between the edit's top output direction `u₁` and the measured loop direction
`v_l`.

| metric | base | trained (SGD-150) | adam (Adam-150) |
| --- | --- | --- | --- |
| ρ_bad (loops ≥ 0.5 on 20 bad) | 0.60 | **1.00** | **0.95** |
| ρ_good | 0.10 | 0.10 | 0.05 |
| loop_share mean (bad) | 0.562 | **0.911** | **0.908** |
| loop_waste mean (bad), tokens | 147.1 | **180.1** | **177.9** |
| KL(base‖arm), control | 0 | 0.2213 | 0.0248 |
| TW verdict | — | violation ×3 | violation ×3 |
| cos(u₁ of the edit, v) | — | 0.32 / 0.18 / 0.31 | 0.08 / 0.05 / 0.14 |
| rel_Frobenius of the edit (L10/11/12) | — | 0.026 / 0.017 / 0.008 | 0.019 / 0.019 / 0.018 |
| columns changed > 1 % (of 1 179 648) | — | 557 k / 375 k / 262 k | 822 k / 824 k / 827 k |
| PPL control (mean) | 10.9 | **10.2** | 11.0 |

Four conclusions from the table.

1. **Abilities are preserved** — the only part of the MNN result that
   transferred cleanly. PPL parity holds in both arms (10.2 and 11.0 vs base
   10.9); ρ_good did not degrade; the anchor did its job.
2. **The behavioral cure inverted**. ρ_bad went **0.60 → 1.00 (SGD)** and
   **0.60 → 0.95 (Adam)**: after training, every bad prompt loops, and loops
   harder (loop share 0.56 → 0.91; waste 147 → 180 tokens). Both arms agree,
   so this is not an optimizer accident: suppressing the `v`-projection at
   L10–L12 does not remove the repetition attractor of Gemma3-1B-IT — it
   removes (or destabilizes) whatever part of the circuit was letting the
   model escape the loop. The penalized component is correlated with the
   defect, not its cause.
3. **SGD did not converge to a spectrally clean edit** — the direct opposite
   of the MNN finding ("the model found a spectrally clean solution of the
   same class on its own"). TW control fails ×3 in both arms: the residual
   (v-orthogonal) bulk of the edit is structured, not random (z_resid in TW
   units: 7304 / 5194 / 854 for the SGD arm), and the edit's top direction is
   not `v` (cos 0.32 / 0.18 / 0.31 vs MNN's 0.72 / 0.64 / 0.29). The MNN
   convergence-to-the-primitive is therefore **architecture-specific**: a
   3-o_proj-localized defect produces it; Gemma3's distributed loop does not.
4. **What did transfer**: the probe methodology (an unimodal writer band with
   z_diff ≈ 200 above a quiet background), the penalty machinery (the model
   will minimize `(v̂ᵀh)²` to 1 % of its init value under a hard abilities
   anchor), and the evaluator. What did not: the causal claim. The RCDA-6
   hypothesis for Gemma3: the loop is written jointly by attention output
   *and* MLP down_proj across a wider band (L8–L15 all show z_diff > 130), and
   the probe direction must be conditioned on the loop *phase*, not only the
   collapse point.

## 4. What this says about "weight-space programming"

- The MNN-Pocket result — "the formula as a loss finds a spectrally clean,
  behaviorally effective edit on its own" — is real but **not a law**: it
  holds when the defect is localized in the tensors the penalty touches. On
  Gemma3-1B-IT the same protocol produces a *penalty-compliant,
  behaviorally wrong* solution: the model happily sacrifices the measured
  direction while the defect migrates/strengthens elsewhere. Any claim that
  spectral suppression implies behavioral repair needs an architecture-level
  control like this one.
- The anchor design (hard cap + β·relu, trailing-32 logits) is validated: it
  absorbed even the SGD arm's cap-riding dynamics with zero PPL damage. For
  Gemma3 the anchor is not the bottleneck; the penalty target definition is.
- The LR calibration ladder matters on CPU-scale budgets: 3e-5/3e-4 SGD were
  inert (pen −0 %/−3 % over 20 steps), 3e-3 collapsed the penalty in 150
  steps but overdrove the anchor. The recipe is recorded in
  `config/gemma_rcda5.yaml`.

## 5. Reproduction and artifacts

```bash
python3 Code/gemma_probe.py        --model <gemma-3-1b-it>          # stage 1: writers + v
python3 Code/gemma_rcds_train.py   --model <gemma-3-1b-it> --steps 150 --lr 3e-3 \
                                   --optim sgd  --bench-at 75 150    # the user-spec arm
python3 Code/gemma_rcds_train.py   --model <gemma-3-1b-it> --steps 150 --lr 3e-5 \
                                   --optim adam --bench-at 75 150    # MNN-recipe control
python3 Code/gemma_benchmark.py    --model <gemma-3-1b-it> --edited-state none \
                                   --tag base --out part5 --reuse-gens
python3 Code/gemma_benchmark.py    --model <gemma-3-1b-it> \
                                   --edited-state <work>/trained_state.safetensors \
                                   --tag trained --out part5
python3 Research/RCDA/tools/rcda5_ppl.py      # abilities PPL parity
python3 Research/RCDA/tools/rcda5_analyze.py  # spectral digest + rcds5_vectors.npz
python3 Research/RCDA/tools/rcda5_figures.py  # fig15 + figG1–G3
```

Artifacts (this folder): `results/` — `training_log.json` (SGD arm: 150 steps
with `mse`, `pen_raw`, `pen_norm`, `pen_layers`, benches at 0/75/150, the final
TW/cos analysis), `training_log_adam.json`, `part5_eval_{base,trained,adam}.json`,
`part5_rows_*.jsonl` (per-prompt rows), `ppl_check.json`,
`rcda5_spectral_digest.json`, `rcds5_vectors.npz` (v per writer layer, top-16
singular values of ΔW, u₁), `probe_report.json`, `probe_generations.jsonl`,
`sentinel_selection.json`; `figures/` fig15 + figG1–G3 (mirrored in
`../../Graphic/`). The fine-tuned weights (SGD arm) ship in
[`Model ZIP/`](../../Model%20ZIP/) as a checksummed self-assembling archive
(17 volumes, each ≤ 95 MB, GitHub-safe).
