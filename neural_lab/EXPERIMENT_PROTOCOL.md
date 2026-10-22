# EXPERIMENT PROTOCOL — H1 v1.0

**Experiment No. 1: predicting LLM errors from internal features**
`experiment_id: error_prediction_rmt_v1` · `hypothesis_id: H1` ·
`protocol_version: 1.0` · status: **frozen before the run**

> This document fixes the protocol BEFORE the experiment runs. After
> the results are seen, changing it is forbidden: any change is issued
> as version v2 with an explicit deviation list. The pilot SMOKE run
> (49 tasks, greedy generation only) was used exclusively for pipeline
> debugging and sample-size calibration (n_indist 48 → 96) — it
> revealed the degeneracy of the primary endpoint under greedy
> generation but did not peek at any feature–label relationship.

---

## 1. Hypothesis

**H1.** The spectral characteristics of internal activations (the
eigenvalues of hidden-state covariances relative to the
Marchenko–Pastur bounds) contain information about the model's errors
**beyond** what standard confidence signals provide (probabilities,
entropies, perplexity).

A scientifically testable consequence: the combined predictor D
(confidence + internal) significantly improves ROC-AUC over predictor B
(confidence only) on held-out tasks, and the gain survives
leave-one-family-out.

Operational boundaries: an "error" is a numeric deviation from the
truth within a stated tolerance — not "hallucination / deceit /
intent". A spectral shift is not interpreted as the model's intention.

## 2. Model and its provenance

| Component | Value |
|---|---|
| Model | TinyGPT v3 (NumPy-only, ADR-001): pre-LN, RoPE, GQA 6q/2kv, 4 layers, hidden 96, ctx 64 |
| Parameters | 447,360 |
| Tokenizer | byte-BPE, vocab 512 |

> **Protocol v1.1 (ladder replication).** For the M (6 layers) and L
> (10 layers) models, the per-layer spectral features generalize to the
> actual depth (`gap_ratio_L0..L{n-1}`, `effrank_L0..L{n-1}`); the
> vector size becomes 2n+25 (S: 33, M: 37, L: 45). All other groups,
> targets, splits, and tests are unchanged; see the changelog (§9).

| Training | `research/tinygpt_formula/scripts/train_formula_model.py`, 40 epochs, seed 42, corpus `data/formula_corpus_train.txt` |
| Weights | `*.npz` are not committed (gitignored); the sha256 of the weights and tokenizer is recorded in the run's `passport.json` |
| Mode | eval (dropout off), deterministic forward pass |

Reproducing the weights: the training script is deterministic
(seed 42) and reproduces val_loss ≈ 1.954 on the canonical corpus;
small floating-point deviations between BLAS libraries are permitted
and controlled by comparing `model_revision` (sha256) and the final
training metrics.

## 3. Tasks and independent labeling

Three difficulty bands; each task's provenance is labeled:

| Band | Origin | Count | Expectation |
|---|---|---|---|
| `indist` | pairs from the training corpus (the right-hand side is the library's truth) | 96 (pilot 48 → expanded for the run) | the model partially "knows" the answers |
| `near` | corpus parameters perturbed by ±4–7% | 16 | the interpolation zone |
| `far` | the canonical held-out grid of fresh parameters | 37 | extrapolation, numeric ≈ 0 |

Independent verifier: the truth is **recomputed by the library**
(`src/rmt_llm`: `marchenko_pastur`, `bbp_transition`, `caputo_fractional`,
`tracy_widom`) from the prompt's parameters — never taken from a corpus
string. Corpus pairs whose right-hand side diverges from the
recomputation are excluded with the reason recorded (corpus-drift
detection). Families without a verifier are excluded (logged in
`tasks.json`).

Target variable (primary): `numeric_ok_025` — every numeric field of
the answer within **±25%** of the truth. The sensitivity grid
**[±15%, ±25%, ±40%]** was fixed in advance. Rationale for the
tolerance: at ±15% (the research-benchmark standard) this model
produces almost no positives; ±25% was chosen before the full data were
seen, based on the pilot, in the direction of MORE power — not to fit a
result (the choice makes the predictor's task no easier than ±15%).

Secondary endpoint (declared in advance): `format_ok` — reproducing the
answer's structure (the "Utility Trap": a confident form with unreliable
content).

Generation: **4 sampled answers** (T = 0.8, seed =
crc32(task_id)·100 + i) + **1 greedy** per task; `max_new_tokens = 28`.

## 4. Features (frozen list — 33)

| Group | n | Composition |
|---|---|---|
| `confidence` | 7 | ce_first, ce_mean, ppl_answer, maxprob_first, maxprob_mean, entropy_mean, margin_first |
| `spectral` | 15 | gap_ratio_L0..L3, effrank_L0..L3, gap_ratio_mean, gap_ratio_last, effrank_mean, effrank_last, participation_mean, ncrit_free_last, lambda_max_last |
| `temporal` | 7 | lam_traj_slope/mean/std, ent_traj_slope/delta, ce_traj_slope/std |
| `geometry` | 4 | rep_norm_mean, rep_norm_last, drift_mean, drift_last |

Spectral conventions: the covariance of representations across positions
(T, H); at T < H (rank deficiency) sigma² = trace/H (the full variance),
q = H/T, the MP bound = sigma²·(1+√q)²; `gap_ratio` = λ_max / MP bound.
The convention is verified with control zeros: white noise gives
gap_ratio ∈ [0.85, 1.15] (`tests/test_spectral.py`). The feature order is
fixed in `features/build.py`; observations with non-numeric features are
excluded with the reason recorded.

Model groups:

| Model | Features |
|---|---|
| A | ConstantRateBaseline (train base rate) |
| A2 | FamilyRateBaseline (per-family rate, Laplace smoothing) |
| B | confidence |
| C | internal = spectral + temporal + geometry |
| D | combined = confidence + internal |
| E | combined minus spectral (the "what is lost without the spectrum" ablation) |

Predictor: L2-regularized logistic regression (l2 = 1.0, Newton,
standardization on the train fold) in pure NumPy.

## 5. Splits and leakage checks

Primary scheme `grouped_kfold_v1`: 5 folds grouped by TASK (all repeated
generations of a task stay in one part), stratified by
(family × band), seed 42. Formal checks before every fit: train/test
task overlap is forbidden (`LeakageError`); duplicate prompts across
parts are forbidden. FAMILY overlap in the primary scheme is allowed by
design; the strict generalization control is the separate
`leave_one_family_out` scheme (train on 4 families → test on the 5th).

Platt calibration — on training data only: within each train fold the
tasks are split 80/20 into fit and calibration parts; the test fold
never participates in fitting or calibration.

## 6. Metrics and success criteria

Evaluated jointly (all on pooled out-of-fold predictions):

| Metric | What it shows |
|---|---|
| ROC-AUC (+ clustered 95% CI) | error ranking |
| PR-AUC (average precision) | error detection under imbalance |
| Brier, log loss | probability accuracy |
| ECE (10 equal-mass bins), before and after Platt | calibration |
| ΔAUC(D−B), ΔAUC(D−C), ΔAUC(C−B) | feature-group gains |

Uncertainty: **task-clustered bootstrap** (2000 replicas) — repeated
generations of one task are correlated; percentile CI. Checks:

1. **Clustered paired permutation test** of ΔAUC (2000 permutations):
   a whole task is exchanged between the two models' predictions.
2. **Label permutation** (up to 500 replicas): labels are shuffled at
   the task level within strata (family × band); a check for
   "supernaturally good" AUC against the null "features are unrelated
   to the label".
3. **LOFO**: point AUCs on an unseen family.

**H1 success criterion**: ΔAUC(D−B) > 0, the 95% CI excludes 0, AND the
clustered permutation p < 0.05. Failure criterion: the CI covers 0, or
the effect disappears under LOFO. **A negative result is published as
it is** — it is a useful outcome that bounds the applicability of
spectral features.

## 7. Power and honest limitations

- The 447K model: numeric accuracy ≈ 0 on far; positives concentrate in
  indist. Pilot: numeric_ok ≈ 0 under greedy generation → indist
  expanded to 96 pairs + sampling at T=0.8.
- With fewer than ~20 positives, the primary endpoint is deemed
  **underpowered**: the report states this explicitly, and the
  statistical weight shifts to the pre-declared secondary endpoint
  format_ok.
- One seeded training run; the weights are not committed (reproduction
  is by script); repeated full-pipeline runs are deterministic given the
  same weights.
- The results apply to TinyGPT v3 (447K) and this task class;
  transfer to other models is a separate check (Roadmap, stage 05).

## 8. Run artifacts

`results/h1_v1/`: `passport.json` (protocol_version, sha256 of
weights/corpus/dataset, seeds, environment versions, exclusions) ·
`tasks.json` · `observations.jsonl` (raw observations WITHOUT labels) ·
`labels.jsonl` · `predictions.jsonl` (out-of-fold predictions of all
models) · `report.json` · `REPORT.md` (generated from report.json).

The passport is filled automatically from the real artifacts; placeholder
strings like `"model_revision": "immutable"` are forbidden.

## 9. Protocol changelog

| Version | Change | Rationale |
|---|---|---|
| 1.0 | first freeze | pilot SMOKE run (pipeline debugging, greedy-generation degeneracy); the indist expansion 48→96 and the ±25% tolerance were fixed BEFORE the full run |
| 1.0.1 | (a) numeric labeling is restricted to the answer segment after the last `=>` (the echo of the prompt tail no longer enters the comparison); (b) fixed the secondary endpoint's implementation: predictions for format_ok are fitted SEPARATELY instead of reusing the primary fit; (c) the 015/040 sensitivity targets are also fitted separately | (a) a labeling-operationalization artifact — prompt echo inflated negatives; the direction was conservative, but the labeling must match the benchmark's semantics; (b, c) implementation defects, not fitting: the primary analysis (target, features, models, splits, tests) was not changed |
| 1.1 | generalization of the feature scheme for the size-ladder replication: per-layer names `L0..L{n-1}` are built from the model's actual depth (`science.features.feature_groups(n_layers)`), predictors are assembled per run (`build_models(n_layers)`); the "exactly 4 layers" guard became "≥ 2 layers"; incremental observation checkpoints added (`RMT_H1_OBS_RESUME=1`, computationally identical to a single-shot run — tasks/order/seeds are deterministic). For 4 layers the v1.1 scheme is byte-identical to v1.0 (unit-tested) — the S run was not recomputed | replicating H1 on the M/L cells of the scaling ladder (primary-endpoint power); changes to the feature-assembly mechanics, not the analysis (targets/models/splits/tests unchanged) |
