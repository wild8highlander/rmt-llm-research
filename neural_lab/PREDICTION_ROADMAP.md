# PREDICTION ROADMAP — from a research bench to a predictive laboratory

> The target architecture: `hypothesis → measurement → prediction →
> validation → falsification`. The ordering principle: first prove the
> measurements are correct, then prove they predict something, and only
> then scale. Current state: **stages 01–03 are complete in their first
> working version** (see the checklists); stages 04–06 are next.

The key distinction the whole roadmap stands on:

| Layer | Question | Where in the repository |
|---|---|---|
| Research bench | what do we see? | `research/tinygpt_formula/` |
| Predictive model | what will happen before we know the answer? | `science/` (instrumentation → features → predictors) |
| Scientifically verified theory | does it reproduce on independent data, with falsification? | `science/experiments/` + `science/results/` + the hypothesis registry |

The third does not arise automatically from the first two — that is why
every stage below has a readiness criterion, not a list of files.

---

## Stage 01 — Inventory and audit ✅ (done)

- [x] The run → result path traced through the actual code
- [x] The synthetic-looking misleading wording found and fixed
      (reasoning_trace) — there are no synthetic metrics in the code
- [x] The RMT math verified: the σ² convention defect on rank-deficient
      covariances found and fixed
- [x] The duplication map compiled (gt_pairs/format_ok/numeric_ok/quick_eval)
- [x] Modules labeled "done / partial / hypothesis" (see AUDIT.md §7)

Outcome: `AUDIT.md`, the prioritized backlog, canonicalization into
`science/`.

## Stage 02 — A scientifically sound measurement loop ✅ (done)

- [x] A single interface for loading the real model with provenance
      (`science/instrumentation/model_loader.py`: sha256 of
      weights/tokenizer, eval mode)
- [x] Raw observation capture separated from derived features
      (`instrumentation/capture.py` → `features/`); synthetic data forbidden
- [x] Independent correctness: the truth is recomputed by the
      `src/rmt_llm` library, not taken from the corpus; divergences → exclusion
- [x] Raw data (`observations.jsonl`), features, and metrics stored
      separately; every artifact can be rebuilt from the previous one
- [x] 52 unit/integration tests, including RMT control zeros

**Readiness criterion** (met): every H1 chart/conclusion is
reconstructible from `observations.jsonl` + the fixed conventions.

## Stage 03 — The first real predictor ✅ (done, run fixed)

- [x] Baselines: A (base rate), A2 (per-family rate) — a mandatory
      comparison, not a decoration
- [x] B — standard confidence only; C — internal states only;
      D — combined; E — the no-spectral ablation
- [x] Grouped splits (by task, family×band stratification) +
      leave-one-family-out + formal leakage checks
- [x] Clustered bootstrap CI, paired clustered permutations, label
      permutation with randomization-space diagnostics
- [x] Platt calibration on train data, ECE before/after
- [x] Predictions saved before disclosure: predictions.jsonl → report.json

**The v1.0.1 run's outcome** (`science/results/h1_v1/REPORT.md`):
the primary endpoint (numeric error, 14 positives) — UNDERPOWERED:
ΔAUC(D−B) = +0.206 [+0.084, +0.319], p = 0.097 — the direction is
positive, the strict criterion was not met; the ablation shows the
entire gain comes from the spectrum (D minus spectral = 0.412 ≈ B).
The secondary endpoint (format_ok, the "Utility Trap") — a
**significant result**: internal states 0.871 vs confidence 0.794,
ΔAUC = +0.075 [+0.038, +0.115], p = 0.0025.

**Readiness criterion** (met): the predictive value of every feature
group measured numerically, with CIs and falsification tests; a
negative/inconclusive outcome recorded as it is.

## Stage 04 — Testing the RMT hypotheses ⏳ (partial)

- [x] Control zeros for the spectral conventions (white noise, planted spike)
- [x] The hypothesis registry with prediction/control/falsification
      (`science/hypotheses/registry.yaml`: H1–H4)
- [x] Robustness checks: the indist/near/far difficulty bands + LOFO
- [ ] Fixing the definition of N in N_crit (tokens/sample/dimension/layer) —
      blocks any confirmation claims (H4: `registered`)
- [ ] Moving Q3 (the "lie detector") into a pre-registered protocol with
      dozens of tasks per class
- [ ] Comparison against alternative explanations (e.g. gap_ratio as a
      proxy for norm/context length rather than "knowledge")

**Readiness criterion:** a "supported / not supported / bounded" table in
the registry, each line linked to a fixed run.

## Stage 05 — Transfer across time and models ⏳

- [x] The near/far bands as a mini generalization check; LOFO across families
- [x] The H1 pipeline run on the M/L cells of the scaling ladder
      (`science/results/h1_scaling/REPORT.md`): the same dataset/seed;
      positives 14 → 19 → 19 (not "hundreds" — the ~3.4% base rate
      persists); the direction D > B replicated 3/3, and the L bootstrap
      CI excludes 0 (+0.116 [+0.012, +0.215]); the instability of the
      confidence baseline across scale discovered. XL — queued until a
      GPU is available (an epoch is ~20 min on 2 CPUs).
- [x] Protocol versioning between runs (the passport already stores
      sha256; protocol v1.1 documents the per-layer generalization)
- [ ] Transfer to an unfamiliar task class (new formula families with
      verifiers: keating_snaith, thermodynamics, circular_ensembles)
- [ ] Sequential experiments with pre-registered predictions

**Readiness criterion:** it is stated on which new models/tasks the
prediction survives and where it breaks — with CIs.

## Stage 06 — An automated research system ⏳

- [x] The hypothesis registry (registry.yaml) and the run passport (passport.json)
- [x] Reproducible configurations: one CLI (`experiments/run_h1.py`)
      with a smoke mode; failed/degenerate outcomes are recorded in the report
- [x] Automatic baseline comparison in every table
- [ ] The dataset and run registry (results/index.md, generated)
- [ ] A "Science" tab in the web app from report.json (+ relabeling the
      Q1–Q4 statuses to "empirical / single trajectory")
- [ ] A "prediction → fact" dashboard over the run history

**Readiness criterion:** a new hypothesis travels from a registry entry
to a CI-bearing report with no manual steps outside the CLI.

---

## How progress is judged (readiness criteria)

| Direction | Criterion | Status |
|---|---|---|
| Measurements | features reproducible from the saved raw data | ✅ (H1 artifacts) |
| Math | numerical tests/control zeros confirm the formulas | ✅ (52 tests) |
| Predictions | comparison against simple and strong baselines | ✅ (A/A2/B/C/D/E) |
| Independence | final metrics without test-set fitting | ✅ (OOF + LOFO + permutations) |
| Calibration | probabilities checked on independent outcomes | ✅ (Platt on train, ECE) |
| Generalization | it is stated where the prediction survives | ⏳ (LOFO yes, scale pending) |
| Reproducibility | re-running per the protocol yields the same artifacts | ✅ (determinism + passport) |
| Scientific honesty | negative results are published | ✅ (the underpowered outcome in REPORT.md) |

A universal threshold (like "AUC > 0.9") is deliberately not set in
advance: an acceptable level depends on the base error rate and the cost
of false positives. What matters more is a metric chosen in advance and
a pre-declared gain of practical value (protocol v1.0).

## What to do right now (in order)

1. **Power**: run H1 on the M cell of the scaling ladder (~1.1M
   parameters, Colab T4) — expectation: ×10 positives, giving the strict
   ΔAUC(D−B) criterion real power. *(Status: M and L are done — see
   `science/results/h1_scaling/REPORT.md`; power now hinges on task-set
   expansion to ≥ 40 positives per cell, and XL awaits a GPU.)*
2. **H3 per protocol**: known/fabricated sets of 30+ tasks, a registry
   pre-header, one run — settle the "BBP detector" question honestly.
3. **The Science web tab**: report.json → a table in the app, with the
   Q1–Q4 status labels.
4. **Canonicalization**: the research scripts → import from `science/`.
