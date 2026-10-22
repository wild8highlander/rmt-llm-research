# H1 — run result (error_prediction_rmt_v1)

Protocol v1.1 · model `51f72ab690c4…` · dataset `411eb0e9bb23…` · seed 42

Observations: **565** (tasks 113), positives: **19** (base rate 3.4%). Primary endpoint: `numeric_ok_025`, metric: `roc_auc`, CI — task-clustered bootstrap.

## Main table (out-of-fold)

| Model | ROC-AUC [95% CI] | PR-AUC | Brier | ECE raw→Platt |
|---|---|---|---|---|
| A_constant_rate | 0.386 [0.259, 0.535] | 0.027 | 0.033 | 0.027 → 0.027 |
| A2_family_rate | 0.566 [0.425, 0.698] | 0.040 | 0.032 | 0.017 → 0.017 |
| B_confidence | 0.478 [0.367, 0.612] | 0.033 | 0.034 | 0.023 → 0.033 |
| C_internal | 0.676 [0.537, 0.819] | 0.093 | 0.033 | 0.023 → 0.024 |
| D_combined | 0.594 [0.456, 0.747] | 0.056 | 0.036 | 0.033 → 0.032 |
| E_combined_minus_spectral | 0.547 [0.413, 0.708] | 0.044 | 0.036 | 0.031 → 0.036 |

## Key comparisons (ΔAUC, clustered tests)

- **D_combined_minus_B_confidence**: +0.116 [+0.012, +0.215], permutation p = 0.0695
- **D_combined_minus_C_internal**: -0.082 [-0.120, -0.046], permutation p = 0.0005
- **C_internal_minus_B_confidence**: +0.198 [+0.077, +0.306], permutation p = 0.0005

## Label permutation (null hypothesis)

- D_combined: observed AUC 0.594, p = 0.1058 (permutable strata 3/11)
  - warning: weak randomization space: most strata are single-class — the p-value is close to 1 and uninterpretable
- B_confidence: observed AUC 0.478, p = 0.7206 (permutable strata 3/11)
  - warning: weak randomization space: most strata are single-class — the p-value is close to 1 and uninterpretable

## Secondary endpoint: format_ok — the “Utility Trap” (separate fit, out-of-fold)

| Model | ROC-AUC [95% CI] | Brier | ECE raw→Platt |
|---|---|---|---|
| A_constant_rate | 0.453 [0.383, 0.531] | 0.127 | 0.045 → 0.045 |
| A2_family_rate | 0.675 [0.612, 0.734] | 0.117 | 0.031 → 0.031 |
| B_confidence | 0.721 [0.661, 0.779] | 0.115 | 0.029 → 0.039 |
| C_internal | 0.846 [0.789, 0.899] | 0.077 | 0.033 → 0.034 |
| D_combined | 0.858 [0.806, 0.906] | 0.076 | 0.031 → 0.023 |

**delta_D_minus_B**: +0.137 [+0.080, +0.195], permutation p = 0.0040

## Tolerance sensitivity (each target fitted separately)

| Tolerance | base rate | B | C | D |
|---|---|---|---|---|
| numeric_ok_015 | 0.023 (13+) | 0.507 | 0.6002 | 0.555 |
| numeric_ok_025 | 0.034 (19+) | 0.4779 | 0.6759 | 0.5943 |
| numeric_ok_040 | 0.053 (30+) | 0.5477 | 0.6868 | 0.6264 |

## Leave-one-family-out (generalization to an unseen family)

| Family | n_test | n_pos | A | B | C | D |
|---|---|---|---|---|---|---|
| bbp_lambda_max | 315 | 11 | 0.500 | 0.496 | 0.158 | 0.272 |
| caputo_mean_collapse_time | 105 | 0 | — | — | — | — |
| mp_bounds | 135 | 7 | 0.500 | 0.435 | 0.430 | 0.405 |
| tracy_widom_mean | 5 | 0 | — | — | — | — |
| tracy_widom_variance | 5 | 1 | 0.500 | 0.500 | 0.250 | 0.000 |

## Verdict on the success criterion (automatically from the protocol)

Primary endpoint UNDERPOWERED: 19 (< 20) — any p-values on the primary endpoint should be read with caution.
ΔAUC(D−B) = +0.116 [+0.012, +0.215], p = 0.0695 → **H1 NOT STRICTLY SUPPORTED (the CI excludes 0, but the permutation p = 0.069 >= 0.05): the direction is positive, power is insufficient — more data / a larger model is required.**


## Honest limitations

- a single run of a single seeded training; the weights are not committed (see reproduction in EXPERIMENT_PROTOCOL.md);
- repeated generations of one task are correlated — all CIs and tests are clustered by task;
- A_constant_rate below 0.5 is an artifact of between-fold variation of the base rate under a constant prediction; read jointly with A2;
- the small number of positive labels limits power; a negative or inconclusive outcome is still a result and is recorded as it is.

Artifacts: passport.json · tasks.json · observations.jsonl · labels.jsonl · predictions.jsonl · report.json · REPORT.md
