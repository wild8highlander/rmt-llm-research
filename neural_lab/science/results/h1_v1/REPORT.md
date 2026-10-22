# H1 — run result (error_prediction_rmt_v1)

Protocol v1.0 · model `30652295516b…` · dataset `411eb0e9bb23…` · seed 42

Observations: **565** (tasks 113), positives: **14** (base rate 2.5%). Primary endpoint: `numeric_ok_025`, metric: `roc_auc`, CI — task-clustered bootstrap.

## Main table (out-of-fold)

| Model | ROC-AUC [95% CI] | PR-AUC | Brier | ECE raw→Platt |
|---|---|---|---|---|
| A_constant_rate | 0.320 [0.196, 0.460] | 0.019 | 0.024 | 0.022 → 0.022 |
| A2_family_rate | 0.579 [0.467, 0.691] | 0.029 | 0.024 | 0.018 → 0.018 |
| B_confidence | 0.415 [0.264, 0.561] | 0.024 | 0.025 | 0.025 → 0.026 |
| C_internal | 0.542 [0.394, 0.674] | 0.031 | 0.025 | 0.024 → 0.036 |
| D_combined | 0.621 [0.451, 0.766] | 0.038 | 0.026 | 0.020 → 0.036 |
| E_combined_minus_spectral | 0.412 [0.266, 0.552] | 0.021 | 0.026 | 0.025 → 0.067 |

## Key comparisons (ΔAUC, clustered tests)

- **D_combined_minus_B_confidence**: +0.206 [+0.084, +0.319], permutation p = 0.0970
- **D_combined_minus_C_internal**: +0.079 [+0.002, +0.166], permutation p = 0.0800
- **C_internal_minus_B_confidence**: +0.127 [-0.060, +0.279], permutation p = 0.3878

## Label permutation (null hypothesis)

- D_combined: observed AUC 0.621, p = 1.0000 (permutable strata 0/11)
  - warning: weak randomization space: most strata are single-class — the p-value is close to 1 and uninterpretable
- B_confidence: observed AUC 0.415, p = 1.0000 (permutable strata 0/11)
  - warning: weak randomization space: most strata are single-class — the p-value is close to 1 and uninterpretable

## Secondary endpoint: format_ok — the “Utility Trap” (separate fit, out-of-fold)

| Model | ROC-AUC [95% CI] | Brier | ECE raw→Platt |
|---|---|---|---|
| A_constant_rate | 0.467 [0.407, 0.523] | 0.159 | 0.029 → 0.029 |
| A2_family_rate | 0.598 [0.538, 0.652] | 0.154 | 0.042 → 0.042 |
| B_confidence | 0.794 [0.740, 0.845] | 0.124 | 0.081 → 0.084 |
| C_internal | 0.871 [0.833, 0.905] | 0.103 | 0.028 → 0.040 |
| D_combined | 0.869 [0.828, 0.907] | 0.102 | 0.036 → 0.037 |

**delta_D_minus_B**: +0.075 [+0.038, +0.115], permutation p = 0.0025

## Tolerance sensitivity (each target fitted separately)

| Tolerance | base rate | B | C | D |
|---|---|---|---|---|
| numeric_ok_015 | 0.012 (7+) | 0.4286 | 0.5732 | 0.5466 |
| numeric_ok_025 | 0.025 (14+) | 0.4153 | 0.542 | 0.6212 |
| numeric_ok_040 | 0.055 (31+) | 0.526 | 0.6925 | 0.7181 |

## Leave-one-family-out (generalization to an unseen family)

| Family | n_test | n_pos | A | B | C | D |
|---|---|---|---|---|---|---|
| bbp_lambda_max | 315 | 12 | 0.500 | 0.366 | 0.327 | 0.307 |
| caputo_mean_collapse_time | 105 | 0 | — | — | — | — |
| mp_bounds | 135 | 2 | 0.500 | 0.102 | 0.387 | 0.278 |
| tracy_widom_mean | 5 | 0 | — | — | — | — |
| tracy_widom_variance | 5 | 0 | — | — | — | — |

## Verdict on the success criterion (automatically from the protocol)

Primary endpoint UNDERPOWERED: 14 (< 20) — any p-values on the primary endpoint should be read with caution.
ΔAUC(D−B) = +0.206 [+0.084, +0.319], p = 0.0970 → **H1 NOT STRICTLY SUPPORTED (the CI excludes 0, but the permutation p = 0.097 >= 0.05): the direction is positive, power is insufficient — more data / a larger model is required.**


## Honest limitations

- a single run of a single seeded training; the weights are not committed (see reproduction in EXPERIMENT_PROTOCOL.md);
- repeated generations of one task are correlated — all CIs and tests are clustered by task;
- A_constant_rate below 0.5 is an artifact of between-fold variation of the base rate under a constant prediction; read jointly with A2;
- the small number of positive labels limits power; a negative or inconclusive outcome is still a result and is recorded as it is.

Artifacts: passport.json · tasks.json · observations.jsonl · labels.jsonl · predictions.jsonl · report.json · REPORT.md
