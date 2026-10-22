# H1 — run result (error_prediction_rmt_v1)

Protocol v1.1 · model `4f5f9916524e…` · dataset `411eb0e9bb23…` · seed 42

Observations: **565** (tasks 113), positives: **19** (base rate 3.4%). Primary endpoint: `numeric_ok_025`, metric: `roc_auc`, CI — task-clustered bootstrap.

## Main table (out-of-fold)

| Model | ROC-AUC [95% CI] | PR-AUC | Brier | ECE raw→Platt |
|---|---|---|---|---|
| A_constant_rate | 0.416 [0.276, 0.553] | 0.028 | 0.033 | 0.021 → 0.021 |
| A2_family_rate | 0.486 [0.371, 0.610] | 0.032 | 0.033 | 0.025 → 0.025 |
| B_confidence | 0.688 [0.544, 0.804] | 0.079 | 0.033 | 0.020 → 0.028 |
| C_internal | 0.661 [0.550, 0.763] | 0.067 | 0.034 | 0.027 → 0.027 |
| D_combined | 0.726 [0.616, 0.812] | 0.101 | 0.034 | 0.031 → 0.026 |
| E_combined_minus_spectral | 0.576 [0.434, 0.703] | 0.046 | 0.036 | 0.030 → 0.025 |

## Key comparisons (ΔAUC, clustered tests)

- **D_combined_minus_B_confidence**: +0.037 [-0.042, +0.127], permutation p = 0.5307
- **D_combined_minus_C_internal**: +0.064 [-0.009, +0.130], permutation p = 0.2924
- **C_internal_minus_B_confidence**: -0.027 [-0.162, +0.122], permutation p = 0.7811

## Label permutation (null hypothesis)

- D_combined: observed AUC 0.726, p = 0.0299 (permutable strata 2/11)
  - warning: weak randomization space: most strata are single-class — the p-value is close to 1 and uninterpretable
- B_confidence: observed AUC 0.688, p = 0.0080 (permutable strata 2/11)
  - warning: weak randomization space: most strata are single-class — the p-value is close to 1 and uninterpretable

## Secondary endpoint: format_ok — the “Utility Trap” (separate fit, out-of-fold)

| Model | ROC-AUC [95% CI] | Brier | ECE raw→Platt |
|---|---|---|---|
| A_constant_rate | 0.417 [0.329, 0.500] | 0.094 | 0.044 → 0.044 |
| A2_family_rate | 0.640 [0.568, 0.709] | 0.090 | 0.034 → 0.034 |
| B_confidence | 0.731 [0.665, 0.799] | 0.087 | 0.023 → 0.026 |
| C_internal | 0.757 [0.671, 0.838] | 0.078 | 0.035 → 0.031 |
| D_combined | 0.768 [0.687, 0.843] | 0.078 | 0.036 → 0.046 |

**delta_D_minus_B**: +0.037 [-0.024, +0.096], permutation p = 0.3328

## Tolerance sensitivity (each target fitted separately)

| Tolerance | base rate | B | C | D |
|---|---|---|---|---|
| numeric_ok_015 | 0.023 (13+) | 0.7142 | 0.6629 | 0.712 |
| numeric_ok_025 | 0.034 (19+) | 0.6884 | 0.6614 | 0.7257 |
| numeric_ok_040 | 0.053 (30+) | 0.6893 | 0.6842 | 0.7262 |

## Leave-one-family-out (generalization to an unseen family)

| Family | n_test | n_pos | A | B | C | D |
|---|---|---|---|---|---|---|
| bbp_lambda_max | 315 | 13 | 0.500 | 0.621 | 0.650 | 0.647 |
| caputo_mean_collapse_time | 105 | 0 | — | — | — | — |
| mp_bounds | 135 | 6 | 0.500 | 0.532 | 0.554 | 0.615 |
| tracy_widom_mean | 5 | 0 | — | — | — | — |
| tracy_widom_variance | 5 | 0 | — | — | — | — |

## Verdict on the success criterion (automatically from the protocol)

Primary endpoint UNDERPOWERED: 19 (< 20) — any p-values on the primary endpoint should be read with caution.
ΔAUC(D−B) = +0.037 [-0.042, +0.127], p = 0.5307 → **H1 NOT SUPPORTED: the CI includes 0.**


## Honest limitations

- a single run of a single seeded training; the weights are not committed (see reproduction in EXPERIMENT_PROTOCOL.md);
- repeated generations of one task are correlated — all CIs and tests are clustered by task;
- A_constant_rate below 0.5 is an artifact of between-fold variation of the base rate under a constant prediction; read jointly with A2;
- the small number of positive labels limits power; a negative or inconclusive outcome is still a result and is recorded as it is.

Artifacts: passport.json · tasks.json · observations.jsonl · labels.jsonl · predictions.jsonl · report.json · REPORT.md
