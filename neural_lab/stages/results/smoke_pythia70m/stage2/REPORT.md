# Stage 2 — Hallucination detector vs baselines

*Run*: `stage2_detector_v1` · hypothesis `S2` · protocol v1.1

## Passport

- model: `EleutherAI/pythia-70m`
- config_hash: `af0d05652f5b0f67`, weight_sha256: `ebfa4e2f18696ebd…`
- n_params: 70,426,624
- transformers 5.22.1, torch 2.15.0+cpu

*code*: `fad992198d24 (dirty)`

## AUROC race (higher = better hallucination detection)

| detector | AUROC | 95% CI | PR-AUC | n |
|---|---|---|---|---|
| sem_ent | 0.500 | [0.500, 0.500] | 0.083 | 12 |
| spec_gap_last | 1.000 | [1.000, 1.000] | 1.000 | 2 |
| spec_ncrit_free | 1.000 | [1.000, 1.000] | 1.000 | 2 |
| tok_last_entropy | 0.692 | [0.357, 1.000] | 0.321 | 16 |
| tok_log_perplexity | 0.205 | [0.000, 0.500] | 0.155 | 16 |
| tok_max_entropy | 0.282 | [0.000, 0.800] | 0.193 | 16 |
| tok_mean_entropy | 0.308 | [0.000, 1.000] | 0.274 | 16 |
| tok_neg_logprob | 0.205 | [0.000, 0.500] | 0.155 | 16 |

- tok_* = token-entropy baselines; sem_ent = semantic entropy (Farquhar et al. 2024);
- probe_hidden = linear probe on last-token hidden states (Azaria & Mitchell protocol, OOF);
- spec_* = RMT spectral detector (gap_ratio / ncrit_free / spectral probe).

## Spectral vs baselines: paired ΔAUROC (clustered bootstrap)

| spectral | baseline | ΔAUROC | 95% CI | p | n_paired |
|---|---|---|---|---|---|
| spec_gap_last | sem_ent | +0.500 | [+0.500, +0.500] | 0.004 | 2 |
| spec_gap_last | tok_last_entropy | +0.000 | [+0.000, +0.000] | 1.000 | 2 |
| spec_gap_last | tok_log_perplexity | +0.000 | [+0.000, +0.000] | 1.000 | 2 |
| spec_gap_last | tok_max_entropy | +0.000 | [+0.000, +0.000] | 1.000 | 2 |
| spec_gap_last | tok_mean_entropy | +0.000 | [+0.000, +0.000] | 1.000 | 2 |
| spec_gap_last | tok_neg_logprob | +0.000 | [+0.000, +0.000] | 1.000 | 2 |
| spec_ncrit_free | sem_ent | +0.500 | [+0.500, +0.500] | 0.004 | 2 |
| spec_ncrit_free | tok_last_entropy | +0.000 | [+0.000, +0.000] | 1.000 | 2 |
| spec_ncrit_free | tok_log_perplexity | +0.000 | [+0.000, +0.000] | 1.000 | 2 |
| spec_ncrit_free | tok_max_entropy | +0.000 | [+0.000, +0.000] | 1.000 | 2 |
| spec_ncrit_free | tok_mean_entropy | +0.000 | [+0.000, +0.000] | 1.000 | 2 |
| spec_ncrit_free | tok_neg_logprob | +0.000 | [+0.000, +0.000] | 1.000 | 2 |

- CI excluding 0 = the difference survives item-clustered resampling.
- n_paired: the Δ was computed only on units carrying BOTH detectors.

## Probe diagnostics

{'skipped': 'insufficient labeled units for probe fitting'}
