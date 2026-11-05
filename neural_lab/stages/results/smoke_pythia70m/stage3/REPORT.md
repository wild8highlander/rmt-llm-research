# Stage 3 — N_crit from first-error survival

*Run*: `stage3_ncrit_v1` · hypothesis `S3` · protocol v1.1

## Passport

- model: `EleutherAI/pythia-70m`
- config_hash: `af0d05652f5b0f67`, weight_sha256: `ebfa4e2f18696ebd…`
- n_params: 70,426,624
- transformers 5.22.1, torch 2.15.0+cpu

*code*: `fad992198d24 (dirty)`

## First-error survival fit (Weibull scale, tokens)

| quantity | value | 95% CI |
|---|---|---|
| tau (scale) | 2.92 | [2.79, 2.98] |
| gamma (shape) | 10.09 | [6.65, 19.98] |
| mean T_crit = tau*Gamma(1+1/gamma) | 2.78 | — |
| error rate | 100.00% | — |

- gamma ~ 1 = constant hazard (the H1 null A0); gamma >> 1 = accelerating collapse.
- Censored trajectories (never wrong) enter the likelihood as survival terms.

## Breakpoint (the plan's 'izlem')

changepoint at token 2.0 (p_perm=0.966)

- Exact two-rate Poisson split; permutation p over error-label shuffles.

## Error-rate curve

see error_curve.csv (bin centers, at-risk normalized rates)
