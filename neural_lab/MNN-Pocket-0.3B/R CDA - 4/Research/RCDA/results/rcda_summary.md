# RCDA before/after — repetition_loop defect

baseline: `../model` · edited: `../edited_model` · greedy, KV cache on · max_new_tokens=64 · defect: collapse within 80% of max_new OR RepRate-4 >= 0.2

| quantity | baseline | edited |
|---|---|---|
| rho on P_bad (defect rate) | 65.0% (13/20) | 65.0% (13/20) |
| false-trigger rate on P_good | 50.0% (10/20) | 55.0% (11/20) |
| PPL on control texts | 3793.62 | 3779.51 |
| Wilcoxon p (rep4 / distinct4 / collapse) | — | identical (0 diffs) / identical (0 diffs) / identical (0 diffs) |
| changed greedy generations (of 20 bad / 20 good) | — | 0 / 2 |

| edited matrix | TW control | violations | target shift (TW units) |
|---|---|---|---|
| L9 down_proj | pass | 0 | -0.42 |
| L11 o_proj | pass | 0 | -0.27 |

TW control: healthy top singular values must stay within +-2 TW-scale units (plan section 7.2); any component shifted by more than 3 units is a violation.
