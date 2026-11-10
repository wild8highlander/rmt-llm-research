# RCDA before/after — repetition_loop defect

baseline: `../model` · edited: `../edited_model_v2` · greedy, KV cache on · max_new_tokens=96 · defect: collapse within 80% of max_new OR RepRate-4 >= 0.2

| quantity | baseline | edited |
|---|---|---|
| rho on P_bad (defect rate) | 70.0% (14/20) | 65.0% (13/20) |
| false-trigger rate on P_good | 55.0% (11/20) | 60.0% (12/20) |
| PPL on control texts | 3793.62 | 3662.44 |
| Wilcoxon p (rep4 / distinct4 / collapse) | — | 0.4652 / 0.4652 / 0.1088 |
| changed greedy generations (of 20 bad / 20 good) | — | 5 / 3 |

