# RESULTS_SUMMARY — one evaluator, three arms + the router round

All numbers are fields of committed JSONs (`eval_*.json`, `summary_all_arms.json`) — nothing hand-typed.

## Headline

| arm | F1 phantom ↓ | F3 CER clean | F3 CER degraded | control KL | TW gate |
|---|---|---|---|---|---|
| base | 0.1571 | 0.0089 | 0.0834 | 0.0000 | — |
| classic | 0.0095 | 0.0094 | 0.1001 | 0.0095 | pass ×1 |
| rcda_x | 0.0333 | 0.0095 | 0.1096 | 0.0832 | pass ×3 |

## Per-family

| family | base | classic | rcda_x |
|---|---|---|---|
| F1 phantom rate | 0.1571 | 0.0095 | 0.0333 |
| F1 phantom char rate | 0.2667 | 0.0095 | 0.0476 |
| F1 blank specificity | 1.0 | 1.0 | 1.0 |
| F2 rho_loop | 0.000 | 0.000 | 0.000 |
| F2 mean loop_share | 0.0000 | 0.0000 | 0.0000 |
| F3 CER clean | 0.0089 | 0.0094 | 0.0095 |
| F3 CER degraded | 0.0834 | 0.1001 | 0.1096 |
| F3 CER control | 0.0078 | 0.0114 | 0.0172 |
| F3 exact match clean | 0.8833 | 0.8667 | 0.85 |
| F6 AUC confidence | 0.0961 | 0.001 | 0.0083 |
| F6 AUC geometry (−v·ℓ) | 0.8344 | 0.8575 | 0.9017 |
| control logits MSE | 0.0 | 0.481529 | 1.370705 |

## Spectral section (edited tensors)

- **classic** `mix_fc2.2.weight`: rel_frobenius 0.03441, ΔKS -0.0001 (pass), cos(u₁, v) = 0.3214, σ₁ share 0.4446
- **rcda_x** `mix_fc2.0.weight`: rel_frobenius 0.02896, ΔKS -0.0002 (pass), cos(u₁, v) = 0.7834, σ₁ share 0.7498
- **rcda_x** `mix_fc2.1.weight`: rel_frobenius 0.02239, ΔKS -0.0 (pass), cos(u₁, v) = 0.8486, σ₁ share 0.7329
- **rcda_x** `mix_fc2.2.weight`: rel_frobenius 0.02942, ΔKS 0.0006 (pass), cos(u₁, v) = 0.7888, σ₁ share 0.7279

## Router round (F4 head edit — the 5th 'reject' class)

winner arm **reject_row**, gates pass: True

| metric | base | router_reject |
|---|---|---|
| F4 fit non-text reject rate | 0.0 (no class — 100% routed into recognisers) | **1.0** |
| F4 fit text route acc | 1.0 | 1.0 |
| F4 ability route acc (clean / control) | 0.9667 / 1.0 | 0.9667 / 1.0 |
| F4 ability reject wins | n/a (no class) | 0.0 |
| F4 transfer reject rate (structured/noise/blank) | n/a | **0.9111** (0.8333/0.9/1.0) |
| head edit spectrum | — | λmax/λ₊ 3.71, spikes>2λ₊ 1 (single_spike true), cos(v₁, w_init) 0.5692 |
| export | — | trunk identical True, flip bitwise True, script rows max|Δ| 0.0 |