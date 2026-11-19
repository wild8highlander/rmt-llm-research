# 🗃️ JSON Results — the raw real-Gemma record

<p align="center">
  <img src="https://img.shields.io/badge/record-real%20Gemma3--1B--IT%20rounds%205%20%C2%B7%206-FF3B6B" alt="record badge">
  <img src="https://img.shields.io/badge/model-unsloth%2Fgemma--3--1b--it%20%40%205b1141-00D4FF" alt="model badge">
  <img src="https://img.shields.io/badge/prompt%20suite-20%20bad%20%C2%B7%2020%20good%20%C2%B7%205%20ctrl-38E1C0" alt="suite badge">
  <a href="https://github.com/hiyouga/LlamaFactory"><img src="https://img.shields.io/badge/next%20round-LlamaFactory%2FRCDA--8-FFB020" alt="rcda8 badge"></a>
  <a href="../README.md"><img src="https://img.shields.io/badge/%E2%86%90-Gemma3--1B--IT-FFB020" alt="parent"></a>
</p>

The measured ground truth of the **real-model** rounds (RCDA-5 transfer
test + RCDA-6 band extension): every number the reports and figures of
[`Reports MD/`](../Reports%20MD/README.md) and [`Graphics/`](../Graphics/README.md)
read is a field of a file in this folder. No hand-typed numbers anywhere.

**Model**: `unsloth/gemma-3-1b-it` @ snapshot `5b1141…` — the same weights
the RCDA-8 LlamaFactory round later trained (its
[`probe_report.json`](../LlamaFactory/results/probe/probe_report.json)
pins the same hash and replicates the probe byte-exact).

## The files

| File | What it records |
|------|-----------------|
| `bad.jsonl` / `good.jsonl` / `control.jsonl` | the **frozen protocol sets** — 20 sentinel-selected bad prompts (`bad_g01…g20`, top-20 by base loop share), 20 good, 5 control (RU+EN); `bad(1).jsonl` etc. are the raw duplicate captures |
| `sentinel_selection.json` | the pre-registered selection: 50 candidates scanned on the base model, top-20 frozen, per-prompt base loop share + collapse point |
| `probe_report.json` | the real-Gemma behavioral probe: per-layer z_diff, looping pool, writers **L10/11/12** |
| `probe_phase_report.json` | the phase probe (RCDA-6): v_early / v_deep per layer, band check, cycle periods |
| `probe_generations.jsonl` / `probe_gens.jsonl` | the frozen greedy generations the probes read |
| `training_log.json` / `training_log_adam.json` | RCDA-5 on the real model: SGD-150 and Adam-150 arms — 150 steps × (pen_norm, mse), pen0, mini-benches at 0/75/150 |
| `part5_eval_{base,trained,adam}.json` | the one evaluator, RCDA-5 arms: ρ_bad 0.60 → 1.00 / 0.95 (the honest negative), per-traj rows |
| `part5_rows_*.jsonl` | per-prompt evaluation rows of the RCDA-5 arms |
| `part6_eval_rcda6_trained.json` / `part6_rows_*.jsonl` | RCDA-6 arm: ρ_bad 0.85, hard-12 0.827 → 0.620, 2/12 repaired |
| `part6_tw_rcda6_trained.json` | Tracy–Widom forensics of the 16-tensor RCDA-6 edit |
| `training_log(1).json` | the RCDA-6 training log (150 SGD steps, phase-matched penalty) |
| `ppl_check.json` / `ppl_check(1).json` | abilities PPL parity, per control text, per arm |
| `rcda5_spectral_digest.json` / `rcda6_spectral_digest.json` | spectral digests: top singular values, u₁ alignment, TW stats |

## Headline numbers (read them straight from the files)

| metric | base | RCDA-5 SGD | RCDA-5 Adam | RCDA-6 SGD |
|--------|------|-----------|-------------|------------|
| ρ_bad | 0.60 | 1.00 | 0.95 | 0.85 |
| loop_share mean (bad) | 0.562 | 0.911 | 0.908 | 0.627 |
| hard-12 loop share | 0.827 | 0.908 | 0.884 | 0.620 |
| prompts fully repaired | — | 0/12 | 0/12 | 2/12 |
| KL(base‖arm) | 0 | 0.221 | 0.025 | 0.307 |
| PPL control | 10.9 | 10.2 | 11.0 | 14.1 |

The continuation of this record — the formula re-implemented as a LoRA
(3/12 repaired) against a data arm (5/12, ρ_bad 0.40) — lives in
[`../LlamaFactory/results/`](../LlamaFactory/README.md).
