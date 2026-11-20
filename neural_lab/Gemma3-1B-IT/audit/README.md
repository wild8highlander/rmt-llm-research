<p align="center">
  <img src="../assets/logos/gemma-rcda-avatar.svg" width="48" alt="RCDA kit avatar">
</p>

# 🔍 audit — the static RCDA-8 verification

<p align="center">
  <img src="https://img.shields.io/badge/checks-72-00D4FF" alt="checks badge">
  <img src="https://img.shields.io/badge/pass-70-38E1C0" alt="pass badge">
  <img src="https://img.shields.io/badge/fail-1%20(documented)-FF3B6B" alt="fail badge">
  <img src="https://img.shields.io/badge/warn-1-FFB020" alt="warn badge">
  <img src="https://img.shields.io/badge/model%20required-no%20(static%20audit)-B026FF" alt="static badge">
  <a href="../RCDA-9/README.md"><img src="https://img.shields.io/badge/produced%20by-RCDA--9%20round-526CFE" alt="round badge"></a>
  <a href="../README.md"><img src="https://img.shields.io/badge/%E2%86%90-Gemma3--1B--IT-FFB020" alt="parent badge"></a>
</p>

> **Every number the RCDA-8 round prints, checked against the JSONs it
> committed — no model, no GPU, no trust.** The audit re-reads the round's
> README/REPORT claims one by one, recomputes them from
> [`../LlamaFactory/results/`](../LlamaFactory/results/) and reviews the
> training script logic. It shipped as part of the RCDA-9 formula round —
> the "verify the predecessor before you build on it" step.

## The verdict

| Result | Count | Meaning |
|--------|-------|---------|
| PASS | **70** | the claim matches the committed JSON (or the script logic holds) |
| WARN | 1 | 160 log records = 150 steps + 10 bench/service entries — checked manually, unique step numbers = 150 |
| FAIL | 1 | **RCDA-8 REPORT §4.3** claims the LoRA edit's rel-Frobenius is "0.4–1.5 %" while its own `part8_eval_rcda8.json` says **1.35–4.14 %** (median 2.0 %) — the prose understates the edit magnitude ≈3× |

Everything else — the probe replication (z_diff 212.6/219.8/205.1, ρ_bad
0.60, writers L10/11/12), the penalty descent (−97.9 %), the mini-bench
trajectory, the repair counts, KL/PPL, TW ×16, cos(u₁,v) ≤ 0.44, dataset
sizes, adapter configs, the loop-metric implementation — checks out against
the artifacts. The RCDA-9 re-run of the pen arm measured 1.4–4.1 %
rel-Frobenius, i.e. the JSON was right and the prose was wrong.

## The files

| File | What it is |
|------|------------|
| [`rcda8_audit.md`](rcda8_audit.md) | the human-readable protocol — all 72 checks with expected vs actual (Russian, as produced) |
| [`rcda8_audit.json`](rcda8_audit.json) | the machine-readable protocol: `summary` + per-check records |
| [`rcda8_audit_runner.py`](rcda8_audit_runner.py) | the runner — pure static analysis, reads only committed files |

## Re-run it

```bash
# from the repository root; needs numpy only
python3 neural_lab/Gemma3-1B-IT/audit/rcda8_audit_runner.py
```

The runner resolves `../LlamaFactory/results/` relative to its own
location, rewrites the two protocol files next to itself, and exits
non-zero if any check fails.

---

<p align="center">
  <sub><a href="../README.md">← kit root</a> ·
  <a href="../RCDA-9/README.md">the RCDA-9 round →</a> ·
  <a href="../LlamaFactory/RCDA8_REPORT.md">the audited RCDA-8 report →</a></sub>
</p>
