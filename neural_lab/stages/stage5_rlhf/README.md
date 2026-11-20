# 🎯 stage 5 — the RLHF response

The pressure-response stage: sweeps RLHF-like pressure (preference
optimization strength) against the spectral statistics, producing the
utility-trap evidence curve for the monograph's thermodynamic claims.

| Path | Role |
|------|------|
| `run_stage5.py` | the stage entry point (CLI) |
| [`../results_schema/`](../results_schema/README.md) | the artifact contracts it writes under |
| [`../README.md`](../README.md) | the pipeline overview |

Slow-lane runs (nightly `stages.yml` workflow) cover the real-model
sweep; the offline lane runs the tiny-model property tests from
[`../tests/`](../tests/).

Every written artifact carries `schema_version` and is validated at
write time by `stages.common.schemas`.
