# 🕵️ stage 2 — the detector

The spectral hallucination detector: probes a model with the bilingual
bad/good prompt batteries, extracts activations, and scores every prompt
with the RMT statistics (loop share, band membership, spectral purity).

| Path | Role |
|------|------|
| [`run_stage2.py`](run_stage2.py) | the stage entry point (CLI) |
| `tests/` (in [`../tests/`](../tests/)) | offline unit + property tests |
| [`../results_schema/`](../results_schema/README.md) | the artifact contracts it writes under |
| [`../README.md`](../README.md) | the pipeline overview |

Run (offline smoke — tiny random model, no network):

```bash
python neural_lab/stages/stage2_detector/run_stage2.py --help
```

Every artifact it writes carries `schema_version` and is validated at
write time; committed smoke results live under
[`../results/smoke_pythia70m/stage2/`](../results/smoke_pythia70m/stage2/).
