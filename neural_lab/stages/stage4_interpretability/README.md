# 🔬 stage 4 — interpretability

Spectral interpretability of the interventions: writer-layer
identification, phase probes, band statistics and the per-prompt
forensics that connect the RMT picture back to behavior.

| Path | Role |
|------|------|
| [`run_stage4.py`](run_stage4.py) | the stage entry point (CLI) |
| [`../results_schema/`](../results_schema/README.md) | the artifact contracts it writes under |
| [`../README.md`](../README.md) | the pipeline overview |

Offline smoke run (tiny random model, no network):

```bash
python neural_lab/stages/stage4_interpretability/run_stage4.py --help
```

Committed pythia-70m smoke artifacts:
[`../results/smoke_pythia70m/stage4/`](../results/smoke_pythia70m/stage4/)
— every `report.json` schema-validated at write time.
