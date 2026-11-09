# Stage 2 — hallucination detector

Detector stack on top of the stage-1 spectral features:

- `spectral_features.py` — MP/BBP/TW-based features per layer
- `semantic_entropy.py` — semantic-entropy baseline detector
- `probes.py`, `detectors.py` — trainable probes and the fusion detector
- `benchmarks.py` — evaluation protocol
- `run_stage2.py` — entry point

Docs on extending: [`../docs/ADD_A_DETECTOR.md`](../docs/ADD_A_DETECTOR.md).
Protocol: [`../README.md`](../README.md) and
[`../../EXPERIMENT_PROTOCOL.md`](../../EXPERIMENT_PROTOCOL.md).
