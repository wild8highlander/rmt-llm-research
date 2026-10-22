<p align="center">
  <img src="../../../assets/logos/sc_instr.banner.svg" width="760" alt="science/instrumentation — animated logo: oscilloscope screen with running wave">
</p>

# 🎛️ science/instrumentation — Observation Layer

<p align="center">
  <em>What is actually observed inside the model — with provenance</em><br><br>
  <a href="../README.md"><img src="https://img.shields.io/badge/module-science--instrumentation-00ffa3" alt="instrumentation"></a>
  <img src="https://img.shields.io/badge/sha256%20provenance-22d3ee" alt="sha256 provenance">
</p>

The first layer of the three-layer core: everything here is about
*faithful observation*. The instrumentation opens the NumPy-only model,
runs prompts, and records logits and per-layer hidden states exactly as
they are — it computes no features, knows no targets, and cannot be
tuned to make a hypothesis look better. What it *does* guarantee is
identity: the exact weights observed are pinned by their sha256 into
every downstream passport, so "the model" in a report always means a
specific, verifiable artifact.

## Modules

| File | Responsibility |
|---|---|
| `capture.py` | runs the model over a task set and captures observations: logits, per-layer hidden states, per-layer covariances for the spectral probe |
| `model_loader.py` | loads weights, verifies integrity, records `model_revision` (sha256) into the run passport |
| `__init__.py` | public surface |

## Design rules

- **Observation ≠ interpretation**: capture stores; only
  [`features/`](../features/README.md) computes.
- **No framework opacity**: the model is pure NumPy (ADR-001), so every
  internal state is directly observable without hooks into a framework.
- **Provenance or it did not happen**: an observation set without its
  weight hash is not usable by the core's experiments — the runners
  refuse it.

## Volume notes

A full H1 pass over the 113-task dataset produces 565 observations per
model cell (S: 33 → L: 45 features each after the features layer). Long
captures support resumable checkpoints (`RMT_H1_OBS_RESUME=1`), which is
how the L cell (12+ minutes of observation on 2 CPUs) was completed in
chunks without changing the result: tasks, order, and seeds are
deterministic, so a resumed run is computationally identical to a
single-shot one.

---

<p align="center">
  <img src="../../../assets/logos/sc_instr.avatar.svg" width="56" alt="science/instrumentation animated avatar"><br>
  <sub>
    <a href="../README.md">← science</a> · <a href="../../README.md">← neural_lab</a> · <a href="../data/README.md">data →</a>
  </sub>
</p>
