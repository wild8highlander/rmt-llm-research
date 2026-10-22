<p align="center">
  <img src="../../../assets/logos/sc_pred.banner.svg" width="760" alt="science/predictors — animated logo: target rings with an animated arrow">
</p>

# 🎯 science/predictors — Prediction Layer

<p align="center">
  <em>Pre-registered models and their honest baselines</em><br><br>
  <a href="../README.md"><img src="https://img.shields.io/badge/module-science--predictors-f472b6" alt="predictors"></a>
  <img src="https://img.shields.io/badge/A%E2%80%93E%20models-f472b6" alt="A-E models"><img src="https://img.shields.io/badge/calibrated-a78bfa" alt="calibrated">
</p>

The third layer of the core: the models whose predictions get scored.
The defining discipline is that the model lineup is **pre-registered**
in the hypothesis before the run — A through E below are not a menu
picked after peeking at results, but the fixed cast declared in
[`hypotheses/registry.yaml`](../hypotheses/registry.yaml). Every model
is simple on purpose: with rare positives (~3.4%) and 565 observations,
an over-parameterized predictor would hide the scientific question
behind its own variance.

## Models

| ID | Model | Meaning |
|---|---|---|
| A | constant rate | the floor: always predict the base rate |
| A2 | family rate | base rate conditioned on task family |
| B | confidence | 7 standard confidence features — *the* baseline to beat |
| C | internal | internal-state features (spectral + temporal + geometry) |
| D | combined | B + C — the full predictor |
| E | combined minus spectral | ablation: D without the spectral group |

## Modules

| File | Responsibility |
|---|---|
| `logreg.py` | the regularized logistic regression used for C/D/E (fit identically across cells) |
| `baselines.py` | A, A2, and B — the standards every claim must beat |
| `calibration.py` | Brier score, log loss, calibration error for every model |

## What the ladder showed

The S/M/L replication made one thing visible that a single cell could
not: the **B baseline itself is non-monotonic across scale**
(0.415 → 0.688 → 0.478 AUC), so the D−B difference partially tracks
baseline drift rather than spectral strength. The protocol consequently
reads D against both B and E (and C alone) — the predictors module
provides exactly these comparisons as first-class outputs.

Scored by [`evaluation/`](../evaluation/README.md); the frozen lineup
lives in [`../../EXPERIMENT_PROTOCOL.md`](../../EXPERIMENT_PROTOCOL.md).

---

<p align="center">
  <img src="../../../assets/logos/sc_pred.avatar.svg" width="56" alt="science/predictors animated avatar"><br>
  <sub>
    <a href="../README.md">← science</a> · <a href="../../README.md">← neural_lab</a> · <a href="../hypotheses/README.md">hypotheses →</a>
  </sub>
</p>
