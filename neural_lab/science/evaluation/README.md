<p align="center">
  <img src="../../../assets/logos/sc_eval.banner.svg" width="760" alt="science/evaluation — animated logo: gauge with sweeping needle">
</p>

# 📏 science/evaluation — Metrics & Statistics

<p align="center">
  <em>How the core decides whether a prediction is actually better</em><br><br>
  <a href="../README.md"><img src="https://img.shields.io/badge/module-science--evaluation-34d399" alt="evaluation"></a>
  <img src="https://img.shields.io/badge/bootstrap%20CI-22d3ee" alt="bootstrap CI"><img src="https://img.shields.io/badge/clustered%20permutation-22d3ee" alt="clustered permutation">
</p>

The statistical conscience of the science core. Every headline number in
the reports — ΔAUC, p-values, intervals — is produced here, under rules
fixed before any run: bootstrap confidence intervals for effect sizes,
cluster-aware permutation tests for significance, and explicit leakage
checks that verify the split policy was respected. The module exists to
make "looks better" falsifiable: a difference counts only when the
statistics say it does.

## Modules

| File | Responsibility |
|---|---|
| `metrics.py` | ROC-AUC, PR-AUC, Brier score, log loss, calibration error — computed identically for every model and baseline |
| `confidence_intervals.py` | 95% bootstrap CIs and clustered permutation tests (observations are permuted in task-family clusters, not independently) |
| `leakage_checks.py` | asserts train/test family disjointness and flags any family overlap before results are trusted |

## Why clustering matters

The 565 observations of the H1 dataset come from 113 tasks — several
observations per task are *not* independent evidence. Treating them as
independent would shrink p-values and manufacture significance. The
permutation tests here therefore shuffle whole task clusters, which is
the conservative choice, and the reason some honest effects (H1 primary
endpoint) report p = 0.097 instead of a triumphant 0.03.

## Reading checklist

1. Effect first: ΔAUC with its bootstrap CI — does the interval exclude 0?
2. Then significance: clustered permutation p — below the pre-registered 0.05?
3. Then robustness: does the direction survive leave-one-family-out?
4. Only then claim anything — and register it in the
   [`hypotheses/`](../hypotheses/README.md) registry.

---

<p align="center">
  <img src="../../../assets/logos/sc_eval.avatar.svg" width="56" alt="science/evaluation animated avatar"><br>
  <sub>
    <a href="../README.md">← science</a> · <a href="../../README.md">← neural_lab</a> · <a href="../predictors/README.md">predictors →</a>
  </sub>
</p>
