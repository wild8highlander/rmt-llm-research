<p align="center">
  <img src="../../../assets/logos/sc_feat.banner.svg" width="760" alt="science/features — animated logo: spectrum bars with a traveling sine wave">
</p>

# 🌈 science/features — Feature Extraction

<p align="center">
  <em>What is mathematically computed from the model's internals</em><br><br>
  <a href="../README.md"><img src="https://img.shields.io/badge/module-science--features-a78bfa" alt="features"></a>
  <img src="https://img.shields.io/badge/protocol%20v1.1-22d3ee" alt="protocol v1.1"><img src="https://img.shields.io/badge/33%E2%80%9349%20features-22d3ee" alt="33-49 features">
</p>

The second layer of the three-layer core: pure functions from raw
observations (logits, hidden states) to numeric feature vectors. This
layer is deliberately *blind* — nothing here looks at labels, targets,
or correctness, which is what makes it safe to reuse across hypotheses.
The feature protocol is versioned: v1.0 named the 4 layers of the S
model explicitly; v1.1 generalizes to per-layer names `L0..L{n-1}` by
actual depth (byte-identical output for 4 layers, unit-tested), which is
what allowed M (6 layers) and L (10 layers) cells to join the same
study.

## Modules

| File | Group | Content |
|---|---|---|
| `spectral.py` | spectral | λ_max of hidden-state covariances vs the Marchenko–Pastur bound per layer, effective rank, spectral gap |
| `confidence.py` | confidence | standard uncertainty signals: entropy, margin, max-prob |
| `temporal.py` | temporal | layer-to-layer dynamics of the internal state |
| `geometry.py` | geometry | shape metrics of the representation space |
| `build.py` | assembly | builds the protocol-versioned feature vector; validates layer count, errors loudly on 0 valid observations |

## Feature counts by cell

| Cell | Layers | Features |
|---|---|---|
| S | 4 | 33 |
| M | 6 | 37 |
| L | 10 | 45 |
| XL (smoke) | 12 | 49 |

## Ablation contract

The spectral group is the one the hypotheses care about, so the
predictors construct an explicit ablation (`E` — combined minus
spectral) from the same vectors. Across all S/M/L cells the ablation
drops below the full model (0.412 / 0.576 / 0.547 AUC) — the spectral
group carries signal the confidence group does not, which is the core's
main empirical thread so far.

Consumed by [`predictors/`](../predictors/README.md); protocol details
in [`../../EXPERIMENT_PROTOCOL.md`](../../EXPERIMENT_PROTOCOL.md).

---

<p align="center">
  <img src="../../../assets/logos/sc_feat.avatar.svg" width="56" alt="science/features animated avatar"><br>
  <sub>
    <a href="../README.md">← science</a> · <a href="../../README.md">← neural_lab</a> · <a href="../evaluation/README.md">evaluation →</a>
  </sub>
</p>
