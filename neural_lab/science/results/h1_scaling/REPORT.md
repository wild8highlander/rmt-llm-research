# H1 × scaling ladder — replication on the S / M / L cells

Generated: 2026-10-22 16:58:23. Shared dataset (113 tasks, 565 observations,
hash `411eb0e9bb23bc4b…`), shared split protocol (grouped_kfold_v1),
feature protocol v1.0 (S, 4 layers) / v1.1 (M/L — per-layer features by actual depth).

## The ladder and training quality

| Cell | Parameters | Layers | Epochs | val_loss | val match | Features |
|---|---|---|---|---|---|---|
| S | 447,360 | 4 | 40 | 1.954 | 51.57% | 33 |
| M | 1,107,712 | 6 | 60 | 2.461 | 52.34% | 37 |
| L | 4,055,424 | 10 | 40 | 2.332 | 58.53% | 45 |

M was trained to the full ladder specification (60 epochs); L — with a shortened but
completed cosine schedule (40 epochs, parity with the S budget; the 150-epoch ladder
targets GPUs and has not been run anywhere). Overrides are recorded in cell_report.

## Primary endpoint: numeric error (answer_incorrect)

| Cell | Positives | B confidence | C internal | D combined | E no spectral | ΔAUC(D−B) [95% CI] | perm p |
|---|---|---|---|---|---|---|---|
| S | 14 | 0.415 [0.264,0.561] | 0.542 [0.394,0.674] | 0.621 [0.451,0.766] | 0.412 [0.266,0.552] | **+0.206** [+0.084, +0.319] | 0.097 |
| M | 19 | 0.688 [0.544,0.804] | 0.661 [0.550,0.763] | 0.726 [0.616,0.812] | 0.576 [0.434,0.703] | **+0.037** [-0.042, +0.127] | 0.531 |
| L | 19 | 0.478 [0.367,0.612] | 0.676 [0.537,0.819] | 0.594 [0.456,0.747] | 0.547 [0.413,0.708] | **+0.116** [+0.012, +0.215] | 0.070 |

## Secondary endpoint: format_ok (the “Utility Trap”)

| Cell | B confidence | C internal | D combined | ΔAUC(D−B) [95% CI] | perm p |
|---|---|---|---|---|---|
| S | 0.794 | 0.871 | 0.869 | +0.075 [+0.038, +0.115] (p=0.0025) |
| M | 0.731 | 0.757 | 0.768 | +0.037 [-0.024, +0.096] (p=0.3328) |
| L | 0.721 | 0.846 | 0.858 | +0.137 [+0.080, +0.195] (p=0.0040) |

## What the ladder gives (power and robustness)

1. **Positives**: 14 (S) → 19 (M) → 19 (L) at a ~3.4% base rate — the primary endpoint
   remains rare; the clustered permutation remains conservative.
2. **The effect direction replicates**: D > B in all three cells (3/3;
   binomial sign test p=0.25 — the evidential force is limited).
3. **The strict pre-registered criterion** (perm p < 0.05) is met in no cell;
   in L the bootstrap CI of ΔAUC excludes zero for the first time: +0.116 [+0.012, +0.215].
4. **A new finding — confidence-baseline instability**: B_confidence quality is
   non-monotonic across scale (0.415 → 0.688 → 0.478). The D−B gain is the difference of two
   wandering quantities; part of the ΔAUC dynamics is explained by B drift, not the spectrum.
   Protocol consequence: ΔAUC(D−B) must be complemented by D−E and C-alone comparisons.
5. **Internal states are stable**: C_internal is above chance in every cell
   (0.542 / 0.661 / 0.676) and stably stronger than B on the secondary endpoint in every cell.
6. **Spectral ablation**: E without the spectral group is weaker than D everywhere
   (0.412 / 0.576 / 0.547) — the spectral block carries unique signal at every scale.

## H1 status and next steps

- Registry status: **inconclusive** (revised after L: the direction is stable,
  the effect size wanders due to baseline instability).
- XL: training requires a GPU (an epoch is ~20 min on 2 CPUs; the 150-epoch course is ~49 h); the
  pipeline is smoke-verified at 12L/384h (49 features). The cell is queued.
- For power: expanding the task sample (the goal is ≥ 40 positives per cell), a
  near-band, and a replication on an independent task family (roadmap stage 05).

## Provenance

| Cell | run | model_revision | protocol |
|---|---|---|---|
| S | h1_v1 | `30652295516b…` | 1.0 |
| M | h1_v1_M | `4f5f9916524e…` | 1.1 |
| L | h1_v1_L | `51f72ab690c4…` | 1.1 |

Cell weights are not committed (*.npz is gitignored); reproduction means training the cell
via `scaling_experiment.py` (resume chunks) and running `run_h1.py` with the same seed.
