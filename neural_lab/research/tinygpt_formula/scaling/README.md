<p align="center">
  <img src="../../../assets/logos/scaling.banner.svg" width="760" alt="scaling — animated logo: rising bar ladder with a flying trend arrow">
</p>

# 📈 scaling — A/B Study

<p align="center">
  <em>How hallucination confidence scales with parameters — formula vs control</em><br><br>
  <a href="../README.md"><img src="https://img.shields.io/badge/folder-research-a78bfa" alt="tinygpt_formula/"></a>
  <img src="https://img.shields.io/badge/study-a78bfa" alt="study"><img src="https://img.shields.io/badge/a%2Fb%20design-22d3ee" alt="A/B design">
</p>

The parameter-scaling study at the heart of the Formula-Edition
experiment: train a ladder of models on the **formula** corpus and its
**control** twin, then measure how confidently-wrong output grows with
size in each arm. The headline artifact is the plot
`hallucination_vs_params.png` — the empirical counterpart of the
theoretical collapse arguments in the papers.

## The ladder

| Rung | Parameters | Layers | Budget note | H1 features |
|---|---|---|---|---|
| S | 447,360 | 4 | 40 epochs, CPU | 33 |
| M | 1,107,712 | 6 | 60 epochs (full spec) | 37 |
| L | 4,055,424 | 10 | 40 epochs, completed cosine | 45 |
| XL | ~12M | 12+ | queued — GPU required | 49 (smoke-verified) |

M was trained to the full ladder specification; L used a shortened but
completed cosine schedule (parity with the S budget — the 150-epoch
ladder targets GPUs and has not been run anywhere). All overrides are
recorded per cell in `runs/<cell>/cell_report.json`, so no rung's
conditions need to be taken on faith.

## Arms and metrics

- **formula** — true repository formulas; **control** — the twin with
  corrupted numerics (seed 42); **untrained** — same-size zero baseline.
- Benchmarks: **HELD-OUT** (37 fresh tasks, extrapolation) and
  **IN-DISTRIBUTION** (up to 60 corpus pairs, knowledge already present).
- Key metric: **confident_hallucination_rate** — share of formatted
  answers (`format_ok=True`) with wrong numbers; the direct empirical
  measure of the Utility Trap.

## What is known so far

At point S the in-distribution arm comparison already shows the
direction predicted by H2: the formula arm reproduces truth more often
(numeric 8.3% vs 3.3%) and hallucinates less often (66.7% vs 81.8%,
−15 pp) where it has knowledge; held-out, both arms hallucinate on every
formatted answer — capacity, not content, is the limit at 447K. The
science core's H1 instrumentation (spectral features per layer) rides
along on every cell, which is how the S/M/L cells of
[`../../../science/results/h1_scaling/REPORT.md`](../../../science/results/h1_scaling/REPORT.md)
were produced by the same runs: one ladder, two research programs.

## Files

| File | Content |
|---|---|
| `scaling_experiment.py` | the ladder runner — train, evaluate, log per run |
| `build_control_corpus.py` | constructs the control corpus twin |
| `scaling_report.json` | machine-readable results for every rung |
| `SCALING_RESULTS.md` | the written analysis and conclusions |
| `runs/` | per-rung artifacts (`history.json`, `cell_report.json`) |
| `TERMUX_PROOT_UBUNTU.md` | how the ladder was run on an Android/Termux device |

## Running a cell

```bash
# S cell locally (~40 epochs CPU)
python3 research/tinygpt_formula/scaling/scaling_experiment.py --cell S

# M/L/XL belong on GPU: use the notebooks in ../colab/
```

Resume chunks make interruptions cheap — each cell restarts from its
last completed chunk instead of from scratch, which is how M/L were
completed on modest hardware.

Start with `SCALING_RESULTS.md`; the corpora come from
[`data/`](../data/README.md).

---

<p align="center">
  <img src="../../../assets/logos/scaling.avatar.svg" width="56" alt="scaling animated avatar"><br>
  <sub>
    <a href="../README.md">← tinygpt_formula</a> · <a href="README.md">top</a> · <a href="../README.md">↑ parent</a> · <a href="../../README.md">🏠 repository root</a> · <a href="../scripts/README.md">pipeline scripts →</a>
  </sub>
</p>
