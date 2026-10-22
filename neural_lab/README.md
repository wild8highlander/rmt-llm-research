<p align="center">
  <a href="../README.md"><img src="../assets/branding/aurora-icon.svg" width="72" alt="RMT × LLM Research"></a>
</p>

# 🌐 neural_lab — Web App

<p align="center">
  <em>A Next.js playground where the trained formula model lives</em><br><br>
  <img src="../assets/branding/aurora-divider.svg" width="420" alt=""><br><br>
  <a href="../README.md"><img src="https://img.shields.io/badge/folder-neural--lab-00ffa3" alt="repository root"></a>
  <img src="https://img.shields.io/badge/next.js-000000" alt="Next.js"><img src="https://img.shields.io/badge/typescript-3178C6" alt="TypeScript">
</p>

A web application (Next.js + TypeScript) where the trained model from
[`research/tinygpt_formula/`](research/tinygpt_formula/README.md)
"lives": the playground calls `scripts/infer.py` through the API, while
the other tabs display the training artifacts.

## Science core (`science/`)

Besides the app and the research program, this folder now hosts the
**predictive research core** — a reproducible
`hypothesis → measurement → prediction → validation → falsification`
pipeline (NumPy-only, 52 tests):

| Document | Content |
|---|---|
| [`AUDIT.md`](AUDIT.md) | factual audit: what is really measured, where the synthetic-looking objects are, what was fixed |
| [`EXPERIMENT_PROTOCOL.md`](EXPERIMENT_PROTOCOL.md) | pre-registered H1 protocol (error prediction from internal features) |
| [`PREDICTION_ROADMAP.md`](PREDICTION_ROADMAP.md) | staged plan 01→06 with readiness criteria |
| [`science/results/h1_v1/REPORT.md`](science/results/h1_v1/REPORT.md) | fixed H1 run: primary endpoint underpowered (ΔAUC +0.206 [+0.084, +0.319], p = 0.097); secondary «Utility Trap» endpoint **significant** — internal spectral states AUC 0.871 vs confidence 0.794, p = 0.0025 |
| [`science/results/h1_scaling/REPORT.md`](science/results/h1_scaling/REPORT.md) | H1 × scaling ladder (S/M/L, same 113-task dataset): direction D > B replicated 3/3 cells; L-cell ΔAUC +0.116 [+0.012, +0.215] — bootstrap CI excludes 0; confidence baseline non-monotonic across scale; XL queued (GPU) |

Quick start (after training the model):

```bash
python3 neural_lab/science/experiments/run_h1.py   # full run (~3 min CPU)
python3 -m pytest neural_lab/science/tests/ -q     # 52 tests
```

## Tabs

| Tab | What it shows |
|---|---|
| Overview | project summary: model, corpus, ROADMAP goal |
| Playground | live inference: 8 formula presets, temperature, seed, token count |
| Training | loss / perplexity / match_rate curves from the real training history |
| Benchmark | held-out scores (format / numeric / teacher-forced) + answer samples |
| Questions | the 4 ROADMAP open questions and their empirical answers |
| Deployment | models for fine-tuning, free compute servers, repository bugs found |

## Getting started

```bash
cd neural_lab
npm install        # or bun install
npm run dev        # http://localhost:3000
```

---

<p align="center">
  <sub>
    <a href="../README.md">← root</a> · <a href="README.md">top</a> · <a href="../README.md">↑ parent</a> · <a href="../README.md">🏠 repository root</a> · <a href="research/tinygpt_formula/README.md">the model →</a>
  </sub>
</p>
