# 🗺️ Roadmap

This document tracks the public roadmap for `rmt-llm-research`. It is
intentionally short — the goal is to communicate **direction**, not commit
to specific dates. Items may be reordered, expanded, or dropped based on
research findings and community feedback.

> **Have an idea?** Open a [Discussion](https://github.com/wild8highlander/rmt-llm-research/discussions)
> or a [feature request](https://github.com/wild8highlander/rmt-llm-research/issues/new?template=feature_request.yml).

---

## 🏗️ Legend

| Marker | Meaning                                                            |
|--------|--------------------------------------------------------------------|
| ✅     | Done — shipped in the latest release                               |
| 🚧    | In progress — actively being worked on                             |
| 📋    | Planned — accepted, but not started                                |
| 💡    | Proposal — under discussion, not yet accepted                      |
| ❌    | Dropped — explicitly rejected (see linked discussion)              |

---

## 🎯 Themes for 2026

1. **Mathematical depth** — extend RMT coverage to keep pace with current literature
2. **Empirical scale** — apply the framework to real GPT-2/Llama activations
3. **Reproducibility** — make every result bit-for-bit reproducible across 8 languages
4. **Pedagogy** — make the codebase a self-contained textbook on RMT × LLMs

---

## ✅ Recently Shipped (recent release)

- ✅ Pre-LN + MLP transformer block (GPT-2 style) for TinyGPT — `tiny_gpt.py`
- ✅ BPE tokenizer (256 merges, pure-Python + NumPy) — `tiny_gpt_trainer.py`
- ✅ Adam with bias correction + weight decay + cosine LR schedule
- ✅ 30-epoch training run: loss 6.23 → 0.0085, match_rate 0% → 6.5%
- ✅ Full reverse-mode autodiff through the synthetic transformer
- ✅ Cross-language 3D research modules: Python, Julia, Java, Rust, Go, C++, R
- ✅ Cross-implementation JSON schema validation (16 labs emit one schema)
- ✅ TinyGPT unit tests + BPE tests + trainer tests (101 tests total)
- ✅ Docker + docker-compose for reproducible research environment
- ✅ Pre-commit hooks (ruff, mypy, codespell, shellcheck, hadolint, gitleaks)
- ✅ GitHub Actions: CI matrix (Py 3.10/3.11/3.12 × ubuntu/macos/windows), CodeQL, release, Docker
- ✅ Community files: issue/PR templates, CODEOWNERS, SECURITY.md, ARCHITECTURE.md, CONTRIBUTING.md

---

## ✅ Shipped in recent release

- ✅ **TinyGPT (modernized)** — modernized transformer (`tiny_gpt_v3.py`)
  - Rotary Position Embeddings (RoPE) — ADR-009
  - Grouped-Query Attention (GQA, `n_kv_heads=2`) — ADR-010
  - Mixed-precision training (float16 forward, float32 master) — ADR-011
  - Gradient checkpointing (~10× memory reduction) — ADR-012
  - 4M-parameter config (`config_4m`): 10 layers, hidden=192, 6 query heads, 2 KV heads
  - Full reverse-mode autodiff verified by float64 finite-difference gradient check
- ✅ **Dyson Brownian Motion module** (`src/rmt_llm/dyson_brownian.py`)
  - DBM simulator for β ∈ {1, 2, 4} with Euler-Maruyama integration
  - Gaussian ensemble sampling (GOE, GUE, GSE)
  - Wigner surmise and level-spacing statistics
- ✅ **Free probability** (`src/rmt_llm/free_probability.py`)
  - Stieltjes transform, Blue transform, R-transform (free cumulants)
  - S-transform for multiplicative free convolution
  - Additive free convolution via subordination fixed-point iteration
- ✅ **Circular ensembles** (`src/rmt_llm/circular_ensembles.py`)
  - COE (β=1), CUE (β=2), CSE (β=4) via Haar-distributed unitaries
  - Spectral form factor, number variance, Wigner surmise
  - Attention phase spectrum extraction
- ✅ **Real GPT-2 activation probe** (`laboratory/python/lab_en/probe_real_gpt2.py`)
  - Loads GPT-2 small (124M) via `transformers` (optional dep)
  - Extracts hidden states, computes covariance spectra against MP bounds
  - Tracks BBP transition across layers; synthetic fallback for CI
- ✅ **Test suite**: 142 new tests (v3 + RMT modules + probe), all passing
- ✅ **Benchmarks**: v2 vs v3 forward/backward/generation (pytest-benchmark)

---

## 🚧 In Progress (next release — upcoming)

- 🚧 **Web dashboard live mode** (`laboratory/webapp`)
  - WebSocket streaming of training loss + grad-norm
  - Real-time spectral gap chart
  - Token-level match-rate heatmap
- 🚧 **TinyGPT (modernized) training run** — train the 4M config on the expanded corpus,
  target match_rate ≥ 25% (up from v2's 6.5%)

---

## 📋 Planned (upcoming)

### Mathematical extensions

- 📋 **Circular ensembles** (COE, CUE, CSE) — `src/rmt_llm/circular_ensembles.py`
- 📋 **Wigner semicircle + Dyson gas** — interactive 3D simulation
- 📋 **Jacobi ensemble** (MANOVA) — for compare-contrast with MP law
- 📋 **Heavy-tailed Lévy ensembles** — for activations with infinite variance
- 📋 **Quaternionic RMT** (β=4) — connection to self-dual matrices

### Empirical scale

- 📋 **Llama-2 / Llama-3 activation probe** — extend the GPT-2 probe to open Llama models
- 📋 **Mistral activation probe** — third open model for cross-architecture comparison
- 📋 **Spectral signature of fine-tuning** — compare base vs SFT vs RLHF checkpoints
- 📋 **TinyGPT 4M model** — scale up to 4M params, train on 10 MB corpus, target match_rate ≥ 25%

### Pedagogy

- 📋 **Jupyter notebook series** (`notebooks/`)
  - `01_marchenko_pastur.ipynb` — MP law from scratch
  - `02_bbp_transition.ipynb` — BBP spike detection
  - `03_tracy_widom.ipynb` — TW distribution and moments
  - `04_nhse.ipynb` — non-Hermitian skin effect on toy models
  - `05_caputo_rlhf.ipynb` — Caputo fractional dynamics of RLHF
  - `06_tiny_gpt_training.ipynb` — end-to-end TinyGPT training walkthrough
- 📋 **Sphinx/MkDocs documentation site** with API reference
- 📋 **Video lectures** (linked from README) — 5 short videos explaining each RMT module

### Infrastructure

- 📋 **PyPI publication** — `pip install rmt-llm` works out of the box
- 📋 **Conda-forge feedstock** — for Anaconda users
- 📋 **GPU support** — optional CuPy backend for `src/rmt_llm/` (off by default)
- 📋 **ONNX export** of TinyGPT for portability across runtimes

---

## 💡 Proposals Under Discussion

- 💡 **Free probability for attention matrices** — apply free convolution to attention
  covariance matrices across heads. May give a sharper N_crit estimate than MP alone.
  [Discussion #N (to be opened)]
- 💡 **Caputo fractional Langevin for SFT dynamics** — extend the Caputo path to
  supervised fine-tuning, not just RLHF. The drift term changes from μ_RLHF to μ_SFT,
  but the memory kernel may be different.
- 💡 **BBP transition as a lying detector** — empirical test: does λ_max cross the
  MP upper bound exactly when the model switches from recall to fabrication?
- 💡 **Stochastic Trace Estimation (Hutchinson)** — for very large activation matrices
  where full eigendecomposition is infeasible.

> **Note:** The RoPE and GQA proposals have been accepted and shipped in recent release
> (see ADR-009 and ADR-010). They are no longer proposals.

---

## ❌ Explicitly Dropped / Out of Scope

- ❌ **PyTorch / TensorFlow backend** — the project is intentionally NumPy-only for
  pedagogical transparency. Frameworks hide the math.
- ❌ **Training on real LLM weights** — we study activations, not weights. Training
  a real LLM is out of scope (and out of compute budget).
- ❌ **Commercial LLM API integration** — no OpenAI, no Anthropic. We use only
  downloadable open-weight models (GPT-2, Llama, Mistral).
- ❌ **Web-scale visualization** — Plotly dashboards are fine; we won't build a SaaS.
- ❌ **Multi-GPU training** — TinyGPT is intentionally small. Real LLM training is
  out of scope.

---

## 🤝 How to Influence This Roadmap

1. **Vote with 👍 on existing proposals** — sort by reactions to see what's popular.
2. **Open a new proposal** as a [Discussion](https://github.com/wild8highlander/rmt-llm-research/discussions) with the `proposal` category.
3. **Submit a PR** against a 📋 or 🚧 item — see [`CONTRIBUTING.md`](https://github.com/wild8highlander/rmt-llm-research/blob/main/CONTRIBUTING.md).
4. **Cite the project** in your paper — see [`CITATION.cff`](https://github.com/wild8highlander/rmt-llm-research/blob/main/CITATION.cff). Academic
   adoption is the strongest signal for prioritization.

---

## 📅 Cadence

- **Minor releases** (e.g. recent release): every 4–6 weeks
- **Patch releases** (e.g. patch releases): as needed for bug fixes
- **Major releases** (e.g. a future release): when breaking API changes are required

The roadmap is reviewed at the start of each minor release cycle.

---

<sub>Last updated: 2026-08-13 · Maintained by
[Iskhak Hamzatovich Isaev](https://github.com/wild8highlander)</sub>
