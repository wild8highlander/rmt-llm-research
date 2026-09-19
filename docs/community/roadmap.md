# Roadmap

This page mirrors [`docs/ROADMAP.md`](https://github.com/wild8highlander/rmt-llm-research/blob/main/docs/ROADMAP.md)
for the docs site. The canonical version is at the repo root.

---

## Status legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Shipped |
| 🚧 | In progress |
| 📋 | Planned |
| 💡 | Proposal (under discussion) |
| ❌ | Dropped (with rationale) |

---

## Recently shipped (recent release)

- ✅ **Pre-LN + MLP transformer block** (GPT-2 style)
- ✅ **BPE tokenizer** (256 merges, pure-Python + NumPy)
- ✅ **Adam with cosine LR + warmup** (3-epoch warmup, min_lr_ratio=0.1)
- ✅ **30-epoch training** (loss 6.23 → 0.0085, match_rate 0% → 6.5%)
- ✅ **Production-grade engineering infrastructure** (CI matrix, pre-commit, Docker, etc.)
- ✅ **MkDocs Material documentation site** (Sprint 6)
- ✅ **Hypothesis property-based tests** (Sprint 7)
- ✅ **pytest-benchmark performance tests** (Sprint 7)

---

## Shipped in recent release

- ✅ **TinyGPT v3** — modernized transformer (`tiny_gpt_v3.py`)
  - Rotary Position Embeddings (RoPE) — [ADR-009](../architecture/adr.md#adr-009-rotary-position-embeddings-rope-for-tinygpt-v3)
  - Grouped-Query Attention (GQA, `n_kv_heads=2`) — [ADR-010](../architecture/adr.md#adr-010-grouped-query-attention-gqa-for-tinygpt-v3)
  - Mixed-precision training (float16 forward, float32 master) — [ADR-011](../architecture/adr.md#adr-011-mixed-precision-training-float16-forward-float32-master)
  - Gradient checkpointing (~10× memory reduction) — [ADR-012](../architecture/adr.md#adr-012-gradient-checkpointing-for-memory-efficiency)
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
- ✅ **Top-k / top-p (nucleus) sampling** — already in `tiny_gpt.py`, now
  also in `tiny_gpt_v3.py` with KV-cache support via `seq_offset`

---

## In progress (next release — upcoming)

- 🚧 **Web dashboard live mode** — WebSocket streaming of training loss
  and grad-norm, real-time spectral gap chart, token-level match-rate
  heatmap.
- 🚧 **TinyGPT v3 training run** — train the 4M config on the expanded
  corpus, target match_rate ≥ 25% (up from v2's 6.5%).
- 🚧 **Expand training corpus** — currently ~2MB of project source code.
  Goal: 10MB mixed Python + Russian text for better generalization.
- 🚧 **Russian lab training** — sync `lab_ru` trainer to v3 architecture
  and train a RU-specific model.

---

## Planned for 2026–2027

### upcoming

- 📋 **Mixed-precision training** — shipped in recent release (ADR-011)
- 📋 **Gradient checkpointing** — shipped in recent release (ADR-012)
- 📋 **Real LLM activation analysis** — shipped in recent release (`probe_real_gpt2.py`)
- 📋 **Llama-2 / Llama-3 activation probe** — extend the GPT-2 probe to
  open Llama models
- 📋 **Mistral activation probe** — third open model for cross-architecture
  comparison
- 📋 **Spectral signature of fine-tuning** — compare base vs SFT vs RLHF
  checkpoints
- 📋 **TinyGPT 4M model** — shipped in recent release as `config_4m()`

### upcoming (later)

- 📋 **Jupyter notebook series** (`notebooks/`)
  - `01_marchenko_pastur.ipynb` — MP law from scratch
  - `02_bbp_transition.ipynb` — BBP spike detection
  - `03_tracy_widom.ipynb` — TW distribution and moments
  - `04_nhse.ipynb` — non-Hermitian skin effect on toy models
  - `05_caputo_rlhf.ipynb` — Caputo fractional dynamics of RLHF
  - `06_tiny_gpt_training.ipynb` — end-to-end TinyGPT training walkthrough
  - `07_tinygpt_v3_modernizations.ipynb` — RoPE, GQA, mixed-precision walkthrough
- 📋 **Sphinx/MkDocs documentation site** with API reference
- 📋 **Video lectures** (linked from README) — 5 short videos explaining
  each RMT module

### Infrastructure

- 📋 **PyPI publication** — `pip install rmt-llm` works out of the box
- 📋 **Conda-forge feedstock** — for Anaconda users
- 📋 **GPU support** — optional CuPy backend for `src/rmt_llm/` (off by default)
- 📋 **ONNX export** of TinyGPT for portability across runtimes
- 📋 **Web playground** — interactive web UI for the 3D visualizations
- 📋 **Benchmark suite vs. PyTorch** — side-by-side performance comparison
- 📋 **ArXiv submission** — submit the spectral analysis paper to arXiv
- 📋 **9th language port** (Swift or Kotlin) — expand the Rosetta Stone
- 📋 **Book publication** — compile the 6 monographs into a single book

---

## Mathematical extensions (planned)

- 📋 **Wigner semicircle + Dyson gas** — interactive 3D simulation
- 📋 **Jacobi ensemble** (MANOVA) — for compare-contrast with MP law
- 📋 **Heavy-tailed Lévy ensembles** — for activations with infinite variance
- 📋 **Quaternionic RMT** (β=4) — connection to self-dual matrices

---

## Proposals

These are ideas that haven't been committed to yet. Discussion welcome in
[GitHub Discussions](https://github.com/wild8highlander/rmt-llm-research/discussions).

- 💡 **Free probability for attention matrices** — apply free convolution to
  attention covariance matrices across heads. May give a sharper N_crit
  estimate than MP alone.
- 💡 **Caputo fractional Langevin for SFT dynamics** — extend the Caputo path
  to supervised fine-tuning, not just RLHF.
- 💡 **BBP transition as a lying detector** — empirical test: does λ_max
  cross the MP upper bound exactly when the model switches from recall to
  fabrication?
- 💡 **Stochastic Trace Estimation (Hutchinson)** — for very large activation
  matrices where full eigendecomposition is infeasible.
- 💡 **JIT compilation with Numba** — could speed up the forward pass 5-10×
  without changing the API. Concern: adds a dependency.
- 💡 **RLHF sandbox** — implement a minimal RLHF loop to demonstrate the
  "utility trap" theory. Concern: requires a reward model.

> **Note:** The RoPE and GQA proposals have been accepted and shipped in
> recent release (see ADR-009 and ADR-010). They are no longer proposals. The ONNX
> export proposal is still under discussion (concern: defeats the "NumPy-only"
> principle).

---

## Dropped

- ❌ **PyTorch backend** — rejected (see [ADR-001](../architecture/adr.md#adr-001-numpy-only-no-pytorch))
- ❌ **Post-LN transformer** — rejected (see [ADR-002](../architecture/adr.md#adr-002-pre-ln-transformer-blocks))
- ❌ **Word-level tokenizer** — rejected (see [ADR-003](../architecture/adr.md#adr-003-bpe-tokenizer-not-byte-level-not-word-level))
- ❌ **Training on real LLM weights** — we study activations, not weights.
- ❌ **Commercial LLM API integration** — no OpenAI, no Anthropic.
- ❌ **Multi-GPU training** — TinyGPT is intentionally small.

---

## How to influence the roadmap

1. **Open a Discussion** with the `roadmap` label
2. **Upvote** existing proposals with 👍
3. **Volunteer** to lead a planned item — we'll mark it as 🚧 and assign you
4. **Propose a new item** — use the `proposal` template

The roadmap is reviewed monthly. Major changes are announced in
[GitHub Discussions](https://github.com/wild8highlander/rmt-llm-research/discussions).
