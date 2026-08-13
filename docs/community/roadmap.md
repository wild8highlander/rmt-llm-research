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

## Recently shipped (v1.6.0)

- ✅ **Pre-LN + MLP transformer block** (GPT-2 style)
- ✅ **BPE tokenizer** (256 merges, pure-Python + NumPy)
- ✅ **Adam with cosine LR + warmup** (3-epoch warmup, min_lr_ratio=0.1)
- ✅ **30-epoch training** (loss 6.23 → 0.0085, match_rate 0% → 6.5%)
- ✅ **Production-grade engineering infrastructure** (CI matrix, pre-commit, Docker, etc.)
- ✅ **MkDocs Material documentation site** (Sprint 6)
- ✅ **Hypothesis property-based tests** (Sprint 7)
- ✅ **pytest-benchmark performance tests** (Sprint 7)

---

## In progress

- 🚧 **Expand training corpus** — currently ~2MB of project source code.
  Goal: 10MB mixed Python + Russian text for better generalization.
- 🚧 **Russian lab training** — sync `lab_ru` trainer to v2 architecture
  and train a RU-specific model.
- 🚧 **Top-k / top-p sampling** — currently only full-vocab sampling.
  Goal: implement nucleus sampling for better generation quality.

---

## Planned for 2026

### Q1 2026

- 📋 **Mixed-precision training** — optional float16 forward pass (with
  float32 master weights) for 2× training speedup
- 📋 **Gradient checkpointing** — trade compute for memory; allows
  training 4× larger models on the same hardware
- 📋 **Real LLM activation analysis** — extract activations from a real
  GPT-2 model and apply the RMT diagnostics

### Q2 2026

- 📋 **Web playground** — interactive web UI for the 3D visualizations
  (currently CLI/REPL/JavaFX only)
- 📋 **Benchmark suite vs. PyTorch** — side-by-side performance comparison
  to validate that pure-NumPy is "fast enough"
- 📋 **ArXiv submission** — submit the spectral analysis paper to arXiv

### Q3-Q4 2026

- 📋 **Multi-GPU training** (CuPy backend) — for users who want GPU
  acceleration without leaving the NumPy API
- 📋 **9th language port** (Swift or Kotlin) — expand the Rosetta Stone
- 📋 **Book publication** — compile the 6 monographs into a single book

---

## Proposals

These are ideas that haven't been committed to yet. Discussion welcome in
[GitHub Discussions](https://github.com/wild8highlander/rmt-llm-research/discussions).

- 💡 **JIT compilation with Numba** — could speed up the forward pass 5-10×
  without changing the API. Concern: adds a dependency.
- 💡 **ONNX export** — export TinyGPT to ONNX format for interop with
  PyTorch / TensorFlow. Concern: defeats the "NumPy-only" principle.
- 💡 **RLHF sandbox** — implement a minimal RLHF loop to demonstrate the
  "utility trap" theory. Concern: requires a reward model, which is
  out-of-scope for a NumPy-only project.

---

## Dropped

- ❌ **PyTorch backend** — rejected (see [ADR-001](../architecture/adr.md#adr-001-numpy-only-no-pytorch))
- ❌ **Post-LN transformer** — rejected (see [ADR-002](../architecture/adr.md#adr-002-pre-ln-transformer-blocks))
- ❌ **Word-level tokenizer** — rejected (see [ADR-003](../architecture/adr.md#adr-003-bpe-tokenizer-not-byte-level-not-word-level))

---

## How to influence the roadmap

1. **Open a Discussion** with the `roadmap` label
2. **Upvote** existing proposals with 👍
3. **Volunteer** to lead a planned item — we'll mark it as 🚧 and assign you
4. **Propose a new item** — use the `proposal` template

The roadmap is reviewed monthly. Major changes are announced in
[GitHub Discussions](https://github.com/wild8highlander/rmt-llm-research/discussions).
