# Architecture Decision Records (ADR)

This page collects the major architectural decisions that shaped
`rmt-llm-research`. Each ADR explains *why* a choice was made, not just *what*
was chosen.

---

## ADR-001: NumPy-only, no PyTorch

**Status:** Accepted (2026-01-15)
**Deciders:** Iskhak Hamzatovich Isaev

### Context

The project needs to:

1. Implement reverse-mode autodiff through a 12-layer transformer
2. Be portable across 8 programming languages (Python, Julia, Java, Rust, Go, C++, R, TypeScript)
3. Be readable by a student learning transformer internals

### Decision

Use **NumPy only**. No PyTorch, no TensorFlow, no JAX.

### Rationale

- **Portability:** NumPy is the lingua franca. The same code ports to Julia
  `LinearAlgebra`, Rust `ndarray`, Go `gonum`, C++ Eigen, and R `matrix`
  with minimal changes. PyTorch is Python-only.
- **Pedagogical transparency:** Hand-written autodiff forces the author to
  understand every gradient. PyTorch's `autograd` hides the math.
- **Reproducibility:** No GPU dependency. The model trains in 17 minutes on
  a single CPU core.

### Consequences

- **Positive:** Trivial to port to other languages. Easy to debug.
  No framework version conflicts.
- **Negative:** Training is ~10× slower than PyTorch on GPU. We can't use
  HuggingFace `transformers` ecosystem directly. Some advanced techniques
  (mixed-precision, gradient checkpointing) require more work.

### Alternatives considered

- **PyTorch:** Rejected — breaks portability, hides the math.
- **JAX:** Rejected — Python-only, more complex than NumPy for this use case.
- **CuPy:** Rejected — CUDA-only, doesn't help portability.

---

## ADR-002: Pre-LN transformer blocks

**Status:** Accepted (2026-01-20)

### Context

The original Transformer paper (Vaswani et al., 2017) used **post-LN**:

```text
x = LN(x + attn(x))   # post-LN
x = LN(x + mlp(x))
```

GPT-2 onwards uses **pre-LN**:

```text
x = x + attn(LN(x))   # pre-LN
x = x + mlp(LN(x))
```

### Decision

Use **pre-LN** (GPT-2 style).

### Rationale

- Pre-LN is the modern default — every major LLM since GPT-2 uses it
- Post-LN requires learning-rate warmup to avoid training instability
- Pre-LN allows the residual path to be identity at initialization, which
  stabilizes deep training

### Consequences

- **Positive:** Training is stable without long warmup. The model converges
  reliably to loss < 0.01 in 30 epochs.
- **Negative:** Final LayerNorm is needed before the LM head (otherwise the
  residual stream grows unbounded).

---

## ADR-003: BPE tokenizer (not byte-level, not word-level)

**Status:** Accepted (2026-01-25)

### Context

Three options:

1. **Byte-level** (v1 of TinyGPT): vocab=256, every byte is a token
2. **Word-level**: vocab=variable, every whitespace-delimited word is a token
3. **BPE** (v2 of TinyGPT): vocab=512 (256 bytes + 256 merges)

### Decision

Use **BPE** with 256 merges, total vocab=512.

### Rationale

- **Byte-level (v1):** match_rate=32% looked impressive, but only 82× random
  (1/256 ≈ 0.39%). The model was essentially memorizing byte transitions.
- **Word-level:** requires a tokenizer with OOV handling, doesn't generalize
  to multilingual corpora (Russian text breaks word-level tokenizers).
- **BPE:** captures sub-word structure (`Float`, `String`, `return`) while
  keeping the vocab small. 6.5% match_rate on 512-token vocab is 33× random
  (1/512 ≈ 0.20%) — a harder task with comparable lift.

### Consequences

- **Positive:** Model learns real Python tokens. Multilingual text works.
- **Negative:** match_rate number is not directly comparable to v1.
  The BPE training step adds ~2 seconds to the training pipeline.

---

## ADR-004: Pure-Python BPE (no `tokenizers` library)

**Status:** Accepted (2026-01-25)

### Context

The HuggingFace `tokenizers` library is the industry standard for BPE. It's
Rust-backed, fast, and well-tested.

### Decision

Implement BPE in **pure Python + NumPy**, ~150 lines of code.

### Rationale

- **Transparency:** Students can read the BPE algorithm in 5 minutes.
- **Portability:** The same algorithm is now ported to Julia, Java, Rust,
  Go, C++, and R in the laboratory.
- **No new dependencies:** `tokenizers` adds 50MB of binary wheels.

### Consequences

- **Positive:** Code is readable, portable, dependency-free.
- **Negative:** Training is slower (~2 seconds for 256 merges on 2MB corpus,
  vs. ~50ms for `tokenizers`). Encoding is also slower (~1ms per call vs.
  ~50μs).

---

## ADR-005: One JSON schema for all 16 labs

**Status:** Accepted (2026-02-01)

### Context

The laboratory has 8 languages × 2 locales = 16 implementations. Each could
emit its own output format, but then cross-language comparison becomes a
parsing nightmare.

### Decision

Define a single JSON schema at
[`laboratory/shared/schema.json`](https://github.com/wild8highlander/rmt-llm-research/blob/main/laboratory/shared/schema.json)
and require every lab to emit it.

### Rationale

- Cross-language consistency is enforced by `tests/test_cross_lang_json.py`,
  not by manual inspection.
- New fields can be added without breaking old consumers (schema is
  backward-compatible by default).
- The schema serves as living documentation of what each lab produces.

### Consequences

- **Positive:** Adding a new language port is mechanical — emit the schema.
- **Negative:** Adding a new field requires updating **all 16** labs. The
  EN→RU sync script helps but doesn't help with cross-language sync.

---

## ADR-006: Property-based tests with Hypothesis

**Status:** Accepted (2026-02-10)

### Context

Example-based tests catch the bugs you thought of. Property-based tests
catch the bugs you didn't think of.

### Decision

Add `tests/test_hypothesis.py` with property-based tests for:

- TinyGPT forward pass (shape, finiteness, determinism)
- BPE tokenizer (roundtrip, bounded length, compositionality)
- Softmax / cross-entropy (sums to 1, shift-invariant, non-negative)
- Weight initialization (finite, identity LayerNorm, seed-dependent)
- Generation (length, determinism, vocab bounds)

### Rationale

- Hypothesis generates hundreds of random inputs per test, catching edge
  cases that example-based tests miss.
- The tests are **deterministic** (`@settings(max_examples=...)`), so CI
  is reproducible.
- Property-based tests serve as executable specifications — the test name
  *is* the invariant.

### Consequences

- **Positive:** Caught 3 edge-case bugs during initial development (NaN in
  softmax with large inputs, BPE expanding certain UTF-8 sequences,
  LayerNorm init with seed=0 producing all-zeros).
- **Negative:** Tests are slower (each test runs ~20 examples). Mitigated
  by `max_examples=15` and `deadline=2000ms`.

---

## ADR-007: pytest-benchmark for performance regression testing

**Status:** Accepted (2026-02-10)

### Context

TinyGPT's training time is dominated by the forward + backward pass. A
silent O(n²) → O(n³) regression in attention would slow training 10×
without failing any correctness test.

### Decision

Add `tests/test_benchmark.py` with `pytest-benchmark` for:

- Forward pass (1-layer + 4-layer + batched)
- Softmax + cross-entropy (single + batch)
- BPE fit + encode + decode
- Adam.step() (1k + 100k params)
- End-to-end training step
- Memory-pressure (long sequence)

### Rationale

- `pytest-benchmark` is the standard Python benchmarking tool
- Results can be saved and compared across runs (`--benchmark-compare`)
- The `benchmark.yml` workflow runs nightly on `main` to track drift

### Consequences

- **Positive:** Performance regressions are caught in CI, not in production.
- **Negative:** Benchmarks add ~2 minutes to the `benchmark.yml` workflow.

---

## ADR-008: MkDocs Material for documentation

**Status:** Accepted (2026-02-13)

### Context

The project has substantial documentation: tutorials, API reference,
architecture, FAQ. Three options:

1. **GitHub Pages + static HTML** (v1.6.0): simple but no search, no API
   reference, no navigation
2. **Sphinx + ReadTheDocs**: standard for Python, but reST is verbose
3. **MkDocs Material**: Markdown-native, search, mkdocstrings for API

### Decision

Use **MkDocs Material** with **mkdocstrings** for API reference.

### Rationale

- All existing docs are Markdown — no conversion needed.
- Material theme is modern, fast, mobile-friendly, dark-mode native.
- `mkdocstrings` auto-generates API reference from docstrings.
- Built-in search, tags, git-revision-date, social cards.

### Consequences

- **Positive:** Documentation is now searchable, navigable, and auto-generated.
- **Negative:** Adds `mkdocs-material`, `mkdocstrings[python]`,
  `mkdocs-git-revision-date-localized-plugin`, `mkdocs-minify-plugin` as
  dependencies.

---

## How to add a new ADR

1. Copy this template:

   ```markdown
   ## ADR-NNN: Title

   **Status:** Proposed | Accepted | Rejected | Superseded by ADR-XXX
   **Deciders:** Name(s)

   ### Context
   (Why is this decision needed?)

   ### Decision
   (What was decided?)

   ### Rationale
   (Why this option over alternatives?)

   ### Consequences
   - **Positive:** ...
   - **Negative:** ...

   ### Alternatives considered
   - **Option A:** Rejected because ...
   - **Option B:** Rejected because ...
   ```

2. Append it to this file.
3. Open a PR with the title `docs(adr): ADR-NNN — short title`.
4. Tag at least one maintainer for review.

See [CONTRIBUTING.md](https://github.com/wild8highlander/rmt-llm-research/blob/main/CONTRIBUTING.md)
for more.

---

## ADR-009: Rotary Position Embeddings (RoPE) for TinyGPT v3

**Status:** Accepted (2026-08-14)
**Deciders:** Iskhak Hamzatovich Isaev

### Context

TinyGPT v2 uses **absolute position embeddings** — a learned `(max_seq_len, H)`
matrix added to the token embeddings. This has three limitations:

1. **No length generalization** — the model cannot attend to positions beyond
   `max_seq_len` because the position embedding matrix has no entries there.
2. **Wasted parameters** — `max_seq_len × H = 256 × 128 = 32K` parameters
   that do not transfer across sequence lengths.
3. **No relative position information** — the attention score between tokens
   `i` and `j` depends on their absolute positions, not their relative
   distance.

### Decision

Replace absolute position embeddings with **RoPE** (Rotary Position Embeddings,
Su et al., 2021) in TinyGPT v3. RoPE applies a rotation matrix to the Q and K
vectors in each attention head, where the rotation angle depends on the token
position. This encodes **relative** position directly into the dot product
`Q_i · K_j`.

### Rationale

- **Length generalization** — RoPE works for any sequence length because the
  rotation angles are computed on-the-fly, not stored in a matrix.
- **Parameter savings** — eliminates `max_seq_len × H` parameters. For the 4M
  config, this saves `256 × 192 = 49K` parameters.
- **Industry standard** — RoPE is used by Llama, Mistral, GPT-NeoX, PaLM, and
  most modern LLMs.
- **Pure-NumPy compatible** — the rotation is a simple elementwise operation
  on `(even, odd)` pairs, with no new dependencies.

### Consequences

- **Positive:** Better length generalization. Fewer parameters. Matches the
  Llama architecture, enabling more direct spectral comparisons.
- **Negative:** Slightly more complex attention code. The rotation must be
  applied separately to Q and K (not V). Backward pass requires the inverse
  rotation (which is just the transpose, since rotations are orthogonal).
- **Numerical note:** The rotation is exactly unitary, so it preserves the L2
  norm of each `(even, odd)` pair. This is verified by
  `test_rope_preserves_norm` in `test_tiny_gpt_v3.py`.

### Alternatives considered

- **ALiBi (Attention with Linear Biases)** — adds a linear bias to the
  attention scores based on relative distance. Rejected: the bias is
  additive, not multiplicative, so it interacts differently with the softmax.
- **No position encoding** — the model cannot distinguish token order.
  Rejected.
- **Relative position embeddings (Shaw et al., 2018)** — learned relative
  position embeddings added to the attention scores. Rejected: adds
  parameters and is less elegant than RoPE's closed-form rotation.

---

## ADR-010: Grouped-Query Attention (GQA) for TinyGPT v3

**Status:** Accepted (2026-08-14)
**Deciders:** Iskhak Hamzatovich Isaev

### Context

TinyGPT v2 uses **Multi-Head Attention (MHA)**: `n_heads` query heads, each
with its own K and V projection. For the 4M config (`n_heads=6, hidden=192`),
the K and V projections account for `2 × 192 × 192 = 73K` parameters per
layer. At inference time, the KV cache scales linearly with `n_heads`.

### Decision

Use **Grouped-Query Attention (GQA)** (Ainslie et al., 2023) in TinyGPT v3:
`n_kv_heads = 2` (instead of `n_heads = 6`). Each KV head is shared by
`n_heads / n_kv_heads = 3` query heads. The K and V projections produce only
`n_kv_heads × head_dim = 2 × 32 = 64` features instead of `192`.

### Rationale

- **Parameter savings** — the K and V projections shrink from
  `2 × H × H` to `2 × H × (n_kv_heads × head_dim)`. For the 4M config,
  this saves `2 × 192 × (192 - 64) = 49K` parameters per layer, ~490K
  total across 10 layers.
- **KV cache reduction** — at inference, the KV cache stores `n_kv_heads`
  copies instead of `n_heads`, a 3× reduction for this config.
- **Quality preserved** — Ainslie et al. (2023) showed that GQA with
  `n_kv_heads = n_heads / 4` matches MHA quality on downstream tasks.
- **Llama-2 compatibility** — Llama-2 uses GQA, so this enables a more
  direct spectral comparison between TinyGPT v3 and real Llama models.

### Consequences

- **Positive:** Fewer parameters, smaller KV cache, Llama-compatible
  architecture.
- **Negative:** The forward pass must `np.repeat` K and V across the group,
  adding a small overhead (~8% in benchmarks). The backward pass must
  `np.sum` the gradients across the group.
- **Config constraint** — `n_heads` must be divisible by `n_kv_heads`.
  Enforced in `gqa_forward`.

### Alternatives considered

- **Multi-Query Attention (MQA)** — `n_kv_heads = 1`. Rejected: too
  aggressive quality degradation for a 4M model.
- **MHA** (v2 default) — rejected for this v3 config, but still available
  via `n_kv_heads = n_heads`.
- **Sliding window attention** — orthogonal to GQA; could be combined in a
  future iteration.

---

## ADR-011: Mixed-Precision Training (float16 forward, float32 master)

**Status:** Accepted (2026-08-14)
**Deciders:** Iskhak Hamzatovich Isaev

### Context

Pure-NumPy training on CPU is ~10× slower than PyTorch on GPU (ADR-001).
Mixed-precision (float16 forward pass, float32 master weights) can halve
memory bandwidth and accelerate SIMD operations on modern CPUs.

### Decision

Add an optional `mixed_precision` flag to `TinyGPTV3Config`. When enabled,
the `_w()` helper casts master weights to `float16` before each matmul. The
master weights remain `float32` and are updated by the optimizer in `float32`.
Gradients are accumulated in `float32`.

### Rationale

- **Memory bandwidth** — float16 halves the bytes per weight, so matmuls
  are memory-bound on fewer bytes.
- **Industry standard** — mixed-precision is universal in modern LLM training
  (NVIDIA AMP, PyTorch `torch.cuda.amp`).
- **Optional** — off by default, so the pedagogical float32 path remains
  the default. Users opt in via `TinyGPTV3Config(mixed_precision=True)`.

### Consequences

- **Positive:** ~2× speedup on GPU (not yet benchmarked on this CPU-only
  project). Halves activation memory.
- **Negative:** float16 has limited range (max ~65504) and precision
  (~3 decimal digits). The attention mask value must be `-1e4` (not
  `-1e9`) to avoid overflow. On CPU, float16 is actually **slower** than
  float32 because there is no SIMD benefit (benchmarks show 7× slowdown).
  Mixed-precision is therefore primarily useful for future GPU backends.
- **Numerical stability** — the LayerNorm and softmax computations are
  kept in float16, which may lose precision. The gradient check (float64)
  confirms the math is correct; float16 inference is approximate.

### Alternatives considered

- **bfloat16** — has the same range as float32 but less precision. NumPy
  does not natively support bfloat16 (requires `ml_dtypes` dependency).
  Rejected to maintain the NumPy-only constraint.
- **Full float16 (including master weights)** — rejected: the optimizer
  accumulates in float32 to avoid catastrophic cancellation.

---

## ADR-012: Gradient Checkpointing for Memory Efficiency

**Status:** Accepted (2026-08-14)
**Deciders:** Iskhak Hamzatovich Isaev

### Context

The standard forward-with-cache stores all per-layer intermediates (LN
statistics, attention weights, MLP activations) for the backward pass. For
a 10-layer model with `hidden=192` and `seq_len=256`, this is ~10 × 256 ×
192 × 4 bytes × ~10 intermediates ≈ 20 MB per training step. Scaling to
larger models (4M+ params) or longer sequences is memory-bound.

### Decision

Add an optional `gradient_checkpointing` flag to `TinyGPTV3Config`. When
enabled, `forward_with_cache` stores only the **layer inputs** (one `(T, H)`
array per layer) instead of all intermediates. During `backward`, each
layer's intermediates are **recomputed** by calling
`_forward_layer_with_cache` on the stored input.

### Rationale

- **Memory savings** — stores `n_layers × T × H` instead of
  `n_layers × ~10 × T × H`, a ~10× reduction for the default config.
- **Trade compute for memory** — the forward pass is run twice (once for
  the forward output, once during backward). Benchmarks show a 49%
  backward slowdown, which is the expected cost.
- **Exact gradients** — the recomputed intermediates are mathematically
  identical to the original ones (verified by
  `test_gradient_checkpointing_matches_no_checkpointing` which checks
  gradients match to 1e-10).
- **Standard technique** — used by PyTorch (`torch.utils.checkpoint`),
  JAX (`jax.checkpoint`), and TensorFlow (`tf.recompute_grad`).

### Consequences

- **Positive:** ~10× memory reduction. Enables training larger models or
  longer sequences on the same hardware.
- **Negative:** ~49% backward slowdown (measured). The forward pass is
  unaffected (checkpointing only changes what is cached, not the forward
  computation).
- **Implementation note** — the recomputation is deterministic, so
  gradient checkpointing does NOT introduce any randomness or
  approximation. The gradients are bit-for-bit identical to the
  no-checkpointing path.

### Alternatives considered

- **Activation compression** — quantize activations to int8. Rejected:
  introduces approximation and adds complexity.
- **Layer-parallel backward** — compute gradients for different layers in
  parallel. Rejected: NumPy does not benefit from thread parallelism for
  small arrays.
- **No checkpointing** — the default; users opt in when memory is tight.
