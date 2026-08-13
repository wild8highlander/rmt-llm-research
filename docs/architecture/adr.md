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

```
x = LN(x + attn(x))   # post-LN
x = LN(x + mlp(x))
```

GPT-2 onwards uses **pre-LN**:

```
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
