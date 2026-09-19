# Design Principles

This page lists the seven design principles that govern new contributions to
`rmt-llm-research`. Read this before making structural changes.

---

## 1. Numerical first, framework never

We use NumPy because it's the lingua franca — no PyTorch, no TensorFlow,
no JAX. The same code ports to Julia `LinearAlgebra`, Rust `ndarray`, Go
`gonum`, C++ Eigen, and R `matrix` with minimal changes.

**Implication:** do not add `import torch` (or similar) to the core package.
If you need autodiff, write it by hand — that's the point of the project.

---

## 2. One JSON schema to rule them all

Every lab, in every language, emits the **same result envelope**. Cross-language
comparison is then a schema-validation check, not a parsing nightmare.

The schema lives at [`laboratory/shared/schema.json`](https://github.com/wild8highlander/rmt-llm-research/blob/main/laboratory/shared/schema.json).
Every new field added to the schema must be:

1. Documented in the schema itself (with description)
2. Emitted by **all 16** lab implementations
3. Tested by `tests/test_cross_lang_json.py`

---

## 3. Pre-LN over Post-LN

TinyGPT uses GPT-2-style pre-LN transformer blocks:

```text
x = x + attn(LN1(x))   # attention sub-layer
x = x + mlp(LN2(x))    # MLP sub-layer
```

This is the modern default and avoids the training instabilities of post-LN
(original Transformer, BERT used post-LN; GPT-2 onwards uses pre-LN).

**Implication:** do not "improve" the model by switching to post-LN. If you
want to experiment with post-LN, fork the project.

---

## 4. Reverse-mode autodiff in plain NumPy

The trainer implements backprop by hand. This is intentionally
non-idiomatic — the goal is pedagogical transparency. Do not "improve" it
by adding PyTorch; that defeats the point of the project.

The hand-written backward pass covers:

- Attention (causal masking)
- LayerNorm (forward + backward through gamma/beta)
- GELU activation
- MLP (2-layer with bias)
- Softmax + cross-entropy
- BPE embedding lookup

If you need to verify a backward pass, see `test_trainer.py` which includes
a **numerical gradient check** that compares analytical gradients to
finite-difference gradients.

---

## 5. Infinite parameters, finite computation

`parameters.py` accepts `"inf"` for any numeric bound. Every consumer is
responsible for clamping to a finite value before computing — see
`max_finite_epochs` in `tiny_gpt_trainer.py`.

**Implication:** never use `int(float("inf"))` — it raises. Use
`max_finite_epochs(epochs)` or similar helpers.

---

## 6. Bilingual by construction

Every lab has `_en` and `_ru` siblings. They are kept in lock-step by
`scripts/sync_ru_trainer.py` — do not hand-edit one without syncing the other.

**Implication:** if you fix a bug in `lab_en`, run `python scripts/sync_ru_trainer.py`
to propagate the fix to `lab_ru`. The sync script translates docstrings,
comments, and print statements.

---

## 7. Tests > docs > comments > types

When in doubt, write a test. Then write a docstring. Then a comment. Type
hints are nice-to-have, not must-have. Ruff enforces style; mypy is advisory.

**Implication:**

- A new function without a test is a bug waiting to happen
- A new function without a docstring is undocumented
- A new function without type hints is acceptable (we'll add them later)

---

## See also

- [Decision Records (ADR)](adr.md) — why these principles were chosen
- [Request Flow](request-flow.md) — how they play out in practice
- [CONTRIBUTING.md](https://github.com/wild8highlander/rmt-llm-research/blob/main/CONTRIBUTING.md) — how to contribute
