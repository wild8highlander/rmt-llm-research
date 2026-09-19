# FAQ

Frequently asked questions about `rmt-llm-research`. Can't find your answer
here? [Open a Discussion](https://github.com/wild8highlander/rmt-llm-research/discussions).

---

## General

### What is this project?

`rmt-llm-research` is a research program applying Random Matrix Theory (RMT)
to the analysis of large language models (LLMs). It collects:

1. **Pure-Python implementations** of every RMT formula referenced in the
   theory (Marchenko-Pastur, BBP transition, Tracy-Widom, NHSE, Caputo
   fractional dynamics, Keating-Snaith, EP-surfaces, thermodynamics).
2. **A pure-NumPy synthetic transformer (`TinyGPT`)** with hand-written
   reverse-mode autodiff, BPE tokenizer, and Adam optimizer — no PyTorch,
   no TensorFlow, no JAX.
3. **An 8-language laboratory** implementing the same interactive menu in
   Python, Julia, Java, Rust, Go, C++, R, and a React web app.
4. **Three 3D visualization suites** (Python CLI, Julia REPL, Java JavaFX)
   producing 8 identical visualizations each.

### Why "RMT meets LLMs"?

Modern LLMs hallucinate, and the question "*why?*" has a precise mathematical
answer. The same critical token count $N_{\text{crit}}$ falls out of ten
independent derivations — from the MP law, the BBP transition, the Tracy-Widom
distribution, the non-Hermitian skin effect, Caputo fractional dynamics,
exceptional-point surfaces, and Keating-Snaith finite-$N$ corrections.

The same mathematics also explains *why* RLHF amplifies the problem (the
"utility trap") and why autoregressive generation is thermodynamically
irreversible.

### Is this a real LLM?

No. `TinyGPT` is a 2.5M-parameter synthetic transformer trained on the
project's own source code. It is **not** a useful language model — its purpose
is to be transparent enough to teach transformer internals, and to provide a
test bed for the spectral diagnostics developed in the theoretical core.

### Who is this for?

- **Researchers** studying LLM behavior through the lens of RMT
- **Students** learning how transformers actually work (the autodiff is by hand)
- **Engineers** porting numerical code across 8 languages
- **Anyone curious** about the math behind hallucinations

---

## TinyGPT

### Why NumPy and not PyTorch?

Two reasons:

1. **Pedagogical transparency.** The goal is for a reader to follow the
   forward and backward passes by hand. PyTorch's autograd hides the math.
2. **Cross-language portability.** The same NumPy code ports to Julia
   `LinearAlgebra`, Rust `ndarray`, Go `gonum`, C++ Eigen, and R `matrix`
   with minimal changes. PyTorch is Python-only.

The trainer is intentionally non-idiomatic — please don't "improve" it by
adding PyTorch; that defeats the point of the project.

### What is `match_rate`?

`match_rate` is the **greedy next-token accuracy** on the held-out 10% of the
corpus. After 30 epochs on the 512-token BPE vocab, the trained model reaches
6.5% — which is 33× the random baseline ($1/512 \approx 0.2\%$).

For comparison, v1 (byte-level, 256-token vocab, 6 layers) reached 32% —
which is 82× its random baseline ($1/256 \approx 0.39\%$). Both lifts are
significant; v2's task is harder because the vocab is larger.

### Why is `match_rate` only 6.5%?

Three reasons:

1. **Small model.** 2.5M params is tiny by modern standards.
2. **Small corpus.** The training data is the project's own source code
   (~2MB), which is much smaller than GPT-2's WebText (40GB).
3. **Small context.** `max_seq_len=32` means the model sees only 32 tokens
   of context — not enough for long-range structure.

To improve `match_rate`, you would need a larger model, a larger corpus,
and a longer context window. But the goal of this project is not to build a
useful LLM — it's to provide a transparent test bed for RMT diagnostics.

### How long does training take?

On a single CPU core (no GPU): **~17 minutes** for 30 epochs on a 2.5M-param
model with `seq_len=32`, `stride=64`, batch size 1.

The trainer supports checkpoint recovery — see `scripts/train_v2_chunk.py`
for an example of running 5 epochs at a time and resuming from a checkpoint.

### Can I train on GPU?

Not currently. The model is pure NumPy, which doesn't support GPU. Porting to
CuPy or JAX would be straightforward but would break the "NumPy-only" design
principle. If you want GPU support, fork the project and replace `np` with
`cupy` (CUDA only) or `jax.numpy` (TPU + GPU).

### How do I generate text?

Two ways:

1. **Interactive:** run `python main.py` (in `laboratory/python/lab_en/`),
   choose menu item **15**, provide weights + BPE paths and a prompt.
2. **Programmatic:** see [Tutorial: Generation & Sampling](../tutorials/03-generation-and-sampling.md).

```python
from tiny_gpt_trainer import load_trained_model, generate_sample

model, tok = load_trained_model(
    "results/models/tiny_gpt_trained.npz",
    "results/models/tiny_gpt_bpe.json",
)
prompt_ids = tok.encode("def train(")
out_ids = generate_sample(model, prompt_ids, max_new_tokens=50, temperature=0.5)
print(tok.decode(out_ids))
```

### The model output looks like gibberish. Is something wrong?

Probably not. With only 30 epochs on a 2MB corpus, the model has learned
*token-level* structure (real Python keywords like `def`, `return`, `import`,
`Float`, `String`) but not *sentence-level* structure. Greedy decoding
(temperature=0) often gets stuck in repetition loops
(`arararars`, `lllll`) — this is expected for an overfit small model.
Try temperature=0.7 or 1.0 for more varied output.

---

## Theory

### What is the Marchenko-Pastur law?

The Marchenko-Pastur (MP) law describes the asymptotic eigenvalue density of
a large random covariance matrix $\frac{1}{T}XX^{\top}$ where $X$ is $N \times T$
with i.i.d. zero-mean unit-variance entries. For aspect ratio $q = N/T \leq 1$,
the eigenvalue support is:

$$\lambda_{\pm} = \sigma^2 (1 \pm \sqrt{q})^2$$

The MP law is the **null hypothesis** for "what LLM activation spectra would
look like if the model were just doing random projections". Deviations from
MP — particularly the BBP transition (a spike outside the bulk) — are the
spectral signature of structured (factual) generation.

### What is the BBP transition?

The Baik-Ben Arous-Péché (BBP) transition is a phase transition in the
largest eigenvalue of a spiked random covariance matrix. When the spike
strength $\theta$ exceeds a critical value $\theta_c = \sqrt{q}$, the largest
eigenvalue $\lambda_{\max}$ pops out of the MP bulk and follows $\theta$:
this is the **signal separation** regime.

In LLM activations, the BBP transition is the spectral marker of "the model
just recalled a fact" vs. "the model is generating creative text".

### What is $N_{\text{crit}}$?

$N_{\text{crit}}$ is the critical token count beyond which autoregressive
generation becomes effectively irreversible. The project derives
$N_{\text{crit}}$ from ten independent mathematical paths:

1. Complex phase of the generating function
2. Operator dynamics (Fokker-Planck eigenvalue decay)
3. GUE spectral statistics
4. Thermodynamic entropy (Landauer)
5. Lévy-Langevin fractional dynamics
6. Exceptional-point surfaces
7. Tracy-Widom fluctuations
8. Quantum channel degradation
9. Non-Hermitian skin effect (winding number)
10. Caputo fractional-time memory

All ten converge on the same $N_{\text{crit}}$ — which is the central claim
of the project.

### What is the "utility trap"?

RLHF (Reinforcement Learning from Human Feedback) optimization creates an
artificial drift in the Fokker-Planck equation, forcing the model to "lie
beautifully once" rather than risk a self-correction cycle. The Caputo
memory parameter $\beta \approx 0.5$ makes
$\langle T_{\text{crit}} \rangle \propto (\mu_{\text{eff}})^{-2}$ — even
small RLHF pressure quadratically accelerates hallucination onset.

---

## Cross-language laboratory

### Why 8 languages?

The same RMT math must be expressible in any language with linear algebra
support. The 8 ports serve as **cross-implementation verification**: if
all 8 implementations produce the same numerical results (within float
tolerance), we have strong evidence that the math is correctly transcribed.

The 8 ports also serve as a Rosetta Stone for developers learning a new
language — you can compare the same algorithm side-by-side in Python,
Julia, Java, Rust, Go, C++, R, and TypeScript.

### How are the 8 ports kept in sync?

Every lab emits the **same JSON schema** (`laboratory/shared/schema.json`).
Cross-language consistency is enforced by `tests/test_cross_lang_json.py`,
which loads all 16 output files (8 languages × 2 locales) and asserts they
all parse against the schema.

The Python EN and RU trainers are kept in lock-step by
`scripts/sync_ru_trainer.py` — do not hand-edit one without syncing the other.

### Can I add a 9th language?

Yes! See [CONTRIBUTING.md → Adding a new language port](https://github.com/wild8highlander/rmt-llm-research/blob/main/CONTRIBUTING.md#adding-a-new-language-port).
The basic recipe:

1. Pick a language with linear algebra support (Swift, Kotlin, Scala, Elixir, Zig, …)
2. Mirror `laboratory/python/lab_en/` — same menu structure, same scenarios
3. Emit `laboratory/shared/schema.json`-compliant JSON
4. Add tests under `laboratory/<lang>/lab_en/tests/`
5. Add the language to `.github/workflows/ci.yml` matrix
6. Add a CODEOWNERS entry
7. Open a PR

---

## Engineering

### How do I run the full CI pipeline locally?

```bash
make install-dev       # one-time
make ci                # runs lint + tests + coverage + cross-checks
```

This mimics what `.github/workflows/ci.yml` does on every push.

### How do I run benchmarks?

```bash
make bench                                # pytest-benchmark
make test-hypothesis                      # property-based tests
```

Results are saved to `laboratory/python/lab_en/.benchmarks/`. Compare
across runs with:

```bash
pytest tests/test_benchmark.py --benchmark-only --benchmark-compare
```

### Where are the trained weights?

Trained weights + BPE merges live at:

```text
laboratory/python/lab_en/results/models/
├── tiny_gpt_trained.npz       # 10.2 MB, 2,543,104 params
├── tiny_gpt_bpe.json          # 3 KB, 256 merges
└── training_history.json      # per-epoch loss/grad_norm/lr/match_rate
```

These are committed to the repo (LFS not needed at this size) so that
anyone can clone and immediately generate text without re-training.

### How do I report a bug?

[Open an issue](https://github.com/wild8highlander/rmt-llm-research/issues/new?template=bug_report.yml)
using the bug report template. Please include:

- OS and Python version
- Output of `pip freeze | grep -E 'numpy|rmt-llm'`
- The exact command you ran
- The full error traceback
- (If possible) a minimal reproducer

### Can I use this in my commercial product?

**No.** The project is under a Proprietary All-Rights-Reserved license.
All rights belong exclusively to Iskhak Hamzatovich Isaev. See
[`LICENSE`](https://github.com/wild8highlander/rmt-llm-research/blob/main/LICENSE)
for full terms. For academic collaboration inquiries, please open a
Discussion or contact the author directly.

---

## Still stuck?

- [Open a Discussion](https://github.com/wild8highlander/rmt-llm-research/discussions)
- [Read CONTRIBUTING.md](https://github.com/wild8highlander/rmt-llm-research/blob/main/CONTRIBUTING.md)
- [Browse the API reference](../api/index.md)
- [File a bug report](https://github.com/wild8highlander/rmt-llm-research/issues/new?template=bug_report.yml)
