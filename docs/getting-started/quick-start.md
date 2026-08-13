# Quick Start

Get from `git clone` to a passing test suite in under 60 seconds.

---

## 1. Install (one-time)

```bash
git clone https://github.com/wild8highlander/rmt-llm-research.git
cd rmt-llm-research
pip install -e ".[dev]"
```

Need details? See [Installation](installation.md).

---

## 2. Run the test suite

```bash
pytest -v                              # 240+ Python tests
make test-tiny-gpt                     # 101 TinyGPT tests only
make test-all                          # all 8 languages (slow!)
```

Expected output (truncated):

```
laboratory/python/lab_en/tests/test_bpe.py ............  [ 11%]
laboratory/python/lab_en/tests/test_tiny_gpt.py ......  [ 31%]
laboratory/python/lab_en/tests/test_trainer.py ........ [ 58%]
src/rmt_llm/tests/test_marchenko_pastur.py ............ [ 78%]
...
============================ 242 passed in 12.41s =============================
```

---

## 3. Verify the math

```python
from rmt_llm.marchenko_pastur import mp_bounds, mp_density
import numpy as np

# Marchenko-Pastur law: eigenvalue density of a random N×T matrix
lambda_minus, lambda_plus = mp_bounds(q=0.5, sigma=1.0)
print(f"MP support: [{lambda_minus:.3f}, {lambda_plus:.3f}]")

# Sample density at a point inside the bulk
rho = mp_density(1.0, q=0.5, sigma=1.0)
print(f"MP density at λ=1.0: {rho:.4f}")
```

Expected:

```
MP support: [0.028, 1.722]
MP density at λ=1.0: 0.4127
```

---

## 4. Train TinyGPT (optional, ~17 min on CPU)

```bash
cd laboratory/python/lab_en
python main.py
```

Choose menu item **14** to train. The trainer will:

1. Fit a BPE tokenizer (256 merges) on the corpus in `corpus/`
2. Run 30 epochs of cosine-LR Adam optimization
3. Save weights to `results/models/tiny_gpt_trained.npz`
4. Save BPE merges to `results/models/tiny_gpt_bpe.json`

Loss trajectory: `6.23 → 0.0085` (730× reduction).

---

## 5. Generate text

Choose menu item **15** from the same `main.py` menu. Provide:

- Weights path: `results/models/tiny_gpt_trained.npz`
- BPE path: `results/models/tiny_gpt_bpe.json`
- Prompt: `def train(`
- Temperature: `0.5`
- Max new tokens: `50`

The model will produce code-like text with real Python keywords (`def`,
`return`, `import`, `Float`, `String`) drawn from the training corpus.

---

## 6. Run a 3D visualization

```bash
cd python/rmt_llm_viz
python main.py                # Interactive menu
python main.py --viz 1        # Marchenko-Pastur 3D density
python main.py --viz 9        # All 8 visualizations
python main.py --save         # Save to PNG (600 DPI)
```

| # | Visualization | What it shows |
|---|---------------|---------------|
| 1 | Marchenko-Pastur 3D | $\rho(\lambda, q)$ surface over the $\lambda$-q plane |
| 2 | BBP Phase Transition | $\lambda_{\max}(\theta, q)$ landscape with critical curve |
| 3 | NHSE Ring Collapse | Eigenvalue ring → skin collapse in the complex plane |
| 4 | EP-Surface Ridge | $\delta\lambda(\varepsilon, k)$ sensitivity ridgeline |
| 5 | Thermodynamic Landscape | $F(U, T, S)$ phase diagram |
| 6 | Tracy-Widom Waterfall | $F_2$ convergence from finite-$N$ to asymptotic |
| 7 | Keating-Snaith Surface | $\gamma_1(N)$ and $N_{\text{crit}}(N)$ correction surfaces |
| 8 | Consistency Dashboard | 10 cross-module verification checks |

---

## 7. Open the Jupyter notebook

```bash
jupyter notebook notebooks/rmt_llm_verification.ipynb
```

Or try the new tutorial notebooks (Sprint 6):

- [`01_quickstart.ipynb`](https://github.com/wild8highlander/rmt-llm-research/tree/main/notebooks/01_quickstart.ipynb)
- [`02_training_tinygpt.ipynb`](https://github.com/wild8highlander/rmt-llm-research/tree/main/notebooks/02_training_tinygpt.ipynb)
- [`03_generation_and_sampling.ipynb`](https://github.com/wild8highlander/rmt-llm-research/tree/main/notebooks/03_generation_and_sampling.ipynb)
- [`04_multilingual_labs.ipynb`](https://github.com/wild8highlander/rmt-llm-research/tree/main/notebooks/04_multilingual_labs.ipynb)

---

## 8. Run benchmarks

```bash
make bench                              # pytest-benchmark
make test-hypothesis                    # property-based tests
```

---

## What's next?

- [First Run](first-run.md) — train TinyGPT step-by-step
- [Architecture](../architecture/index.md) — how the pieces fit together
- [API Reference](../api/index.md) — every public function documented
- [FAQ](faq.md) — common questions and gotchas
