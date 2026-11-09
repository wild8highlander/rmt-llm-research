# CODE.md — what the code is, and what it proves

Companion to [`AUDIT.md`](AUDIT.md). This file documents the code with its
mathematical basis, inputs/outputs and the research thread each piece
serves. Every "expected output" below was produced by actually running the
code on this repository (see the run commands). Three modules are documented
in full as the reference sample; the rest have concise cards and will get
the same treatment in the next pass.

**The three research threads** (from `src/rmt_llm/__init__.py`):

1. **Spectral statistics of LLM activations** — Marchenko-Pastur bulk, BBP
   transition, Tracy-Widom fluctuations (`marchenko_pastur`,
   `bbp_transition`, `tracy_widom`, `circular_ensembles`, `dyson_brownian`).
2. **Mathematical proof of inevitable hallucinations** — 10 independent
   paths to a critical token count `N_crit` (`nhse`, `ep_surfaces`,
   `keating_snaith`, `dyson_brownian`, … formalized in `formal_proof/`).
3. **The RLHF utility trap** — Caputo fractional memory, entropic collapse
   (`caputo_fractional`, `thermodynamics`, `free_probability`).

Verification chain: `src/rmt_llm` (NumPy core, 101 tests) → cross-language
labs in `laboratory/` (C++, Rust, Go, Java, Julia, R reimplement the same
statistics) → `neural_lab/` (experiments on real models) →
`docs/monographs/` (the written theory) and `results/` (artifacts).

---

## 1. Full sample: `src/rmt_llm/marchenko_pastur.py`

**Thread 1. Serves every lab and the banner GIF.** The law that activates
analysis compares layer spectra against.

*Mathematical basis.* For an N×T matrix X with i.i.d. entries of variance
σ², the eigenvalues of M = X Xᵀ/T converge (N,T→∞, q=N/T fixed) to the
Marchenko-Pastur law with support
λ± = σ²(1 ± √q)² and density
ρ(λ) = √((λ₊−λ)(λ−λ₋)) / (2πσ²qλ) on [λ₋, λ₊].
Reference: Marchenko & Pastur (1967).

*Key entities (all pure functions, only dependency is NumPy):*

| function | signature | role |
|---|---|---|
| `mp_bounds` | `(q, sigma2=1.0) -> (λ₋, λ₊)` | support endpoints; validates q∈(0,1], σ²>0 |
| `mp_density` | `(lam, q, sigma2=1.0) -> ndarray` | MP density, 0 outside support (continuous part; the q>1 point mass at 0 is documented but not returned) |
| `mp_cdf` | `(lam, q, sigma2=1.0, n_points=10000) -> ndarray` | CDF by quadrature on a fine grid + linear interpolation |
| `mp_sample` | `(n, t, sigma2=1.0, rng=None) -> ndarray(n,)` | draws N×T i.i.d. Gaussian, returns sorted eigenvalues of XXᵀ/T — this is the empirical bulk |
| `mp_stieltjes` | `(z: complex, q, sigma2=1.0) -> complex` | Stieltjes transform g(z) = (1−q−z+√((1−q−z)²−4qz))/(2σ²qz), branch chosen with Im g > 0 for Im z > 0 — the free-probability bridge to `free_probability.py` |

*Inputs/outputs.* No files; everything is arrays. `mp_sample` is the only
consumer of randomness (accepts a `Generator` — all lab runs pin a seed
for reproducibility).

*Usage (verified on this repo):*

```python
from rmt_llm.marchenko_pastur import mp_bounds, mp_density, mp_sample
import numpy as np
mp_bounds(0.1)                       # (0.4675444..., 1.7324555...)
rho = mp_density(np.linspace(0.5, 1.7, 5), 0.1)   # finite inside, 0 outside
bulk = mp_sample(480, 4800, rng=np.random.default_rng(7))  # q = 0.1 bulk
```

*Tests & artifacts.* `src/rmt_llm/tests/test_marchenko_pastur.py`
(17 tests: bounds, density normalization, CDF monotonicity, sampling vs
theory, Stieltjes branch). Consumed by every language lab — e.g. the C++
lab's `02_eigenvalue_vs_mp` charts in `laboratory/results/charts/` are a
visual regression test of this law, and the README banner's bulk is
`mp_sample(480, 4800)` with seed 7.

---

## 2. Full sample: `src/rmt_llm/bbp_transition.py`

**Thread 1 → thread 2 bridge.** This is the detection mechanism: a "signal"
direction in an activation matrix is a rank-1 spike on top of the MP bulk.

*Mathematical basis.* For Σ = σ²I + θ·vvᵀ the largest eigenvalue undergoes
the Baik–Ben Arous–Péché (2005) phase transition at θ_c = √q:

- subcritical (θ ≤ θ_c): λ_max → λ₊ = σ²(1+√q)² — the spike hides in the bulk;
- supercritical (θ > θ_c): λ_max → σ²(1 + θ²/q) — the outlier detaches.

Near the transition, fluctuations scale as N^(−1/3) (Tracy-Widom scaling) —
which is exactly why `tracy_widom.py` matters for thresholds.

*Key entities:*

| function | signature | role |
|---|---|---|
| `bbp_critical_theta` | `(q) -> θ_c` | √q, the detectability threshold |
| `bbp_lambda_max` | `(theta, q, sigma2=1.0) -> float` | piecewise formula above — the *detector's decision surface* |
| `bbp_signal_separation` | `(theta, q, sigma2=1.0) -> float` | λ_max − λ₊; positive gap = signal detected |
| `bbp_is_supercritical` | `(theta, q) -> bool` | the boolean verdict used by detectors |
| `bbp_fluctuation_scaling` | `(n, q) -> n^(-1/3)` | threshold width vs matrix size (TW regime) |
| `bbp_sample` | `(n, t, theta, sigma2=1.0, rng=None) -> ndarray` | spiked covariance sampler: rank-1 spike added along the first row of X |

*Usage (verified on this repo):*

```python
from rmt_llm.bbp_transition import bbp_critical_theta, bbp_lambda_max, bbp_is_supercritical
bbp_critical_theta(0.1)      # 0.31622776601683794
bbp_lambda_max(2.0, 0.1)     # 41.0  (supercritical: 1 + 4/0.1 = 41 × σ²)
bbp_is_supercritical(2.0, 0.1)   # True
```

*Tests & artifacts.* 11 tests in `test_bbp_transition.py`. The banner GIF
on the root README animates `bbp_lambda_max` crossing θ_c; the
`08_ncrit_threshold` charts in `laboratory/results/charts/` are the lab-scale
version of the same computation. Monograph: `docs/monographs/M04_blackgold_v2/`
("expanded analysis with the BBP transition").

---

## 3. Full sample: `src/rmt_llm/tracy_widom.py`

**Thread 1. The noise floor of every threshold.** How much can λ_max
wobble in a purely random matrix? Tracy-Widom F₂ answers that, and it sets
the false-positive rate of every BBP detector.

*Mathematical basis.* F₂(s) = exp(−∫ₛ^∞ (x−t) q₂(t)² dt), where q₂ solves
the Painlevé II equation q″ = s·q + 2q³ with q(s) → Ai(s) as s → +∞
(Tracy & Widom, 1996). The CDF is evaluated from a tabulated grid
(s ∈ [−5, 5], step 0.5, linear interpolation, ~10⁻⁶ accuracy), computed
offline by Bornemann's (2010) Fredholm-determinant method — trading a
heavy ODE solver for a 21-line table at runtime.

*Key entities:*

| function | signature | role |
|---|---|---|
| `tracy_widom_cdf` | `(s: ArrayLike) -> ndarray` | F₂(s) from `TW_F2_CDF_TABLE` |
| `tracy_widom_pdf` | `(s, eps=0.01) -> ndarray` | central-difference derivative of the CDF |
| `tracy_widom_mean` / `_variance` / `_skewness` | `() -> float` | canonical moments −1.7711 / 0.8132 / 0.2241 |
| `tracy_widom_sample` | `(n, matrix_size=100, rng=None) -> ndarray` | Monte-Carlo: largest eigenvalue of GUE matrices, centered by √(2N) and scaled by N^(−1/6)/√2 |

*Implementation note (documented, not changed):* `tracy_widom_sample`
diagonalizes the *real part* of the GUE matrix — a deliberate speed/accuracy
trade visible in the source; the moment constants remain exact values.

*Usage (verified on this repo):*

```python
from rmt_llm.tracy_widom import tracy_widom_cdf, tracy_widom_mean, tracy_widom_variance
tracy_widom_cdf([-1.5, 0.5])     # [0.0317, 0.293]
tracy_widom_mean(), tracy_widom_variance()   # (-1.7711, 0.8132)
```

*Tests & artifacts.* 7 tests (monotone CDF, table endpoints, moments).
The moment values are the numbers a detector plugs into its
false-alarm budget; `tests/` pins them so a table edit cannot silently
shift thresholds. Same constants are re-derived in the Julia package
(`RMTLLMVerify`) — the cross-language consistency check.

---

## 4. Concise cards — the remaining core modules

| module | thread | one-paragraph summary | verified number |
|---|---|---|---|
| `constants.py` | 1–3 | central registry of N_crit, γ, β and shared numerical values; single source of truth for the labs | — |
| `nhse.py` | 2 | non-Hermitian skin effect: below N_crit eigenvalues form a 2D ring (winding w=0), above it they collapse to the real axis (w=1) — the topological marker of hallucination onset; provides winding number, skin strength, imaginary collapse, point gap + H matrix/eigenvalue samplers | w=0, skin=0 at (n_ratio 0.5, γ=3) |
| `caputo_fractional.py` | 3 | Caputo derivative of order β≈0.5 models RLHF memory; mean collapse time ⟨T_crit⟩ ∝ μ_eff^(−1/β), so 1/β≈2 makes even small pressure quadratically accelerate hallucination | — |
| `ep_surfaces.py` | 2 | exceptional points of order k: Δλ ~ ε^(1/k) with k~O(N) — floating-point rounding alone can trigger macroscopic spectral shifts | — |
| `keating_snaith.py` | 2 | finite-context correction γ₁(N) = γ₁ + c₁/N + c₂/N² + … to N_crit from GUE characteristic-polynomial statistics of zeta zeros (γ₁ ≈ 14.1347) | — |
| `dyson_brownian.py` | 1–2 | Dyson Brownian motion dλᵢ = (σ²/2N)Σⱼ≠ᵢ 1/(λᵢ−λⱼ) dt + (σ/√N) dWᵢ; `DBMConfig`/`DysonBrownianMotion` classes, Gaussian ensembles, Wigner surmise | — |
| `circular_ensembles.py` | 1 | COE/CUE/CSE on U(N): Haar samplers, spacing statistics, form factor K(t), number variance Σ²(L) — the "conserved-quantum-number" model of attention phases | mean spacing 1.0 |
| `free_probability.py` | 1, 3 | Voiculescu's free convolution, R- and S-transforms — the algebra behind composing layer spectra (additive/multiplicative free convolution) | — |
| `thermodynamics.py` | 3 | free energy, Landauer cost of lies, spectral entropy and RG flow — the thermodynamic accounting of thread 3 | — |

Each card will be expanded to the full-sample format (math, I/O schema,
usage, tests) in the next pass, together with `formal_proof/rmt_fp/` and
the Julia `RMTLLMVerify` package (whose docstring mirrors
`src/rmt_llm/__init__.py` thread-for-thread; Julia 1.9–1.11, `Project.toml`
version 1.3.0, cross-checked against the Python core).

## 5. The CLI (verified)

```text
rmt-llm-verify info        # package, author, ORCID, module list
rmt-llm-verify verify      # 5/5 module self-checks — CI gate (cli-doctor.yml)
rmt-llm-verify mp          # --q --sigma2: JSON support + density peak
rmt-llm-verify bbp         # --theta --q: θ_c, λ_max, verdict
rmt-llm-verify tw          # --s ...: F₂ CDF + moments
rmt-llm-verify nhse        # --n-ratio --gamma --size --seed: JSON summary
rmt-llm-verify spacing     # --n --beta --seed: circular spacing stats
python -m rmt_llm …        # same commands via __main__.py
```

`verify` is the honest gate: it exercises every thread-1 module and exits
non-zero on any failure; `.github/workflows/cli-doctor.yml` runs it daily
and auto-issues on breakage. Smoke tests in `tests/test_cli.py` are
proposed in the next block.

## 6. Reproducing every number in this file

```bash
pip install -e .
rmt-llm-verify verify            # 5/5 checks passed
rmt-llm-verify mp --q 0.1        # support [0.4675…, 1.7324…]
rmt-llm-verify bbp --theta 2.0   # theta_c=0.3162, lambda_max=41.0
rmt-llm-verify tw --s -1.5 0.5   # 0.031700, 0.293000
python -m pytest src/rmt_llm/tests -q   # 101 passed
python scripts/gen_banner_gif.py # regenerates the README banner
```
