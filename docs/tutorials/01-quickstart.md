# Tutorial 1: Quickstart

This tutorial walks you through the absolute basics of `rmt-llm-research`:

1. Import the package
2. Compute Marchenko-Pastur bounds
3. Sample a random matrix and compare its spectrum to MP
4. Detect a BBP spike
5. Verify Tracy-Widom fluctuations

The full runnable version is at
[`notebooks/01_quickstart.ipynb`](https://github.com/wild8highlander/rmt-llm-research/blob/main/notebooks/01_quickstart.ipynb).

---

## 1. Import the package

```python
import numpy as np
import matplotlib.pyplot as plt

from rmt_llm.marchenko_pastur import mp_bounds, mp_density, mp_cdf
from rmt_llm.bbp_transition import bbp_lambda_max, bbp_is_spiked
from rmt_llm.tracy_widom import tracy_widom_cdf, tracy_widom_mean
```

If the import fails, install the package:

```bash
pip install -e ".[dev]"
```

---

## 2. Marchenko-Pastur bounds

For a random $N \times T$ matrix with $q = N/T \leq 1$, the eigenvalue
support of $\frac{1}{T}XX^{\top}$ is $[\lambda_{-}, \lambda_{+}]$ where:

$$\lambda_{\pm} = \sigma^2 (1 \pm \sqrt{q})^2$$

```python
q, sigma = 0.5, 1.0
lambda_minus, lambda_plus = mp_bounds(q, sigma)
print(f"MP support: [{lambda_minus:.3f}, {lambda_plus:.3f}]")
# MP support: [0.086, 2.914]
```

---

## 3. Compare theory to simulation

```python
N, T = 200, 400       # q = N/T = 0.5
X = np.random.randn(N, T) * sigma
eigs = np.linalg.eigvalsh(X @ X.T / T)

# Plot histogram vs MP density
grid = np.linspace(0.01, 3.5, 500)
rho = np.array([mp_density(g, q, sigma) for g in grid])

plt.hist(eigs, bins=50, density=True, alpha=0.6, label='Sample')
plt.plot(grid, rho, 'r-', lw=2, label='MP theory')
plt.axvline(lambda_minus, color='k', ls='--', alpha=0.4)
plt.axvline(lambda_plus,  color='k', ls='--', alpha=0.4)
plt.xlabel('Eigenvalue')
plt.ylabel('Density')
plt.legend()
plt.title(f'Marchenko-Pastur law (N={N}, T={T}, q={q})')
plt.show()
```

You should see a histogram that closely follows the red MP curve, with
support matching the dashed vertical lines.

---

## 4. Detect a BBP spike

Now we add a rank-1 "signal" of strength $\theta$ to the random matrix and
check whether the largest eigenvalue pops out of the MP bulk.

```python
theta = 1.5      # signal strength
q = 0.5

# Is this spiked?
spiked = bbp_is_spiked(theta, q)
print(f"theta={theta}, q={q} -> spiked: {spiked}")
# spiked: True (because theta > sqrt(q) = 0.707)

# Where does the largest eigenvalue go?
lambda_max = bbp_lambda_max(theta, q, sigma=1.0)
print(f"Predicted lambda_max: {lambda_max:.3f}")
# Predicted lambda_max: 2.729
```

The BBP threshold is $\theta_c = \sqrt{q}$. Below it, $\lambda_{\max}$ sticks
to the MP edge; above it, $\lambda_{\max}$ follows $\theta$.

---

## 5. Tracy-Widom fluctuations

The largest eigenvalue of a pure (un-spiked) Wishart matrix has
fluctuations of order $N^{-2/3}$ around the MP edge, governed by the
Tracy-Widom $F_2$ distribution.

```python
# Mean of the Tracy-Widom F2 distribution
tw_mean = tracy_widom_mean()
print(f"Tracy-Widom F2 mean: {tw_mean:.4f}")
# -1.7711

# CDF at the mean
cdf_at_mean = tracy_widom_cdf(tw_mean)
print(f"F2(tw_mean) = {cdf_at_mean:.4f}")
# ~0.60 (since the mean is to the right of the mode at -1.21)
```

---

## 6. Verify against the simulated matrix

```python
# Center and scale the largest eigenvalue
centered = (eigs.max() - lambda_plus) * N**(2/3) / sigma**2
print(f"Centered & scaled lambda_max: {centered:.3f}")
print(f"Tracy-Widom mean:             {tw_mean:.3f}")

# For large N, the centered value should be close to the TW mean.
# For N=200, you'll see substantial finite-size correction — that's
# the Keating-Snaith term (see src/rmt_llm/keating_snaith.py).
```

---

## What's next?

- [Tutorial 2: Training TinyGPT](02-training-tinygpt.md) — train a real
  transformer from scratch
- [API Reference: marchenko_pastur](../api/rmt-llm.md#rmt_llm.marchenko_pastur)
  — every public function
- [Theory overview](https://github.com/wild8highlander/rmt-llm-research/blob/main/docs/en/RMT_LLM_Arxiv_Preprint.docx)
  — the full preprint
