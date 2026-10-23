# Chapter 2 — The Layer Operator: RMSNorm Conservation and Nonlinear Kernels

*(Program Step 2; assumptions: A3; verification: `experiments/verify_layer_ops.py`, Fig. `layer_ops_verification.png`)*

## 2.1 RMSNorm pins the trace exactly

The phenomenological model treats activation covariances as random
ensembles. The first derived statement is that the RMSNorm block imposes
a hard, assumption-free constraint on them.

> **Lemma 2.1 (trace conservation).** Let
> $\mathrm{Norm}(x) = \sqrt N\, x / \|x\|$ and let the columns of
> $X\in\mathbb{R}^{N\times T}$ be the normalized activations. Then
> $\|x_t\|^2 = N$ for every $t$, hence
> $$\mathrm{Tr}\, C \;=\; \frac1T \sum_{t=1}^{T}\|x_t\|^2 \;=\; N \quad\text{exactly},$$
> i.e. the *mean* eigenvalue of $C$ is pinned to $1$.

*Proof.* $\|\mathrm{Norm}(x_t)\|^2 = N\|x_t\|^2/\|x_t\|^2 = N$. Summing
over $t$ and dividing by $T$ gives the claim. $\square$

This is the rigorous core of the plan's statement that normalization
"projects the spectrum onto the $(N-1)$-sphere". Any MP-type fit to a
post-normalization covariance must respect $\mathrm{Tr}\,C = N$; the
fitting routine `estimate_sigma2_from_bulk` enforces it.

## 2.2 Moment propagation through nonlinearities

Where the plan invokes a "Bushnell-Prasad-Billingsley theorem on
nonlinear transformations of random vectors", the standard machinery of
record is Price's theorem and Stein's lemma. We derive all needed kernels
from them.

> **Lemma 2.2 (Stein/Price).** Let $x\sim\mathcal N(0,\gamma^2)$ and let
> $f$ be weakly differentiable. Then
> $\mathbb E[x f(x)] = \gamma^2 \mathbb E[f'(x)]$ and
> $\mathbb E[x^2 f(x)] = \gamma^2\mathbb E[f] + \gamma^4 \mathbb E[f'']$.

> **Proposition 2.3 (GELU moments).** For $x\sim\mathcal N(0,\gamma^2)$
> and $\mathrm{GELU}(x) = x\Phi(x)$, with
> $\varphi_0 = (2\pi)^{-1/2}$:
> $$\mathbb E[\mathrm{GELU}(x)] = \frac{\gamma^2\varphi_0}{\sqrt{1+\gamma^2}},$$
> $$\mathbb E[\mathrm{GELU}(x)^2] = \gamma^2\Big(\tfrac14 + \tfrac{1}{2\pi}\arcsin\tfrac{\gamma^2}{1+\gamma^2}\Big)
>   + \gamma^4\Big(2\,\tfrac{(2\pi)^{-1}}{\sqrt{1+2\gamma^2}}
>   - 2\,\tfrac{\gamma^2 (2\pi)^{-1}}{(1+\gamma^2)\sqrt{1+2\gamma^2}}\Big).$$

*Proof sketch.* $\mathbb E[x\Phi(x)] = \gamma^2\mathbb E[\varphi(x)]$ by
Lemma 2.2, and $\mathbb E[\varphi(x)] = (2\pi)^{-1/2}(1+\gamma^2)^{-1/2}$
by direct integration. For the second moment apply the second identity of
Lemma 2.2 with $f = \Phi^2$, use
$f'' = 2\varphi^2 - 2x\varphi\Phi$, and evaluate
$\mathbb E[\varphi(x)^2] = (2\pi)^{-1}(1+2\gamma^2)^{-1/2}$ and
$\mathbb E[x\varphi(x)\Phi(x)] = \gamma^2\mathbb E[\varphi(x)^2]/(1+\gamma^2)$
(a second application of Stein to $g=\varphi\Phi$). The bivariate
orthant identity
$\mathbb E[\Phi(x)^2] = \tfrac14 + \tfrac{1}{2\pi}\arcsin\!\big(\tfrac{\gamma^2}{1+\gamma^2}\big)$
completes the proof. $\square$

> **Definition 2.4 (bivariate kernel).** For a centered bivariate normal
> pair $(u,v)$ with variances $(\gamma_u^2,\gamma_v^2)$ and correlation
> $\rho$, the *SiLU kernel* is
> $K_{\mathrm{SiLU}}(\gamma_u^2,\gamma_v^2,\rho) = \mathbb E[\mathrm{SiLU}(u)\,\mathrm{SiLU}(v)]$,
> $\mathrm{SiLU}(x) = x\,\sigma(x)$. It has no elementary closed form and
> is evaluated exactly (to quadrature precision) by tensor Gauss-Hermite
> quadrature — `layer_ops.bivariate_kernel`.

> **Proposition 2.5 (covariance transform).** Under A3, the output
> covariance of a GELU layer $y = \mathrm{GELU}(Wx)$, $x\sim\mathcal N(0,C)$,
> is
> $$\mathrm{Cov}(y)_{ab} = K(\gamma_a^2,\gamma_b^2,\rho_{ab}) - \mathbb E[\mathrm{GELU}(u_a)]\,\mathbb E[\mathrm{GELU}(u_b)],$$
> where $\gamma_a^2 = w_a^{\mathsf T}C w_a$,
> $\rho_{ab} = w_a^{\mathsf T}C w_b/(\gamma_a\gamma_b)$, and $K$ is the
> GELU analogue of Definition 2.4 (diagonal entries: Prop 2.3).

This is the deterministic, layer-by-layer replacement for the postulated
Gaussian ensemble: given $C$, the next layer's covariance is *computed*,
not re-postulated.

## 2.3 Numerical status

| Check | Result |
|-------|--------|
| $\mathrm{Tr}\,C = N$ after RMSNorm | deviation $5.7\times10^{-14}$ (machine precision) |
| GELU moments, closed form vs $4\times10^5$ MC | agree to $\le 10^{-3}$ (MC noise) |
| SiLU kernel vs $1.5\times10^6$ MC | agree to $\le 8\times10^{-4}$ |
| Full GELU-layer covariance transform | relative Frobenius deviation $0.0056$ (pure MC noise at $4\times10^5$ samples) |

The kernels are *derived* (Prop 2.3) or *numerically exact* (Def 2.4);
the only assumption is A3, and Experiment 6 shows the assumption is
quantitatively benign at these widths.
