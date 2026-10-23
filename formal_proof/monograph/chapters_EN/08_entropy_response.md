# Chapter 8 — Entropy Response of the Decoder to Bulk Noise

*(Program Step 11, corrected; assumptions: A1 (for the noise spectrum), A3, A5; verification: `experiments/verify_entropy.py`, Fig. `entropy_response.png`)*

## 8.1 Setting

The decoder maps the last-position activation to a token distribution,
$p = \mathrm{softmax}(z)$, $z_i = w_i^{\mathsf T}x/\tau$. Bulk noise
perturbs the logits:

$$z = z_0 + \frac{W\,\xi}{\tau}, \qquad \xi \sim \mathcal N(0,\Sigma_{\mathrm{bulk}}),$$

where $\Sigma_{\mathrm{bulk}}$ has the Marchenko-Pastur spectrum of
Chapter 3 and $W$ stacks the decoder rows $w_i^{\mathsf T}$. Define
$A = W\Sigma_{\mathrm{bulk}}W^{\mathsf T}/\tau^2$.

## 8.2 The derived statement

> **Lemma 8.1 (entropy Hessian).** With $f(z) = \sum_i p_i\log p_i$,
> $p = \mathrm{softmax}(z)$, one has
> $\partial^2 f/\partial z_j\partial z_k = p_j(\delta_{jk}-p_k)\big(z_j - \langle z\rangle_p + 1\big)$.
> At the uniform baseline $z_0 = 0$:
> $\nabla^2 f = \mathrm{diag}(p) - pp^{\mathsf T} \succeq 0$, i.e.
> $H = -f$ is locally concave in the logits.

*Proof.* $f(z) = \langle z,p\rangle - \log Z$; differentiate
$\partial f/\partial z_j = p_j(z_j - \langle z\rangle_p)$ once more and
evaluate at $z_0=0$. $\square$

> **Theorem 8.2 (second-order response).** Under A3 and A5,
> $$\mathbb E_\xi\big[H(\mathrm{softmax}(z_0 + W\xi/\tau))\big]
>   \;=\; H(p_0) \;-\; \tfrac12\,\mathrm{Tr}\!\big((\mathrm{diag}(p_0)-p_0p_0^{\mathsf T})\,A\big)
>   \;+\; \mathcal O(\tau^{-3}).$$
> At the exact uniform baseline this reduces to
> $$\mathbb E[H] = \log K \;-\; \frac{1}{2K}\Big(\mathrm{Tr}\,A - \tfrac1K\,\mathbf 1^{\mathsf T}A\,\mathbf 1\Big)
>   + \mathcal O(\tau^{-3}),$$
> and if the decoder rows are isotropic w.r.t. the noise
> ($W\Sigma W^{\mathsf T} = \bar\sigma^2 I_K$):
> $$\mathbb E[H] \;=\; H_0 \;-\; \Big(1-\tfrac1K\Big)\frac{\bar\sigma^2}{2\tau^2} + \mathcal O(\tau^{-3}),\qquad
>   \bar\sigma^2 = \tfrac1K\mathrm{Tr}\big(W\Sigma_{\mathrm{bulk}}W^{\mathsf T}\big).$$
> The MP law enters through its first moment,
> $\bar\sigma^2 \to \sigma^2_{\mathrm{eff}}\,\langle\|w_i\|^2\rangle$ with
> $\sigma^2_{\mathrm{eff}} = \int\lambda\rho_{MP}(\lambda)d\lambda$.

*Proof.* Taylor-expand $H$ around $z_0$ to second order and average the
quadratic form against the Gaussian logit noise; Lemma 8.1 supplies the
Hessian. $\square$

## 8.3 The sign of the correction — and why the reading matters

The *conditional* entropy of the realized predictive distribution
**decreases**: logit noise makes each sampled distribution more peaked
(concavity, Lemma 8.1). The *mixture* (average predictive) distribution
$E_\xi[p_\xi]$ is flat in $\tau$ to this order (its deviation from
uniform is $\mathcal O(\tau^{-4})$). Both facts are visible in the
verification figure. Consequences:

* any statement of the form "$H = H_0 + \frac{1}{2\tau^2}\int\lambda\rho\,d\lambda$"
  requires specifying *which* entropy and carries, under the natural
  conditional reading, the **opposite sign** and a $(1-1/K)$ factor;
* the honest claim exported from this chapter is Theorem 8.2 with its
  assumptions — the parent repository's Step 11 formula is left as it
  stands (phenomenological), and this chapter replaces it wherever the
  derived statement is required.

## 8.4 Numerical status

$K=64$, $N=200$, MP bulk $q = 0.5$, $6000$ MC trials per temperature:

| $1/\tau^2$ | $\mathbb E[H]$ MC | Theorem 8.2 |
|---|---|---|
| 0.028 | 4.1407 | 4.1406 |
| 0.137 | 4.0179 | 4.0188 |
| 0.391 | 3.9069 | 3.9020 |

The derived slope $-\,(1-1/K)\bar\sigma^2/2 = -0.6577$ per unit
$1/\tau^2$ matches the regression of the MC curve; third-order effects
appear only at $\tau < 2$.
