# Chapter 3 — The Silverstein Equation and the Marchenko-Pastur Bulk

*(Program Steps 4-5; assumptions: A1, A2; verification: `experiments/verify_mp_law.py`, Fig. `mp_law_verification.png`)*

## 3.1 The self-consistent equation

Let $C = \tfrac1T XX^{\mathsf T}$ with $X$ as in A1 and $q = N/T$ as in
A2. The limiting spectral distribution $\mu_{q,\sigma^2}$ of $C$ is
characterized by its Stieltjes transform.

> **Theorem 3.1 (Silverstein; Bai-Silverstein Thm 4.3 for $H=\delta_{\sigma^2}$).**
> Under A1-A2 the empirical spectral distribution of $C$ converges
> almost surely to a deterministic law $\mu_{q,\sigma^2}$ whose Stieltjes
> transform $m(z) = \int (\lambda - z)^{-1}d\mu(\lambda)$ is the unique
> solution in $\mathbb C^+\cup\mathbb R\setminus[\lambda_-,\lambda_+]$ of
> the fixed-point equation
> $$m(z) \;=\; \frac{1}{\sigma^2\big(1 - q - q\,z\,m(z)\big) - z}. \tag{3.1}$$

Equation (3.1) is the *cavity/leave-one-out* self-consistency: the
resolvent of $C$ with one sample column removed differs from the full
resolvent by a rank-one update (Sherman-Morrison), and the resulting
scalar denominator must reproduce $m$ itself. For $H = \delta_{\sigma^2}$
the equation is quadratic.

> **Corollary 3.1a (quadratic form).** Eliminating the denominator of
> (3.1),
> $$\sigma^2 q\, z\, m(z)^2 \;+\; \big(z + \sigma^2(q-1)\big)\, m(z) \;+\; 1 \;=\; 0. \tag{3.2}$$
> Equivalently for the resolvent convention $G = -m =
> \tfrac1N\mathrm{Tr}(zI-C)^{-1}$:
> $\sigma^2 q z G^2 - \big(z+\sigma^2(q-1)\big)G + 1 = 0$.

> **Corollary 3.1b (closed form).** The physical root of (3.2), analytic
> on $\mathbb C\setminus[\lambda_-,\lambda_+]$ with $m(z)\sim -1/z$ at
> infinity, is
> $$m(z) \;=\; \frac{-\big(z+\sigma^2(q-1)\big) + s(z)}{2\sigma^2 q\,z},
>   \qquad s(z) = \sqrt{z-\lambda_-}\,\sqrt{z-\lambda_+},$$
> where each square root is taken on its principal branch; the *product*
> $s(z)$ is the branch analytic off $[\lambda_-,\lambda_+]$, positive for
> $z>\lambda_+$, negative for $z<\lambda_-$, with $\mathrm{Im}\,s>0$ on
> $\mathbb C^+$. This branch selection makes $m$ the Stieltjes transform:
> $\mathrm{Im}\,m > 0$ on $\mathbb C^+$, $m(z)<0$ for real $z>\lambda_+$.

*Remark on normalizations.* The fixed-point equation appears in several
equivalent normalizations across the literature; (3.1)/(3.2) is the form
verified numerically below (residual $3\times10^{-16}$, resolvent
agreement $\sim10^{-4}$ at $N=900$). Any candidate equation should be
accepted only after such a check — in particular it must reproduce the
edges of Theorem 3.2.

## 3.2 The bulk density and its edges

> **Theorem 3.2 (Marchenko-Pastur).** The limiting law has density
> $$\rho(\lambda) \;=\; \frac{\sqrt{(\lambda_+-\lambda)(\lambda-\lambda_-)}}{2\pi\sigma^2 q\,\lambda},
> \qquad \lambda\in[\lambda_-,\lambda_+],$$
> $$\lambda_\pm \;=\; \sigma^2\,(1\pm\sqrt q)^2,$$
> plus an atom of weight $\max(0,\,1-1/q)$ at $0$ (present when $q>1$).

*Derivation.* The support endpoints are the real roots of the
discriminant of the quadratic (3.2),
$\big(z+\sigma^2(q-1)\big)^2 - 4\sigma^2 q z = 0$, i.e.
$z = \sigma^2\big[(1+q)\pm 2\sqrt q\big]$. The density follows from the
Stieltjes-Perron inversion $\rho(\lambda) = -\tfrac1\pi \mathrm{Im}\,
m(\lambda+i0)$ applied to Corollary 3.1b: inside the support
$s(\lambda+i0) = i\sqrt{(\lambda-\lambda_-)(\lambda_+-\lambda)}$, whose
imaginary part produces exactly the stated square root. The atom is the
rank deficit $N - \min(N,T)$ normalized by $N$. $\square$

The asymptotic moments follow from (3.2) as well:
$\int\lambda\,d\mu = \sigma^2$ and
$\int\lambda^2 d\mu = \sigma^4(1+q)$ — the identities used to
cross-check the closed form in `mp.py`.

## 3.3 Numerical status

| Check | Result |
|-------|--------|
| Silverstein residual of the closed form | $3.1\times10^{-16}$ |
| Resolvent $(1/N)\mathrm{Tr}(zI-C)^{-1}$ vs closed form, real & complex $z$, 40 trials | max error $9.0\times10^{-4}$ ($N=900$) |
| Empirical spectrum vs derived density ($N=900$, $T=2571$) | KS distance $0.0033$ |
| Largest eigenvalue vs $\lambda_+$ | relative deviation $0.010$ (TW-scale, Ch. 9) |
