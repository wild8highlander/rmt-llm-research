# Chapter 6 — Outlier Trajectories and Eigenvector Overlaps

*(Program Step 8; assumptions: A1, A2; verification: `experiments/verify_bbp_transition.py`)*

## 6.1 Alignment of sample and population eigenvectors

The outlier eigenvalue carries semantic information only if its
eigenvector actually aligns with the semantic direction. The derived
overlap completes the picture.

> **Theorem 6.1 (overlaps).** Under the hypotheses of Theorem 5.1, let
> $\hat v$ be the sample eigenvector attached to the outlier. For
> $\mu > \mu_c$:
> $$\big|\langle \hat v, v\rangle\big|^2 \;\xrightarrow{a.s.}\;
> \frac{1 - \dfrac{q\,\sigma^4}{\mu^2}}{1 + \dfrac{q\,\sigma^2}{\mu}},$$
> and the limit is $0$ for $\mu \le \mu_c$.

*References.* Nadler (2008) (real Gaussian, first-order formulas);
Paul (2007); Kammoun-Couillet-Alouini-Debbah and subsequent deterministic-
equivalent literature. The formula is implemented as
`spiked.bbp_overlap`.

## 6.2 Reading the formula

Three regimes, all verified numerically:

* **$\mu \downarrow \mu_c$:** numerator $\to 1 - q\sigma^4/\mu_c^2 = 0$:
  the alignment vanishes continuously — at threshold the spike carries
  *no* usable direction, even though its eigenvalue touches the edge.
* **$\mu \gg \sigma^2\sqrt q$:** overlap $\to 1$: the sample direction
  recovers the semantic direction; the residual misalignment is
  $q\sigma^2/\mu$.
* **fixed $\mu$, growing $q$:** alignment degrades polynomially — at
  aspect ratios hostile to the signal, the outlier eigenvalue still
  detaches but points in an increasingly wrong direction. This is the
  rigorous version of "geometric distortion of semantic predictions as
  SNR drops" claimed in the program.

## 6.3 The exact inverse estimator

Inverting the trajectory of Theorem 5.1 turns the spectrum into a spike
estimator — the exact form of the operation the program calls *spectral
clipping*.

> **Proposition 6.2 (inverse-BBP estimator).** For an observed eigenvalue
> $\lambda \ge \lambda_+$, the additive spike is recovered exactly by
> $$\hat\mu(\lambda) \;=\; \frac{A + \sqrt{A^2 - 4q\sigma^4}}{2},
> \qquad A \;=\; \lambda - \sigma^2(1+q),$$
> the large root of $\hat\mu^2 - A\hat\mu + q\sigma^4 = 0$; the map
> $\mu\mapsto\lambda_{\mathrm{out}}(\mu)$ is a bijection
> $(\mu_c,\infty)\to(\lambda_+,\infty)$ and
> $\hat\mu\big(\lambda_{\mathrm{out}}(\mu)\big) = \mu$ identically.
> Below the edge, $\hat\mu = 0$.

*Proof.* The quadratic follows by rearranging
$\lambda = \sigma^2 + \mu + q\sigma^2 + q\sigma^4/\mu$; the branch is
fixed by continuity at $\lambda = \lambda_+$ (both roots coincide at
$\mu_c = \sigma^2\sqrt q$) and by $\hat\mu \sim \lambda - \sigma^2(1+q)$
for large spikes. Note the first-order expansion carries a *minus* sign:
$\hat\mu = \lambda - \sigma^2(1+q) - q\sigma^4/\lambda + O(\mu^{-3})$.
$\square$

*Remark.* A first-order estimator with the opposite sign of the
second-order term circulates in the program notes; it is *not* the
inverse of the BBP map and carries a $2q\sigma^4/\mu$ bias. The
verification suite tests the exact formula (round trip to machine
precision; Experiment 4).

## 6.4 Numerical status

| Check | Result |
|-------|--------|
| Round trip $\hat\mu(\lambda_{\mathrm{out}}(\mu)) = \mu$ | error $\le 8.9\times10^{-16}$ |
| Overlap at $\mu = 3\mu_c$ | empirical $0.757$ vs derived $0.753$ |
| Overlap at $\mu\le\mu_c$ | empirical $\le 0.05$ vs derived $0$ |
