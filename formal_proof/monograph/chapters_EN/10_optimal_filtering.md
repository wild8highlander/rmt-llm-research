# Chapter 10 — Optimal Spectral Filtering: Three Problems, Three Exact Solutions

*(Program Step 14, clarified; assumptions: none beyond earlier chapters; verification: `experiments/verify_shrinkage.py`, Fig. `shrinkage_comparison.png`)*

## 10.1 Why the plan's single-operator claim needs unpicking

Step 14 states a variational problem and names one exact solution. The
derived situation is richer and more precise: *three* distinct
estimation problems are natural at this point of the program, each with
its own exact solution, and they are frequently conflated.

## 10.2 The three problems

> **Theorem 10.1 (Eckart-Young-Mirsky).** The problem
> $$\min_{\hat C}\ \|\,C_{\mathrm{emp}} - \hat C\,\|_F \quad\text{s.t.}\quad \mathrm{rank}(\hat C)\le r$$
> is solved exactly by keeping the top-$r$ sample eigenvalues *unchanged*
> and zeroing the rest. Any soft de-biasing of the kept eigenvalues makes
> the Frobenius error strictly larger.

> **Theorem 10.2 (inverse-BBP estimator; derived in Prop 6.2).** For the
> problem *estimate the population covariance*
> $\Sigma = \sigma^2 I + \sum_k \mu_k v_kv_k^{\mathsf T}$ *from the sample
> spectrum*, the exact estimator of a supercritical eigenvalue is
> $$\hat\lambda_k \;=\; \sigma^2 + \hat\mu(\lambda_k),\qquad
>   \hat\mu(\lambda) = \frac{A+\sqrt{A^2-4q\sigma^4}}{2},\quad
>   A = \lambda-\sigma^2(1+q),$$
> applied to $\lambda_k > \lambda_+$, with $\hat\lambda = \sigma^2$ for
> eigenvalues inside the bulk (or $0$ for the bulk-less target
> $\sum_k\mu_kv_kv_k^{\mathsf T}$). This operator is the exact inverse of
> the BBP map (Theorem 5.1/6.1); its first-order expansion is
> $\lambda - q\sigma^2 - q\sigma^4/\lambda$.

> **Theorem 10.3 (Gavish-Donoho).** For the problem *recover the
> spike-only signal $P$ from $X = P + \sigma Z$* under the oracle
> Frobenius AMSE, the optimal hard threshold for singular values is
> $$\tau^*(q) = \sqrt{\,2(q+1) + \frac{8q}{q+1+\sqrt{q^2+14q+1}}\,},$$
> i.e. keep singular values above $\sigma\,\tau^*(q)$, zero the rest.
> Anchor: $\tau^*(1) = 4/\sqrt 3$ (asserted in the verification to
> $10^{-12}$). Translated to eigenvalues of $C$: the threshold is
> $\sigma^2\tau^*(q)^2$.

*References.* Eckart-Young (1936), Mirsky (1960); Chapter 6 for
Theorem 10.2; Gavish-Donoho (2014) for Theorem 10.3; Donoho-Gavish
(2018) for the soft-shrinkage refinements.

## 10.3 What the experiments show

With $N=300$, $T=600$, $q=0.5$, $25$ trials per point, relative
Frobenius error:

* **Target $\Sigma$ (full population).** The exact inverse-BBP operator
  achieves $0.013$–$0.028$ across the spike grid — *two orders of
  magnitude* below the hard truncation / MP-clip operators
  ($\approx 0.95$–$1.00$, which destroy the $\sigma^2$ bulk) and below
  its own first-order expansion ($0.018$–$0.038$).
* **Target $P$ (spike-only).** The de-biased soft operator (bulk zeroed)
  dominates at and above threshold; the Gavish-Donoho hard threshold is
  conservative for weak spikes by design (its optimality is a minimax
  statement over spike strength, not a pointwise claim), and raw-λ
  operators carry the $\sigma^2(1+q)$ bias.

**Honest summary for the program.** The operator named in Step 14 —
de-bias by inverting the BBP map, clip at the MP edge — is the correct
solution of the *population-estimation* problem (Theorem 10.2), and it
is the one the hallucination pipeline needs (it reconstructs the signal
subspace with calibrated strength). The *rank-constrained Frobenius
problem as literally stated* is solved by plain truncation
(Theorem 10.1), and the *oracle-signal problem* by the Gavish-Donoho
threshold (Theorem 10.3). The three answers differ; the program should
name which problem it means at each use site.
