# Chapter 4 — Free Probability of Weight-Activation Products

*(Program Step 6; assumptions: A4 (a theorem for i.i.d. ensembles); verification: see Ch. 11)*

## 4.1 Why non-commutativity matters

Attention composes weight matrices with activation matrices:
schematically $C_{\mathrm{attn}} \sim W X X^{\mathsf T} W^{\mathsf T}$
(up to normalization and softmax). Weight matrices and activations do
not commute, so the spectrum of the product is not a function of the two
spectra alone. The correct calculus is Voiculescu's free probability.

> **Theorem 4.1 (Voiculescu, asymptotic freeness).** Independent random
> matrix families whose distributions are invariant under conjugation by
> unitary matrices (in particular: deterministic scalars, Gaussian and
> i.i.d. ensembles with a spectral distribution) are asymptotically free
> as $N\to\infty$: mixed *-moments converge to the free cumulant
> functional. Consequently:
>
> (i) *addition*: the limiting spectrum of $A_N + B_N$ is the free
> additive convolution $\mu_A \boxplus \mu_B$, encoded by the R-transform
> $R_{A\boxplus B}(z) = R_A(z) + R_B(z)$;
>
> (ii) *multiplication* (free products of self-adjoint elements): the
> limiting spectrum of $A_N B_N$ is $\mu_A \boxtimes \mu_B$, encoded by
> the S-transform
> $S_{A\boxtimes B}(z) = S_A(z)\,S_B(z)$.

This is exactly the rigorous content behind the program's Step 6 formula
$S_{C_{\mathrm{attn}}}(z) = S_{W_{\mathrm{ATTN}}}(z)\cdot S_X(z)$: it is
a *theorem for i.i.d. weight and activation ensembles* (A4), and an
*unproven ansatz* for trained networks. We keep the theorem and we do
not claim the ansatz.

## 4.2 The R- and S-transforms of the MP law

> **Proposition 4.2.** The free cumulants of $\mu_{q,\sigma^2}$
> (Chapter 3) are $\kappa_n = q^{\,n-1}\sigma^{2n}$, $n\ge 1$; hence
> $$R_{\mathrm{MP}}(z) \;=\; \sum_{n\ge 0}\kappa_{n+1} z^n
>   \;=\; \frac{\sigma^2}{1 - q\sigma^2 z}, \qquad
>   S_{\mathrm{MP}}(z) \;=\; \frac{1}{\sigma^2\,(1 + q\,z)}.$$

*Derivation.* The moments of the MP law are the q-weighted Catalan
numbers, $m_n = \sigma^{2n} C_n(q)$ with generating function
$m(z) = \frac{1 - q - \sigma^{-2}z\cdot(\dots)}{2qz}$ — equivalently they
follow from the quadratic (3.2) by expanding at infinity. The free
cumulant generating function is the compositional inverse
$R\big(z\,m(z)\big) = z + 1/m(z)\cdot(\dots)$; for the MP law the series
inverts to the geometric form above, and the S-transform follows from
the standard relation $S(z) = \frac{1+z}{z}\,\chi(z)$ with
$\chi$ the inverse of $z\mapsto z\,R(z\psi)/\psi\cdot(\dots)$; carrying
it out gives $S_{\mathrm{MP}}(z) = 1/(\sigma^2(1+qz))$. $\square$

Two sanity anchors, both immediate: at $q\to 0$ (many more samples than
dimensions) $S\to 1/\sigma^2$, the S-transform of the deterministic
$\sigma^2 I$; and free multiplicative scaling by a scalar multiplies the
S-transform by that scalar — consistent.

## 4.3 What this buys for the attention operator

Under A4, the bulk of $C_{\mathrm{attn}} = W X X^{\mathsf T}W^{\mathsf T}/T$
is obtained by transform arithmetic rather than by simulation:

$$S_{C_{\mathrm{attn}}}(z) = S_{WW^{\mathsf T}}(z)\cdot S_{XX^{\mathsf T}/T}(z),$$

and each factor is available in closed or quasi-closed form (Chapter 3
for the activation factor; the weight factor is likewise an MP-type law
under A1). This yields *analytic* predictions for how attention reshapes
the bulk — the object the phenomenological layer fits empirically.

**Honest boundary.** For trained transformers the weights are neither
unitarily invariant nor independent of the activations; asymptotic
freeness is an open empirical question, actively investigated in the
literature. Chapter 11 therefore treats the free-probability branch as
*derived mathematics with a stated domain of validity*, not as a
description of trained checkpoints.
