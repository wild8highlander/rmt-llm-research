# Formal Proof: Deriving the Spectral Laws of LLM Activations


**rmt-llm-research · first-principles program · derived-theorems edition**

This monograph contains only statements derived from explicitly stated
assumptions (register in Chapter 1) and verified numerically
(Chapter 11). Everything else in the parent repository remains
phenomenological and untouched.

**Contents:** Chapters 1-12 below · Russian edition: [monograph_RU.md](monograph_RU.md)

---


# Chapter 1 — Scope, Assumptions and Notation

*Formal proof companion to the rmt-llm-research program. This monograph
contains only statements that are derived from explicitly stated
assumptions and verified numerically; everything else in the parent
repository remains phenomenological and is intentionally left untouched.*

## 1.1 What "from first principles" means here

The research program of the parent repository connects the spectral
analysis of LLM activation matrices to the inevitability of
hallucinations through Random Matrix Theory (RMT). In its phenomenological
form, key formulas are postulated: the Marchenko-Pastur (MP) bulk, the
spiked (BBP) outliers, entropy estimates. This monograph upgrades a
specific, well-defined subset of those statements to the status of
*derived theorems*: every result below is either (i) a theorem imported
from the RMT literature together with the exact hypotheses under which it
holds, (ii) an algebraic derivation from previously established results of
this program, or (iii) a definition. Nothing else is claimed.

The dividing line is deliberately strict. Where the microscopic
derivation would require properties of trained transformer activations
that are currently *not theorems* (universality for dependent nonlinear
recursions, freeness of trained weight matrices, concentration bounds for
decoding trajectories), we do not present a proof and we do not import
the statement. Those parts of the 15-step program remain exactly where
they are — phenomenological, in the parent repository.

## 1.2 Assumption register

Every theorem in this monograph references the assumptions it uses.

| # | Assumption | Status |
|---|------------|--------|
| **A1** | *White entry model.* The activated data matrix is $X = \Sigma^{1/2} Z$ with $Z\in\mathbb{R}^{N\times T}$ (or $\mathbb{C}^{N\times T}$) i.i.d. entries, mean $0$, variance $\sigma^2$; $\Sigma$ is a deterministic population covariance. | Standard; exact for linear random-weight models, a modelling hypothesis for trained networks |
| **A2** | *Proportional asymptotics.* $N, T\to\infty$ with $q = N/T \to q^* \in (0,\infty)$. | Exact regime of all cited RMT theorems |
| **A3** | *Gaussian middle layer.* Pre-activations entering a nonlinearity are jointly Gaussian with the covariance tracked by moment propagation. | Exact for linear models; quantitatively verified approximation for nonlinear layers (Ch. 2, Experiment 6) |
| **A4** | *Asymptotic freeness.* The matrix families whose product is taken are asymptotically free. | Theorem (Voiculescu) for i.i.d./unitarily invariant families; *not* established for trained weights |
| **A5** | *Near-uniform decoding baseline.* The pre-noise logit vector $z_0$ is small (high-temperature / uniform predictive distribution). | Stated explicitly in Ch. 8; verified there |

## 1.3 Notation

| Symbol | Meaning |
|--------|---------|
| $N,\ T$ | activation dimension; number of tokens (context length) |
| $q = N/T$ | aspect ratio |
| $\sigma^2$ | effective noise variance of the bulk (the parameter the phenomenological layer attaches to weight norms) |
| $X,\ C = \tfrac1T XX^{\mathsf T}$ | activation matrix and its (sample) covariance |
| $\lambda_\pm = \sigma^2(1\pm\sqrt q)^2$ | MP bulk edges (Thm 3.2) |
| $\mu,\ \mu_c = \sigma^2\sqrt q$ | additive spike strength and its BBP threshold (Thm 5.1) |
| $m(z)=\int\frac{d\mu(\lambda)}{\lambda-z}$, $G(z)=-m(z)=\frac1N\mathrm{Tr}(zI-C)^{-1}$ | Stieltjes transforms (two conventions) |
| $F_\beta$ | Tracy-Widom law of index $\beta$ (Ch. 9) |
| $H(p)=-\sum_i p_i\log p_i$ | Shannon entropy of the predictive distribution |

## 1.4 Map of the program

| Program step | Status | Where |
|--------------|--------|-------|
| Step 2 — layer operator | derived (Lemma 2.1, Prop 2.3-2.4 under A3) | Ch. 2 |
| Steps 4-5 — Silverstein equation, MP bulk | derived (Thm 3.1, 3.2) | Ch. 3 |
| Step 6 — free probability | derived for i.i.d. ensembles under A4 (Thm 4.1) | Ch. 4 |
| Step 7 — BBP threshold | derived (Thm 5.1) | Ch. 5 |
| Step 8 — outliers and overlaps | derived (Thm 6.1) | Ch. 6 |
| Step 10 — hallucination split | *definition*, made operational | Ch. 7 |
| Step 11 — entropy bridge | derived in corrected second-order form under A3, A5 (Thm 8.1) | Ch. 8 |
| Step 13 — Tracy-Widom edges | derived with the $\beta=1$ correction (Thm 9.1) | Ch. 9 |
| Step 14 — optimal clipping | three problems, three exact solutions (Thm 10.1-10.3) | Ch. 10 |
| Steps 1, 3, 9, 12, 15 | *not derived here*; left phenomenological in the parent repository | — |

Every derivation chapter ends with the numerical verification that backs
it; all six experiments reproduce in ~3 minutes via
`formal_proof/experiments/run_all.py`.


---


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


---


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


---


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


---


# Chapter 5 — The BBP Phase Transition

*(Program Step 7; assumptions: A1, A2; verification: `experiments/verify_bbp_transition.py`, Fig. `bbp_transition_verification.png`, animation `bbp_transition.gif`)*

## 5.1 The spiked model

Semantic content carried by activations is low-rank relative to the
noise bulk. The minimal derived model is the *spiked population covariance*

$$\Sigma \;=\; \sigma^2 I_N + \mu\, v v^{\mathsf T}, \qquad \|v\| = 1, \qquad X = \Sigma^{1/2} Z,$$

with $Z$ as in A1. The sample covariance
$C = \tfrac1T XX^{\mathsf T}$ inherits a rank-one perturbation of
strength $\mu$ on top of its MP bulk.

## 5.2 Threshold and trajectory

> **Theorem 5.1 (BBP transition).** Under A1-A2, as $N,T\to\infty$ with
> $q=N/T$:
>
> (i) if $\mu \le \mu_c = \sigma^2\sqrt q$, the largest sample eigenvalue
> converges to the bulk edge $\lambda_+ = \sigma^2(1+\sqrt q)^2$ — the
> spike is *absorbed by the bulk*;
>
> (ii) if $\mu > \mu_c$, the largest sample eigenvalue converges to the
> *outlier*
> $$\lambda_{\mathrm{out}}(\mu) \;=\; (\sigma^2+\mu)\Big(1 + \frac{q\,\sigma^2}{\mu}\Big),$$
> detached from the bulk. At $\mu = \mu_c$ the trajectory attaches
> continuously: $\lambda_{\mathrm{out}}(\mu_c) = \lambda_+$.

*References.* Baik-Ben Arous-Peche (2005) for the complex Gaussian case;
Paul (2007) and Nadler (2008) for the real case; Benaych-Georges-Nadler
(2012) and Bloemendal-Knowles-Lee-Vu (2016) for generalizations. The
additive-spike parametrization above ($\mu = $ population eigenvalue
minus $\sigma^2$) is the one used throughout the parent repository.

## 5.3 The pole-equation derivation

The threshold and the trajectory are not independent postulates: both
follow from one resolvent statement. Outside the bulk the limiting
Stieltjes transform of the *deformed* law picks up a pole condition from
the rank-one spike. Writing $G(\lambda) = -m(\lambda)$ for the MP
resolvent of Chapter 3 (real, positive for $\lambda > \lambda_+$), the
outlier is the root $\lambda > \lambda_+$ of

$$\boxed{\;(\sigma^2+\mu)\,\big(1 - q + q\,\lambda\,G(\lambda)\big) \;=\; \lambda\;} \tag{5.1}$$

> **Proposition 5.2.** The unique root of (5.1) above $\lambda_+$ is
> exactly $\lambda_{\mathrm{out}}(\mu)$ of Theorem 5.1, and it exists iff
> $\mu > \mu_c$.

*Proof sketch.* Substituting the closed form of Corollary 3.1b and
$\lambda = \lambda_{\mathrm{out}}$, one verifies
$b - \sqrt\Delta = 2\sigma^2 q(\sigma^2+\mu)/\mu$ with
$b = \lambda_{\mathrm{out}} + \sigma^2(q-1)$ and
$\sqrt\Delta = (\mu^2 - q\sigma^4)/\mu$ — the algebra reduces to
$(\lambda-\lambda_\pm) = (\mu \mp \sigma^2\sqrt q)^2/\mu$, which is
where the threshold $\mu_c = \sigma^2\sqrt q$ appears as the point where
$\lambda_{\mathrm{out}}$ crosses the edge. Below the threshold the
equation (5.1) has no root above $\lambda_+$. $\square$

Equation (5.1) is implemented as `spiked.bbp_outlier_pole_equation`
(numeric root) and agrees with the closed form to machine precision
(max $|\cdot| \le 8.9\times10^{-16}$ over $q\in[0.25,2]$).

## 5.4 Numerical status

Simulation of the spiked model ($N=400$, $T=1000$, 60 trials per point):

| $\mu/\mu_c$ | $\lambda_{max}$ empirical | derived | overlap empirical | derived |
|---|---|---|---|---|
| 0.24 | 2.6384 | 2.6649 (edge) | 0.003 | 0 |
| 1.17 | 2.8558 | 2.8369 | 0.476 | 0.467 |
| 3.20 | 3.6077 | 3.6215 | 0.757 | 0.753 |

Subcritical spikes stick to the edge (residual deficit is the
Tracy-Widom scale, Chapter 9); supercritical spikes follow the derived
trajectory. The animation `bbp_transition.gif` shows the detachment.


---


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


---


# Chapter 7 — The Hallucination Split: a Spectral Definition

*(Program Step 10; a definition made operational; utility: `rmt_fp/hallucination.py`)*

## 7.1 The orthogonal split

With the bulk edges of Chapter 3 in hand, the activation Hilbert space
admits a canonical, parameter-free decomposition.

> **Definition 7.1.** Let $C = \tfrac1T XX^{\mathsf T}$ have eigenvalues
> $\lambda_1\ge\dots\ge\lambda_N$ and eigenvectors
> $\hat v_1,\dots,\hat v_N$, and let $\lambda_+$ be the MP edge. Define
> $$\mathcal H_{\mathrm{signal}} = \mathrm{span}\{\hat v_k : \lambda_k > \lambda_+\},\qquad
>   \mathcal H_{\mathrm{bulk}} = \mathrm{span}\{\hat v_k : \lambda_k \le \lambda_+\},$$
> $$\mathcal H = \mathcal H_{\mathrm{signal}} \oplus \mathcal H_{\mathrm{bulk}}.$$

> **Definition 7.2 (projector ratio).** For a vector $x\in\mathcal H$
> (e.g. the last-position activation $x_T^{(L)}$):
> $$r(x) \;=\; \frac{\|P_{\mathrm{bulk}}\,x\|^2}{\|P_{\mathrm{signal}}\,x\|^2 + \varepsilon}.$$

> **Definition 7.3 (flagging rule).** $x$ is *noise-dominated* iff
> $r(x) > \eta_{\mathrm{crit}}$, where $\eta_{\mathrm{crit}}$ is a
> decision parameter, not a derived constant.

## 7.2 What is and is not claimed

Everything above is linear algebra plus one threshold: it is a
*definition*, and it is the operational quantity the phenomenological
detectors of the parent repository track. What this monograph does
**not** claim here is a theorem about $r(x)$ along decoding
trajectories. The program's Step 12 (a Talagrand-type exponential lower
bound on hallucination probability) requires concentration properties of
the autoregressive dynamics that are not established; accordingly the
bound is not imported, and its step remains phenomenological.

What *is* derived about the behavior of trajectories is the second-order
entropy statement of Chapter 8, which is the information-theoretic
consequence that survives under explicit assumptions.

## 7.3 Implementation

`rmt_fp/hallucination.py` provides:

* `spectral_split(eigs, lambda_plus)` — the index split of Definition 7.1;
* `noise_signal_ratio(x, vecs, eigs, lambda_plus)` — Definition 7.2 by
  exact projector arithmetic;
* `hallucination_flag(...)` — Definition 7.3;
* `spiked_activation_matrix(...)` — the synthetic harness used by the
  experiments (MP bulk + rank-$r$ semantic spikes).

The module is deliberately thin: it makes the definition executable and
auditable, nothing more.


---


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


---


# Chapter 9 — Tracy-Widom Edge Fluctuations and Finite-Size Corrections

*(Program Step 13, with the β-index correction; assumptions: A1, A2; verification: `experiments/verify_tracy_widom.py`, Fig. `tracy_widom_verification.png`)*

## 9.1 The derived statement

The MP edge of Chapter 3 is a soft edge: at finite $N,T$ the largest
noise eigenvalue fluctuates across it on the $N^{-2/3}$ scale, with a
Tracy-Widom limit law whose *index* depends on the entry field.

> **Theorem 9.1 (Johnstone; real case).** Let $C = \tfrac1T XX^{\mathsf T}$
> with $X\in\mathbb R^{N\times T}$ i.i.d. $\mathcal N(0,\sigma^2)$ (A1,
> real). With Johnstone's centering and scaling
> $$\mu_{N,T} = \sigma^2\,\frac{(\sqrt{T-1}+\sqrt N)^2}{T},\qquad
>   s_{N,T} = \sigma^2\,\frac{(\sqrt{T-1}+\sqrt N)\big(T^{-1/2}+N^{-1/2}\big)^{1/3}}{T},$$
> $$\frac{\lambda_{\max}(C) - \mu_{N,T}}{s_{N,T}} \;\xrightarrow{d}\; F_1\qquad(\beta = 1).$$
> For complex entries the limit is $F_2$ ($\beta = 2$).

*References.* Johnstone (2001) (real Wishart, first-order
centering/scaling); Tracy-Widom (1994, 1996); El Karoui (2006)
(improved centering
$\mu = \sigma^2(\sqrt{T-1/2}+\sqrt{N-1/2})^2/T$, implemented as
`el_karoui_center_scale`); Ramírez-Ridout-... for the complex case;
Bloemendal et al. (2016) for the general finite-rank deformation.

> **Remark 9.2 (the β-index of real activations).** Activations of
> real-valued language models fall in the $\beta = 1$ (GOE) universality
> class. The $\beta = 2$ law is the complex-ensemble counterpart. The two
> laws differ substantially ($F_1$: mean $-1.2065$, std $1.268$;
> $F_2$: mean $-1.7711$, std $0.902$), and the verification below
> separates them at $p \sim 10^{-35}$. Moments: Bornemann tables.

The leading-order scale quoted in the program notes,
$\sigma^2 N^{-2/3}q^{-1/6}(1+\sqrt q)^{4/3}$, differs from Johnstone's
$s_{N,T}$ by a $q$-dependent factor; the derived normalization used here
is the one that collapses the empirical histograms (verified), and the
centering $\mu_{N,T}$ matters at accessible sizes — using the raw edge
$\lambda_+$ shifts the standardized mean by an $\mathcal O(1)$ amount.

## 9.2 Reference samples without tables

$F_1$/$F_2$ reference samples are generated from the $\beta$-ensembles
themselves — no external tables are needed:

$$\xi_1 = (\lambda_{\max}^{\mathrm{GOE}} - 2\sqrt p)\,p^{1/6} \Rightarrow F_1,\qquad
  \xi_2 = (\lambda_{\max}^{\mathrm{GUE}} - 2\sqrt p)\,p^{1/6} \Rightarrow F_2,$$

with GUE/GOE constructed at off-diagonal variance $1$. At $p = 300$ the
generators reproduce the tabulated moments to $\le 0.05$.

## 9.3 Numerical status

$N=220$, $T=440$, $2000$ trials per ensemble:

| Comparison | KS statistic | p-value |
|---|---|---|
| real Wishart (Johnstone-std) vs GOE $\to F_1$ | 0.044 | 0.079 — same law |
| complex Wishart vs GUE $\to F_2$ | 0.029 | 0.473 — same law |
| real vs complex | 0.199 | $4.8\times10^{-35}$ — distinct |

Standardized moments, real case: mean $-1.194$ (target $-1.2065$), std
$1.288$ (target $1.268$), skew $0.336$ (target $0.293$). The
$\beta$-index correction is thus not cosmetic: applying $F_2$ quantiles
to real activation spectra would overstate edge exceedance probabilities
by a wide margin.

## 9.4 Consequence for the detector layer

For finite $N,T$ the noise floor is not $\lambda_+$ but
$\lambda_+ + \mathcal O(s_{N,T}\,F_1^{-1}(\alpha))$: a detector that
flags eigenvalues above the *deterministic* edge will raise false
alarms at rate $\alpha$ determined by the $F_1$ tail. The rigorous
calibration of thresholds is therefore: flag at
$\mu_{N,T} + s_{N,T}\,q_\alpha(F_1)$ — with $F_1$, not $F_2$.


---


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


---


# Chapter 11 — Verification Summary

All results of this monograph are checked by the experiment suite in
`formal_proof/experiments/`. One command reproduces everything
(~3 minutes, CPU only):

```bash
cd formal_proof/experiments
python3 run_all.py
```

## 11.1 What was verified

| # | Experiment | Statement(s) verified | Key numbers |
|---|------------|----------------------|-------------|
| 1 | `verify_mp_law.py` | Thm 3.1-3.2: MP density, edges, closed-form Stieltjes | Silverstein residual $3\times10^{-16}$; resolvent error $9\times10^{-4}$; KS $0.0033$ |
| 2 | `verify_bbp_transition.py` | Thm 5.1, Prop 5.2, Thm 6.1: threshold, trajectory, overlaps | $\lambda_{out}$: emp 3.6077 vs 3.6215 at $3.2\mu_c$; overlap 0.757 vs 0.753; pole-eq match $9\times10^{-16}$ |
| 3 | `verify_tracy_widom.py` | Thm 9.1: $F_1$ for real, $F_2$ for complex | KS real-vs-GOE $p=0.079$; complex-vs-GUE $p=0.473$; real-vs-complex $p=5\times10^{-35}$ |
| 4 | `verify_shrinkage.py` | Thm 10.1-10.3: operator hierarchy | inverse-BBP: $0.013$–$0.028$ vs $\approx1.0$ for hard operators; GD anchor $4/\sqrt3$ |
| 5 | `verify_entropy.py` | Thm 8.2: entropy response | MC 4.1407 vs derived 4.1406 at $1/\tau^2=0.028$; slope $-0.658$ |
| 6 | `verify_layer_ops.py` | Lemma 2.1, Prop 2.3-2.5: layer calculus | RMSNorm $5.7\times10^{-14}$; covariance transform dev $0.0056$ |

## 11.2 Reproducibility

* Seeds are fixed in every script; results are deterministic to MC noise.
* Dependencies: `numpy`, `scipy`, `matplotlib`, `pillow` (see
  `formal_proof/requirements.txt`). No GPU, no network.
* Figures land in `formal_proof/assets/figures/`, animations in
  `assets/animations/` — both are committed so the monograph renders
  without rerunning anything.

## 11.3 Known finite-size effects (documented honestly)

* The largest eigenvalue sits slightly *below* $\lambda_+$ at
  subcritical spikes — the Tracy-Widom scale of Chapter 9, not an error.
* The Johnstone centering leaves an $\mathcal O(N^{-1/2})$ bias in the
  standardized TW mean at $N\sim200$; El Karoui's correction
  (`el_karoui_center_scale`) reduces it.
* The GOE/GUE TW generators converge slowly in $p$; the suite compares
  *distributions* (two-sample KS), which is robust to that residual
  bias, and reports absolute moments only as diagnostics.
* Third-order entropy terms appear at $\tau \lesssim 2$; Theorem 8.2 is
  a second-order statement and should not be extrapolated to low
  temperatures.


---


# References

Primary sources for the derived results, in the order used by the
chapters.

**Random matrix theory — bulk.**
- V. A. Marchenko, L. A. Pastur, *Distribution of eigenvalues for some
  sets of random matrices*, Math. USSR-Sb. **1** (1967) 457–483.
- J. W. Silverstein, *The smallest eigenvalue of a large dimensional
  Wishart matrix*, Ann. Probab. **13** (1985) 1364–1368.
- J. W. Silverstein, *The eigenvalue distribution of a matrix with
  dependent entries*, Ann. Probab. **23** (1995) 1437–1460.
- Z. D. Bai, J. W. Silverstein, *Spectral Analysis of Large Dimensional
  Random Matrices*, 2nd ed., Springer (2010) — Thm 4.3 (Silverstein
  equation), Ch. 3 (MP law).

**Free probability.**
- D. Voiculescu, *Addition of certain non-commuting random matrices*,
  J. Funct. Anal. **123** (1994) 215–235.
- D. Voiculescu, *Multiplication of certain non-commuting random
  matrices*, J. Oper. Theory **18** (1987) 223–235 (S-transform).
- R. Speicher, *Free calculus and the R-transform*, J. Theor. Probab.
  **7** (1994) 381–395.

**Spiked models.**
- J. Baik, G. Ben Arous, S. Péché, *Phase transition of the largest
  eigenvalue for nonnull complex sample covariance matrices*, Ann.
  Statist. **33** (2005) 1753–1797.
- D. Paul, *Asymptotics of sample eigenstructure for a large
  dimensional spiked covariance model*, Statist. Sinica **17** (2007)
  1617–1642.
- B. Nadler, *Finite rank perturbation of random covariance matrices*,
  Ann. Statist. **36** (2008) 741–788.
- F. Benaych-Georges, R. R. Nadler, *The eigenvalues of random
  covariance matrices*, Random Matrices: Theory Appl. **1** (2012)
  1150001.
- W. Knowles, J. Yin, *Anisotropic local laws for random matrices*,
  Probab. Theory Relat. Fields **169** (2017) 257–352.
- I. M. Johnstone, *On the distribution of the largest eigenvalue in
  principal components analysis*, Ann. Statist. **29** (2001) 295–327.
- A. El Karoui, *A rate of convergence result for the largest eigenvalue
  of complex white Wishart matrices*, Ann. Statist. **34** (2006)
  2073–2097.
- A. Bloemendal, A. Knowles, H.-T. Lee, E. Vu, *On the principal
  components of sample covariance matrices*, Probab. Theory Relat.
  Fields **164** (2016) 1–102.
- C. A. Tracy, H. Widom, *Level-spacing distributions and the Airy
  kernel*, Comm. Math. Phys. **159** (1994) 151–174; *On orthogonal and
  symplectic matrix ensembles*, Comm. Math. Phys. **177** (1996)
  727–754. (Moments: Bornemann tables; see also the Tracy–Widom
  distribution reference tables.)

**Spectral estimation.**
- C. Eckart, G. Young, *The approximation of one matrix by another of
  lower rank*, Psychometrika **1** (1936) 211–218.
- L. Mirsky, *Symmetric gauge functions and unitarily invariant norms*,
  Quart. J. Math. **11** (1960) 50–59.
- M. Gavish, D. L. Donoho, *The optimal hard threshold for singular
  values is $4/\sqrt 3$*, IEEE Trans. Inform. Theory **60** (2014)
  5040–5053.
- D. L. Donoho, M. Gavish, *Optimal shrinkage of eigenvalues in the
  spiked covariance model*, Comm. Pure Appl. Math. (2018).

**Layer calculus.**
- R. Price, *A useful theorem for nonlinear devices having Gaussian
  inputs*, IRE Trans. Inform. Theory **4** (1958) 69–72.
- C. Stein, *A bound for the error in the normal approximation to the
  distribution of a sum of dependent random variables*, Proc. 6th
  Berkeley Symp. **2** (1972) 583–602.
- D. Hendrycks, K. Gimpel, *Gaussian Error Linear Units (GELU)*,
  arXiv:1606.08415 (2016).
- N. Shazeer, *GLU variants improve transformer*, arXiv:2002.05202
  (2020).

**In-repo companions.**
- Phenomenological core: `src/rmt_llm/` (unchanged by this monograph).
- Derived core: `formal_proof/rmt_fp/`.
- Verification suite: `formal_proof/experiments/run_all.py`.
