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
