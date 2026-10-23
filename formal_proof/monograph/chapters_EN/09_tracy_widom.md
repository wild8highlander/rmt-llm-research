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
