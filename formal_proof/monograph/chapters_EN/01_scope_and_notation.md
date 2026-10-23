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
