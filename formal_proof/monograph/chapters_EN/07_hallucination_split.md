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
