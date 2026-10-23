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
