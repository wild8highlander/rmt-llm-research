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
