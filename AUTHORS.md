# Authors

## Primary author

**Iskhak Hamzatovich Isaev**
Independent researcher, Russia
GitHub: https://github.com/wild8highlander

Author of:
* The **RMT spectral analysis** programme — full monograph applying
  Marchenko-Pastur law, BBP phase transition, and Tracy-Widom
  fluctuations to covariance matrices of GPT-2 hidden-state activations
  for cognitive-mode detection (EN/RU, `docs/en/RMT_LLM_Spectral_Analysis.docx`).
* The **complex analytical model of inevitable hallucinations** — ten
  independent mathematical paths converging on the critical token count
  N_crit (NHSE winding number, Caputo fractional-time memory, EP-surfaces,
  Keating-Snaith, Tracy-Widom GUE, quantum channel fidelity, etc.)
  (EN/RU, `docs/en/Complex_Analytical_Model_Inevitable_Hallucinations.docx`).
* The **RLHF utility trap** analysis — Caputo memory parameter
  β ≈ 0.5 making ⟨T_crit⟩ ∝ (μ_eff)<sup>-2</sup>, with Landauer
  principle and thermodynamic irreversibility of lies
  (EN/RU, `docs/en/LLM_Analysis_Merged.docx`, `papers/LLM_Analysis_Merged.pdf`).
* The **arXiv and BlackGold preprints** (v1 and v2, EN/RU).
* The interactive in-browser demo (`docs/site/demo/demo.js`) visualising
  the Marchenko-Pastur law, BBP transition, Tracy-Widom F₂ distribution,
  and NHSE winding-number topology.
* The GitHub Pages documentation site (`docs/site/`), Zenodo DOI
  configuration, CITATION.cff, and all accompanying metadata files.

## Citation

If you use any part of this repository, please cite:

> Isaev, Iskhak Hamzatovich (2026).
> *Spectral Statistics of LLM Activation Matrices: A Random Matrix
> Theory Approach to Cognitive Mode Detection.* Version 1.2.0.
> https://github.com/wild8highlander/rmt-llm-research

A machine-readable `CITATION.cff` is provided alongside this file.

## Acknowledgements

The mathematical content builds on classical results of Marchenko &
Pastur (1967), Baik–Ben Arous–Péché (BBP transition, 2005), Tracy &
Widom (1996), Dyson (GUE, 1962), Caputo (fractional calculus, 1967),
Keating & Snaith (zeta-zero statistics, 2000), and the modern
non-Hermitian skin effect literature (Okuma–Sato–Yao, 2018+). The
GPT-2 experimental verification was facilitated by the open-source
Hugging Face Transformers library.

## How to report issues

Bugs, numerical regressions, or questions about the derivations should
be filed at https://github.com/wild8highlander/rmt-llm-research/issues.
