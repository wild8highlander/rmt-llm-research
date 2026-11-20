# Restoring the monographs to the repository

The twelve DOCX editions (and the merged PDF) referenced by
`docs/monographs/*/README.md` are not in the current git tree — they were
dropped during the `monographs/` → `docs/monographs/` reorganization
(commit `cf5cb49`, 2026-11-09, which moved only the READMEs). The files
still exist as blobs in the repository history and can be restored with:

```bash
git fetch --unshallow --filter=blob:none     # once: full commit/tree history
mkdir -p docs/monographs/M01_spectral_analysis docs/monographs/M02_arxiv_preprint \
         docs/monographs/M03_blackgold_v1 docs/monographs/M04_blackgold_v2 \
         docs/monographs/M05_llm_analysis_merged docs/monographs/M06_complex_hallucinations \
         docs/papers
# then, per file (blobs fetch lazily on first access):
git cat-file -p <blob-sha> > docs/monographs/<folder>/<file>
```

| blob | target |
| --- | --- |
| `695cb1945d3b8d79e8e7a6e7da5c13dfdf0052fc` | `docs/monographs/M01_spectral_analysis/RMT_LLM_Spectral_Analysis.docx` |
| `16ca95482613bb2420ad4083ee6bf5fd677e7892` | `docs/monographs/M01_spectral_analysis/RMT_LLM_Spectral_Analysis_RU.docx` |
| `d645cb6d07f97de4deeae3ce52e9f3fb0ac993d2` | `docs/monographs/M02_arxiv_preprint/RMT_LLM_Arxiv_Preprint.docx` |
| `a52c5d7d26474ac3da721b9ca9bc55dc2139ff52` | `docs/monographs/M02_arxiv_preprint/RMT_LLM_Arxiv_Preprint_RU.docx` |
| `3449c0345e3359604cad5b010d5102254c216e11` | `docs/monographs/M03_blackgold_v1/RMT_LLM_BlackGold_Preprint_v1.docx` |
| `ad4b873d55756a0012338129ecb12b0c579c7f1c` | `docs/monographs/M03_blackgold_v1/RMT_LLM_BlackGold_Preprint_v1_RU.docx` |
| `2615e19f9f857fe5b25191c0a099f9518df0fe3d` | `docs/monographs/M04_blackgold_v2/RMT_LLM_BlackGold_Preprint_v2.docx` |
| `5654fe2dd126e32dcc1626424dd297cf9660aa56` | `docs/monographs/M04_blackgold_v2/RMT_LLM_BlackGold_Preprint_v2_RU.docx` |
| `c04e82ae8e95009efdbaeb904b6d01adf731789c` | `docs/monographs/M05_llm_analysis_merged/LLM_Analysis_Merged.docx` |
| `8761fd74a4b9688b74c29f25d1aa7f12aee4f0df` | `docs/monographs/M05_llm_analysis_merged/LLM_Analysis_Merged_RU.docx` |
| `1c0f3fb78cce90f4fd4c01c021ecc5ea5ed446de` | `docs/monographs/M06_complex_hallucinations/Complex_Analytical_Model_Inevitable_Hallucinations.docx` |
| `3585996883f43792f96730ea5534fcb15e2759fb` | `docs/monographs/M06_complex_hallucinations/Complex_Analytical_Model_Inevitable_Hallucinations_RU.docx` |
| `52542146084d1815bb914808e1b5fd958c6fb700` | `docs/papers/LLM_Analysis_Merged.pdf` |

All thirteen files were recovered this way for the RCDA-9 round, verified
(zip container checks, Word 2007+ format, PDF 1.7), and are archived next
to this kit as `monographs_restored.zip` (72 MB).

Two content notes for future editors:

1. **M03/M04 equations are embedded as PNG images** with `Equation N-M:`
   captions; the formulas are also present as ASCII in the prose. The
   RU editions of M03/M04 additionally carry machine-translation damage
   («творческая рация», «ambda_max»); consider re-exporting those two.
2. The formula corpus of RCDA-9 (`data/formula_corpus_*.txt`) was built
   from these files — see `data/corpus_manifest.json` for the extraction
   rules and the E[F1]/E[F2] RULES fix.

---

<p align="center">
  <sub><a href="README.md">↑ the RCDA-9 round</a> ·
  <a href="RCDA9_REPORT.md">the round report →</a></sub>
</p>
