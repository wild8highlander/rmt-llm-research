<p align="center">
  <a href="../README.md"><img src="../assets/branding/aurora-icon.svg" width="72" alt="RMT × LLM Research — Black Aurora"></a>
</p>

# 📚 monographs — Standalone Research Editions

<p align="center">
  <em>6 self-sufficient studies · 12 documents · DOCX · RU / EN · Aurora edition</em><br><br>
  <img src="../assets/branding/aurora-divider.svg" width="420" alt=""><br><br>
  <a href="../README.md"><img src="https://img.shields.io/badge/folder-monographs-a78bfa" alt="repository root"></a>
  <img src="https://img.shields.io/badge/6%20monographs-f472b6" alt="6 monographs"><img src="https://img.shields.io/badge/12%20docx-8ca3c7" alt="12%20docx"><img src="https://img.shields.io/badge/aurora%20edition-00ffa3" alt="aurora%20edition">
</p>

<p align="center">
  <img src="../assets/branding/covers/cover_M01_EN.png" width="230" alt="Aurora cover M01">
  &nbsp;&nbsp;
  <img src="../assets/branding/covers/cover_M05_RU.png" width="230" alt="Aurora cover M05 RU">
  &nbsp;&nbsp;
  <img src="../assets/branding/covers/cover_M06_EN.png" width="230" alt="Aurora cover M06">
</p>

Each research paper of the project is packaged as a **standalone monograph**:
a full-page Aurora cover, the original study, and — new in this edition — a
repository supplement that maps every chapter onto working code, registers the
measured numbers and reproduces the figure atlas. The monographs can be read,
cited and used individually — at any time and in any order.

---

## 🗂 Full index

| № | Folder | Monograph | Core contribution |
|---|--------|-----------|-------------------|
| 1 | [`M01_spectral_analysis/`]( M01_spectral_analysis/ ) | **M01 · Random Matrix Theory and Activations of LLMs** | the foundational monograph: MP / BBP / Tracy–Widom machinery, spectral entropy, free-energy analogy and the RG flow across layers — 8 chapters + repository supplement |
| 2 | [`M02_arxiv_preprint/`]( M02_arxiv_preprint/ ) | **M02 · Spectral Statistics of LLM Activation Matrices** | the arXiv-style pilot: GPT-2 small (117M), the 9-experiment spectral suite, real-GPT-2 verification and the three unexpected findings |
| 3 | [`M03_blackgold_v1/`]( M03_blackgold_v1/ ) | **M03 · RMT Meets Large Language Models (v1)** | the full preprint, version 1: temperature sweep, cross-layer transfer, IPR profiles, prompt-length dependence and F = U − TS_spec |
| 4 | [`M04_blackgold_v2/`]( M04_blackgold_v2/ ) | **M04 · RMT Meets Large Language Models (v2)** | the extended preprint: comprehensive layer × metric heatmap and the empirical scale-out (TinyGPT ladder, polyglot laboratories, 600-dpi atlas) |
| 5 | [`M05_llm_analysis_merged/`]( M05_llm_analysis_merged/ ) | **M05 · The Merged Monograph: RMT & LLM** | spectral analysis of activations, the mathematical inevitability of hallucinations and the RLHF utility trap — the whole story in one narrative; merged PDF edition in papers/ |
| 6 | [`M06_complex_hallucinations/`]( M06_complex_hallucinations/ ) | **M06 · Complex Analytical Model of Inevitable Hallucinations** | ten independent paths to N_crit: NHSE winding numbers, Caputo fractional dynamics, EP-surfaces, Keating–Snaith correction, Tracy–Widom and Kraus channels |

---

## 📐 The uniform structure of every edition

All 12 documents (DOCX in both languages) follow one scheme:

```
┌────────────────────────────────────────────────────────────┐
│  COVER — full-page Black Aurora plate (M0X / 06, RU or EN) │
├────────────────────────────────────────────────────────────┤
│  ORIGINAL STUDY — the paper exactly as written, with its   │
│  own TOC, tables and figures; headings recolored to the    │
│  Aurora spectrum (emerald → cyan → violet → pink)          │
├────────────────────────────────────────────────────────────┤
│  AURORA SUPPLEMENT — the repository part:                  │
│    · theory → module map (src/rmt_llm/, laboratory/)       │
│    · measured-results registry with provenance             │
│    · figure atlas: 600-dpi plates from assets/charts/      │
│    · notation cross-reference + reproduction checklist     │
└────────────────────────────────────────────────────────────┘
```

Files in each folder:

| File | Content |
|---|---|
| `<monograph>.docx` | the full study + Aurora supplement (English, DOCX) |
| `<monograph>_RU.docx` | the same edition in Russian |
| `README.md` | this navigation page: essence, key numbers, companions |

The 600-dpi cover plates live in
[`assets/branding/covers/`](../assets/branding/README.md); the figure atlas in
[`assets/charts/`](../assets/charts/README.md).

---

## 🌌 What the Aurora edition adds

- **Restyled identity** — legacy gold/blue accents are remapped to the Black
  Aurora spectrum; every document opens with a full-page dark cover.
- **Repository supplement** — a new closing part (10+ pages per edition) that
  the original papers could not contain: the theory-to-code map for
  `src/rmt_llm/`, the polyglot laboratory (7 languages × en/ru), the 480-test
  verification perimeter, the three-arm TinyGPT ladder and the 13-scenario
  behavioural registry.
- **Measured registries** — verbatim numbers with provenance
  (`laboratory/results/reports/`, `scaling_report.json`, `scenarios.json`) so
  future runs can be diffed against the printed values.
- **Figure atlas** — the 600-dpi plates of [`assets/charts/`](../assets/charts/README.md)
  embedded with tailored captions per monograph.

---

## 🧭 Reading paths

```
M02 pilot ──────→ M03 v1 ──────→ M04 v2        (the empirical campaign)
   │
M01 theory ──────→ M06 ten paths               (the mathematical core)
   │
M05 merged ──────→ papers/LLM_Analysis_Merged.pdf   (the whole story)
```

Every monograph is self-sufficient: reading its proof requires none of the
others — the cross-references are navigation, not dependency.

---

## 🔬 Verification

| Monograph | Laboratory entry | Core tests |
|---|---|---|
| M01 theory | `run_full_lab.py`; `probe_real_gpt2.py` | `pytest src/rmt_llm/tests -q` |
| M02 pilot | `probe_real_gpt2.py`; Colab cells → lab modules | same suite |
| M03 / M04 preprints | `run_full_lab.py`; `research_3d.py`; `scaling_experiment.py` | 480 tests total |
| M05 merged | 13 scenarios of `shared/scenarios.json`; ladder | 379 lab tests |
| M06 ten paths | module-by-module mechanism tests | 101 core tests |

Exit codes of all stands: `0` — accepted, `1` — failure.

---

## 📖 How to cite

```bibtex
@misc{isaev2026rmtllm,
  author = {Isaev, Iskhak Khamzatovich},
  title  = {RMT \& LLM: Standalone Research Monographs (M01--M06)},
  year   = {2026},
  howpublished = {rmt-llm-research repository},
  url    = {https://github.com/wild8highlander/rmt-llm-research}
}
```

The repository-level citation lives in [`../CITATION.cff`](../CITATION.cff)
(Zenodo DOI `10.5281/zenodo.21825389`) and the main
[`../README.md`](../README.md).

---

## ⚖️ License

See [`../LICENSE`](../LICENSE) and the authorship notes in
[`../AUTHORS.md`](../AUTHORS.md). The monographs are the theoretical backbone
of the repository — cite them through the DOI above.

---

<p align="center">
  <sub>
    <a href="../papers/README.md">← papers</a> · <a href="README.md">top</a> · <a href="../README.md">↑ parent</a> · <a href="../README.md">🏠 repository root</a> · <a href="M01_spectral_analysis/">M01 →</a>
  </sub>
</p>
