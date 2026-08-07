# Changelog

All notable changes to this repository are documented in this file.

## [1.3.0] - 2026-08-08

### Added

- **Python verification package** (`src/rmt_llm/`) with 8 modules:
  `marchenko_pastur`, `bbp_transition`, `tracy_widom`, `nhse`,
  `caputo_fractional`, `keating_snaith`, `ep_surfaces`, `thermodynamics`,
  plus `constants` and `__init__`
- **70+ pytest tests** covering all 8 modules plus 7 cross-module
  consistency checks (`src/rmt_llm/tests/test_rmt_llm.py`)
- **Julia verification package** (`julia/RMTLLMVerify/`) with matching
  implementations of all 8 mathematical objects and comprehensive test
  suite (`julia/RMTLLMVerify/test/runtests.jl`)
- **Jupyter verification notebook** (`notebooks/rmt_llm_verification.ipynb`)
  with 7 sections: Marchenko-Pastur, BBP transition, Tracy-Widom, NHSE,
  Caputo dynamics, Keating-Snaith & EP surfaces, cross-implementation
  consistency
- **pyproject.toml** with ruff/mypy/pytest/coverage configuration
- **CI workflow** (`.github/workflows/ci.yml`): lint (ruff + mypy),
  pytest+coverage+Codecov on Python 3.10/3.11/3.12, Julia 1.9/1.10
  tests, cross-implementation consistency check job
- **OpenSSF Scorecard workflow** (`.github/workflows/scorecard.yml`)
- **CODE_OF_CONDUCT.md** (Contributor Covenant v2.1)
- ORCID identifier `0009-0003-7299-0701` added to:
  - `.zenodo.json` (creators[0].orcid)
  - `CITATION.cff` (authors[0].orcid)
  - `README.md` (ORCID badge + footer link)
- Mermaid architecture diagram in README.md
- Scorecard badge and Codecov badge in README.md
- pytest count badge and Julia package badge in README.md

### Changed

- Version bumped from 1.2.0 to **1.3.0**
- `.zenodo.json`: removed invalid `communities` field, added ORCID,
  updated version and publication_date
- `CITATION.cff`: added ORCID, updated version and date-released
- Repository structure section in README.md expanded to include
  verification package, Julia package, and notebooks

## [1.2.0] - 2026-08-06

### Added

- GitHub Pages documentation site (`docs/site/`) with modern"modern dark-theme HTML,
  hero section, navigation, key results tables, document catalogue, and BibTeX
  citation
- Zenodo DOI auto-archive workflow (`.github/workflows/zenodo.yml`) — uploads
  source archive to Zenodo on every GitHub release; produces a citable DOI
- Machine-readable `CITATION.cff` and human-readable `AUTHORS.md`
- Interactive demo web app (`docs/site/demo/`) — Marchenko-Pastur law
  visualizer, BBP phase transition slider, Tracy-Widom distribution plot,
  and NHSE winding-number simulator
- Visible author name printed on the title page of every DOCX monograph
- Author metadata set in all DOCX (`dc:creator`) and PDF (`author`) files

### Changed

- Author name corrected to the proper transliteration
  `Iskhak Hamzatovich Isaev` (First Middle Last format) across:
  - README.md (BibTeX `author = {}` field and footer by-line)
  - LICENSE (copyright line)
  - CHANGELOG.md (attribution note)
  - All 12 DOCX files (metadata + visible title-page paragraph)
  - All 1 PDF file (metadata.author)

## [1.1.0] - 2026-08-01

### Added

- Russian translation of `RMT_LLM_Arxiv_Preprint` (EN → RU)
- Russian translation of `RMT_LLM_BlackGold_Preprint_v1` (EN → RU)
- Russian translation of `RMT_LLM_BlackGold_Preprint_v2` (EN → RU)
- Every document is now available in both English and Russian (12 documents total: 6 EN + 6 RU)
- Updated README documents table and badge count (9 → 12)

### Changed

- Author attribution corrected to `Iskhak Hamzatovich Isaev` across README.md, LICENSE, and git commit metadata

## [1.0.0] - 2026-07-29

### Added

- Full monograph: RMT spectral analysis of LLM activations (EN + RU)
- Complex analytical model of inevitable hallucinations (EN + RU)
- LLM analysis merged: RLHF utility trap and thermodynamics (EN + RU)
- RMT LLM arxiv preprint (EN)
- RMT LLM BlackGold pre1 and v2 (EN)
- LLM Analysis Merged PDF
- Professional README with key results table and citation
- LICENSE (CC BY-NC-SA 4.0), .gitignore, CONTRIBUTING.md
