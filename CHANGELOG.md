# Changelog

All notable changes to this repository are documented in this file.

## [1.2.0] - 2026-08-06

### Added

- GitHub Pages documentation site (`docs/site/`) with modern dark-theme HTML,
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
- RMT LLM BlackGold preprint v1 and v2 (EN)
- LLM Analysis Merged PDF
- Professional README with key results table and citation
- LICENSE (CC BY-NC-SA 4.0), .gitignore, CONTRIBUTING.md
