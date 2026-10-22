<p align="center">
  <img src="../logos/covers.banner.svg" width="760" alt="assets/covers — animated logo: book stack with sliding bookmark">
</p>

# 📕 assets/covers — Monograph Cover Art

<p align="center">
  <em>Publication covers for every monograph, EN and RU editions</em><br><br>
  <a href="../README.md"><img src="https://img.shields.io/badge/folder-assets-f472b6" alt="assets"></a>
  <img src="https://img.shields.io/badge/12%20covers-fbbf24" alt="12 covers"><img src="https://img.shields.io/badge/png-f472b6" alt="PNG">
</p>

Rendered cover images for the six monographs in
[`monographs/`](../../monographs/README.md), one cover per language
edition (EN / RU). These are the deliverable-grade files: they are
embedded in the monograph READMEs, used as document previews, and
referenced by the branding set. If you are looking for the *generator*
side of the visual identity (SVG banners, icons, dividers), that lives
one level up in [`assets/branding/`](../branding/README.md).

## Naming scheme

```text
cover_M<NN>_<LANG>.png      NN = 01..06 monograph number, LANG = EN | RU
```

| Monograph | Topic |
|---|---|
| M01 | Spectral analysis of RNN activations |
| M02 | arXiv preprint — spectral statistics of activation matrices |
| M03 | BlackGold preprint v1 — RMT-based cognitive-regime detection |
| M04 | BlackGold preprint v2 — extended analysis with the BBP transition |
| M05 | LLM analysis (merged) — RLHF trap, Landauer, thermodynamics of lies |
| M06 | Complex-analytical model of inevitable hallucinations |

## Conventions

- Files are committed PNGs; regenerate cover art via the maintainer's
  cover scripts rather than editing pixel files by hand, so that both
  language editions stay visually consistent.
- A second, *branding-tuned* copy of each cover (different resolution /
  compression) lives in [`assets/branding/covers/`](../branding/covers/)
  — the two are intentionally not identical; branding copies are
  optimized for web previews, these are optimized for documents.

---

<p align="center">
  <img src="../logos/covers.avatar.svg" width="56" alt="covers animated avatar"><br>
  <sub>
    <a href="../README.md">← assets</a> · <a href="../../README.md">🏠 repository root</a> · <a href="../../monographs/README.md">📚 monographs →</a>
  </sub>
</p>
