<p align="center">
  <img src="github.avatar.svg" width="72" alt=".github avatar">
  <img src="root_scripts.avatar.svg" width="72" alt="scripts avatar">
  <img src="covers.avatar.svg" width="72" alt="covers avatar">
  <img src="nl_src.avatar.svg" width="72" alt="src avatar">
  <img src="nl_assets.avatar.svg" width="72" alt="assets avatar">
  <img src="sc_data.avatar.svg" width="72" alt="data avatar">
  <img src="sc_eval.avatar.svg" width="72" alt="evaluation avatar">
  <img src="sc_exp.avatar.svg" width="72" alt="experiments avatar">
  <img src="sc_feat.avatar.svg" width="72" alt="features avatar">
  <img src="sc_hyp.avatar.svg" width="72" alt="hypotheses avatar">
  <img src="sc_instr.avatar.svg" width="72" alt="instrumentation avatar">
  <img src="sc_pred.avatar.svg" width="72" alt="predictors avatar">
  <img src="sc_res.avatar.svg" width="72" alt="results avatar">
  <img src="sc_store.avatar.svg" width="72" alt="storage avatar">
  <img src="sc_tests.avatar.svg" width="72" alt="tests avatar">
  <img src="lab_charts.avatar.svg" width="72" alt="charts avatar">
  <img src="lab_charts3d.avatar.svg" width="72" alt="charts 3D avatar">
  <img src="lab_reports.avatar.svg" width="72" alt="reports avatar">
  <img src="runs.avatar.svg" width="72" alt="runs avatar">
</p>

# 🌈 logos — Repo-wide Animated SVG Set

Animated SVG logos for the folder READMEs outside `neural_lab/`: one
**banner** (800×220, header of each README) and one **avatar** (128×128,
footer of each README) per section. Pure-SMIL animation only (no CSS, no
JS, no external resources), so the logos animate on GitHub and in any
standards-compliant viewer. Motifs: terminal (scripts), shield (.github),
books (covers), brackets (src), palette (assets), database (data), gauge
(evaluation), rocket (experiments), spectrum (features), bulb (hypotheses),
oscilloscope (instrumentation), target (predictors), trophy (results),
archive cube (storage, runs), checklist (tests), frames (2D charts),
wireframe cube (3D charts), document stack (reports).

| Pattern | Usage |
|---|---|
| `<name>.banner.svg` | top header of the corresponding README, `width="760"` |
| `<name>.avatar.svg` | footer navigation block, `width="56"`, or standalone icon |

The `neural_lab`-specific set (9 sections: neural_lab, research,
tinygpt_formula, scripts, data, model, scaling, colab, science) lives in
[`neural_lab/assets/logos/`](../../neural_lab/assets/logos/README.md).
Regenerate both sets with the maintainer's `gen_neural_lab_logos.py`
(single source of truth); the files are committed artifacts.

[← back to assets](../README.md)
