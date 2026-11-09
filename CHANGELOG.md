# Changelog

All notable changes to this repository are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions match `pyproject.toml`. Early history lives in the git log.

## [Unreleased]

### Added
- `rmt-llm-verify` CLI is back and wired to the NumPy core:
  `cli.py` + `__main__.py` implement `info`, `verify`, `mp`, `bbp`, `tw`,
  `nhse`, `spacing`. `verify` is a self-check gate used by CI.
- `__version__` exported from `rmt_llm` (kept in sync with `pyproject.toml`).
- `.github/workflows/cli-doctor.yml` — the self-healing bot workflow:
  runs the CLI gate and the markdown link check on schedule, opens an issue
  with diagnostics on failure, and labels them `ci:cli` / `ci:links`.
- Root community files: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`,
  `CHANGELOG.md`.

### Fixed
- 41 broken relative links across `README.md`, `assets/`, `docs/` and
  `laboratory/` after the `monographs/` and `papers/` trees moved to
  `docs/`.
- Missing READMEs added for `neural_lab/MNN-Pocket-0.3B` (Code / Docs /
  Graphic / Model ZIP), `neural_lab/stages/*`, `neural_lab/h1_weights`,
  `formal_proof/{experiments,rmt_fp,assets}`, monograph chapter folders,
  `laboratory/webapp/{src,server}`, `laboratory/results/*` and other
  directories listed in the cleanup report.

### Notes
- `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md` were referenced by the root
  README before they existed in the repository; the files are now real.
