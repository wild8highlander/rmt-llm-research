# neural_lab/stages — changelog

Protocol-level changes live in `EXPERIMENT_PROTOCOLS.md` (the passport
pins its sha256); this file tracks the same changes in short form.

## v1.1 — measurement-integrity revision

`results/smoke_pythia70m/` regenerated; v1.0 artifacts are NOT comparable.

- `score_continuation` returns real log-probabilities (was raw logits).
- Permutation p-values carry the +1 correction (never exactly 0).
- `probe_hidden` falls back to within-benchmark grouped OOF for
  single-benchmark runs; LOBO stays the ≥2-benchmark criterion.
- Stage-5 pairs write stage-3-compatible `fit.json`; `--scaling-in`
  accepts stage-5 outputs; `--mu-eff-from` added.
- Generation step positions survive left truncation (NaN + warning).
- Honest `patching_profile` (dead pre-pass removed, length-mismatch
  warning); changepoint null respects at-risk exposure; β without the
  (0,1) ceiling, real bootstrap CI under `--mu-eff`; HF-hub weight
  hashing; git commit + protocol doc hash in every passport.

## v1.1 (infrastructure, same protocol semantics)

- `results_schema/v1/`: JSON-Schema validation at write time for
  report.json / fit.json / record.json / scaling.json; artifacts carry
  `schema_version`.
- `common/cli.py`: shared StageRunner for all four stages — `--config`
  (YAML, CLI overrides), `--resume` (stage2 units / stage3 trajectories),
  `--only` (stage4 steps: lens|patching|heavy-tail|sae), `--dry-run`,
  `--verbose/--quiet/--log-json`, tqdm progress.
- `science/hypotheses/check.py`: the S2–S5 registry criteria evaluated
  in code at the end of every run (section "Pre-registered criterion").
- `common/index.py`: sqlite index (`make index-results`).
- `compare_reports.py`: differential reports between two runs.
- `KNOWN_DETECTORS` registry + `Unit.score()`; unknown detector names
  fail at construction.
- CI: `.github/workflows/stages.yml` (offline tests on push, nightly
  smoke + validation); pyproject extras `stages`, `stages-sae`,
  `stages-dev`; `make smoke / validate-results / index-results /
  compare-reports / test-stages-slow`.
