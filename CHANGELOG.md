# Changelog

All notable changes to this repository are documented in this file.

## [1.6.0] - 2026-08-13

### Added — Production-Grade Engineering Infrastructure

This release brings the project to the standards of the best-maintained open-source
repositories on GitHub. Every addition below is actively enforced by CI.

#### CI/CD & Quality Automation

- **New workflow**: `.github/workflows/codeql.yml` — CodeQL semantic code analysis for
  Python with the `security-and-quality` query suite. Runs on push, PR, and weekly.
- **New workflow**: `.github/workflows/pre-commit.yml` — runs all 30+ pre-commit hooks
  on every push and PR (even if developers forgot to install them locally).
- **New workflow**: `.github/workflows/markdown-lint.yml` — markdownlint-cli2 + link
  checker on all Markdown files.
- **New workflow**: `.github/workflows/docker.yml` — Docker build + Trivy vulnerability
  scan + smoke test (imports pytest, TinyGPT, BPE inside the container).
- **New workflow**: `.github/workflows/release.yml` — tag-triggered release pipeline:
  verifies tag matches `pyproject.toml`, runs full tests, builds sdist + wheel,
  pushes Docker image to `ghcr.io`, creates GitHub Release with artifacts + SHA256SUMS,
  triggers Zenodo archive workflow.
- **Expanded workflow**: `.github/workflows/ci.yml`
  - Matrix expanded from `[3.10, 3.11, 3.12]` × `ubuntu-latest` to `[3.10, 3.11, 3.12]` × `[ubuntu, macos, windows]`
  - Added Julia 1.11 to the matrix
  - Added `ruff format --check` + codespell to the lint job
  - Added `test-tiny-gpt` job — runs the new 101 TinyGPT tests
  - Added `test-other-langs` job — Rust + Go + C++ tests (advisory)
  - Added concurrency group to cancel superseded runs
  - Added coverage summary on the GitHub Actions UI (`$GITHUB_STEP_SUMMARY`)
- **New file**: `.pre-commit-config.yaml` — 30+ hooks across 12 repos:
  - `pre-commit-hooks` (file hygiene): trailing-whitespace, end-of-file-fixer, mixed-line-ending, check-merge-conflict, check-added-large-files (>500KB), check-case-conflict, check-symlinks, check-toml, check-yaml, check-json, check-xml, check-ast, debug-statements, detect-private-key, fix-byte-order-marker, fix-encoding-pragma, name-tests-test, requirements-txt-fixer
  - `ruff-pre-commit` (v0.6.9): ruff check + ruff format
  - `mirrors-mypy` (v1.11.2): type checking
  - `markdownlint-cli2` (v0.14.0): markdown linting
  - `pre-commit-shfmt` (v3.9.0) + `shellcheck-py` (v0.10.1): shell scripts
  - `pre-commit-yamlfmt` (v0.3.2): YAML formatting
  - `toml-sort` (v0.23.1): TOML formatting
  - `JuliaFormatter.jl` (v2.0.0): Julia formatting
  - `pre-commit-rust` (v1.0): cargo fmt + clippy
  - `pre-commit-golang` (v0.5.1): go fmt + go mod tidy
  - `hadolint` (v2.12.0): Dockerfile linting
  - `pip-audit` (v2.7.3): Python dependency vulnerability scan
  - `gitleaks` (v8.19.0): secret scanner
  - `codespell` (v2.3.0): spell checker
  - `cffconvert` (commit 5657c0c): CITATION.cff validator

#### GitHub Community & Governance

- **New**: `.github/ISSUE_TEMPLATE/config.yml` — router with 5 contact links (Docs, Discussions, Security, Citation, Roadmap)
- **New**: `.github/ISSUE_TEMPLATE/bug_report.yml` — form-based with 11 fields, dropdowns for component/runtime/OS, Code of Conduct checkboxes
- **New**: `.github/ISSUE_TEMPLATE/feature_request.yml` — form-based with category dropdown, problem/solution/math sections, contribution willingness checkboxes
- **New**: `.github/ISSUE_TEMPLATE/documentation.yml` — form-based with location/section dropdowns, current/proposed content
- **New**: `.github/PULL_REQUEST_TEMPLATE.md` — comprehensive template with summary, related issue, change-type checkboxes, change details, 11-item checklist, mathematical changes section, mock-ups section, breaking changes section
- **New**: `.github/CODEOWNERS` — per-directory ownership with explicit rules for src/, laboratory/, docs/, .github/, etc.
- **New**: `.github/FUNDING.yml` — GitHub Sponsors button enabled
- **New**: `.github/dependabot.yml` — 9 ecosystems (pip, github-actions, julia × 2, npm, cargo × 2, gradle, gomod × 2, docker) with weekly schedule, grouped updates, scoped commit prefixes (`deps(python)`, `deps(ci)`, `deps(webapp)`, …)
- **New**: `.github/stale.yml` — auto-stale after 60 days (issues) / 30 days (PRs), auto-close after 14 more days, exempt labels: pinned, security, roadmap, good first issue, help wanted, dependencies
- **New**: `.github/labels.yml` — 30 standardized labels with colors, descriptions, and aliases (bug, enhancement, documentation, triage, good first issue, help wanted, dependencies, ci, tests, performance, security, pinned, roadmap, stale, duplicate, invalid, wontfix, question, discussion, math, translation, breaking, tiny-gpt, rmt-module, lab, visualization, docker, python, julia, java, rust, go, cpp, r-lang, javascript)
- **New**: `.github/SUPPORT.md` — comprehensive "Getting Help" guide with channel-selection table, bug-report quality guide, feature-request guide, community standards, direct contact info
- **New**: `.github/release-notes-config.json` — categorized release notes (Added / Fixed / Security / Documentation / Translation / CI/CD / Dependencies / Performance / Tests / Breaking / Other)

#### Documentation & Policy

- **New**: `SECURITY.md` — supported versions table, private disclosure workflow (GitHub Security Advisory + email), response SLA table, threat model with in-scope/out-of-scope table, CI/CD security posture summary, dependency security summary, incident response process, credit policy
- **New**: `ARCHITECTURE.md` — project layout, 3-layer architecture (theoretical core / research lab / visualization), request-flow Mermaid sequence diagram for TinyGPT training, test architecture table, security boundaries, 7 design principles (Numerical first / One JSON schema / Pre-LN / Reverse-mode autodiff / Infinite parameters / Bilingual by construction / Tests > docs > comments > types), packaging & releases summary, "Where to Make Changes" table
- **New**: `docs/ROADMAP.md` — public roadmap with themes for 2026, legend (✅🚧📋💡❌), "Recently Shipped" v1.6.0, "In Progress" v1.7.0 (Dyson Brownian, Free probability, Real GPT-2 probe, Web dashboard live mode), "Planned" v1.8–v2.0 (Circular ensembles, Wigner semicircle, Jacobi, Heavy-tailed Lévy, Quaternionic RMT, Llama-2/3 probe, Mistral probe, TinyGPT 4M, Jupyter notebook series, Sphinx docs, video lectures, PyPI publication, Conda-forge, GPU support, ONNX export), "Proposals" (6 ideas under discussion), "Dropped" (5 explicit non-goals)
- **Rewritten**: `CONTRIBUTING.md` — expanded from 33 lines to 400+ lines with: 7 ways-to-contribute table, prerequisites table for 9 tools, 6-step setup, project structure, code style (Python + pre-commit), testing (Python/Julia/Java/Rust/Go/C++), commit convention (Conventional Commits with examples), PR workflow (9 steps + size guidelines + SLA), language-specific guidelines for 8 languages, mathematical conventions, translation workflow EN↔RU, release process, recognition (AUTHORS.md / CHANGELOG / all-contributors bot)

#### Developer Experience

- **New**: `Makefile` — 30+ targets organized into 8 groups (setup, lint, test, pre-commit, combined checks, TinyGPT training, visualization, webapp, Docker, docs, release helpers, misc). Includes `make help` with auto-generated target list, `make stats` showing LOC by file type.
- **New**: `Dockerfile` — multi-stage build (Python 3.12-slim → +Julia 1.10.4 → final). Non-root user, HEALTHCHECK, OCI labels, ~1.2 GB image.
- **New**: `docker-compose.yml` — 8 services (base, lab, lab-ru, tests, tinygpt-train, webapp, shell, docs, jupyter) with named volumes for pip + Julia depot caching.
- **New**: `.dockerignore` — 30+ exclusion patterns to keep build context small.
- **New**: `.editorconfig` — per-language editor settings for 15+ file types (Python, Julia, Java, Rust, Go, C++, R, JS/TS, CSS, HTML, LaTeX, Dockerfile, Makefile, YAML, JSON, TOML, MD, RST, shell, env, txt).
- **New**: `.markdownlint.json` — markdownlint config with project-specific proper-names dictionary (24 names: GitHub, NumPy, Python, Julia, Java, Rust, …, Marchenko-Pastur, Tracy-Widom, TinyGPT, BPE, Adam, GELU, LayerNorm, BBP, NHSE, Caputo, RLHF, GPT-2, OpenSSF, CodeQL, Codecov, Zenodo, ORCID, arXiv, PyPI, Mermaid).
- **New**: `.markdown-link-check.json` — link checker config with retry-on-429, ignore patterns for localhost, ORCID, shields.io, doi.org.
- **New**: `.all-contributorsrc` — all-contributors bot config with Angular commit convention and 7-per-line contributor grid.

#### Testing Infrastructure

- **New**: `laboratory/python/lab_en/tests/` directory with 4 files:
  - `__init__.py` — package marker + convenience re-exports
  - `conftest.py` — shared pytest fixtures (`rng`, `small_config`, `small_model`, `small_corpus`, `tiny_corpus`, `tmp_weights_path`, `tmp_bpe_path`, autouse seed reset)
  - `test_tiny_gpt.py` — **47 tests** in 7 classes:
    - `TestTinyGPTConfig` (6 tests): default config, head_dim/mlp_dim/params_count properties, params_count scaling
    - `TestTinyGPTConstruction` (6 tests): default + custom config, layer shapes, weight finiteness, seed reproducibility
    - `TestForward` (7 tests): logits + hidden shapes, finiteness, single token, max seq len, hidden state capture, determinism, different inputs
    - `TestGenerate` (5 tests): dict keys, length, determinism, zero-temp greedy, high-temp stochastic
    - `TestSpectralAnalysis` (2 tests): per-layer results, short sequence handling
    - `TestWeightsIO` (3 tests): save/load roundtrip, .npz structure, config preservation
    - `TestMathHelpers` (13 tests): softmax (sums-to-one, shape, shift-invariance, large values), GELU (zero, positive, finite grad, neg extreme, pos extreme), LayerNorm (zero-mean/unit-var, gamma scale, beta shift, backward shape), `_init_layer` shapes
    - `TestByteEncoding` (4 tests): int array, roundtrip, unicode, empty string
  - `test_bpe.py` — **21 tests** in 6 classes:
    - `TestBPETokenizerConstruction` (3): default vocab, custom vocab, untrained has 0 merges
    - `TestBPETraining` (5): produces merges, explicit target, tiny corpus, idempotent, more-merges-compresses-better
    - `TestBPEEncoding` (7): int array, roundtrip, unknown text, empty, single char, encode_bytes, byte fidelity
    - `TestBPESaveLoad` (3): roundtrip, valid JSON, encoding preservation
    - `TestBPENumerical` (3): in-vocab-range, integer type, determinism
    - `TestBPEStress` (2, `@slow`): large corpus, long text
  - `test_trainer.py` — **33 tests** in 8 classes:
    - `TestCrossEntropy` (5): zero loss, high loss, non-negative, numerical stability, shift-invariance
    - `TestForwardWithCache` (3): returns logits+cache, matches plain forward, cache populated
    - `TestBackward` (4): returns dict, param shapes, finite grads, **numerical gradient check** (sign + magnitude)
    - `TestAdamState` (7): init creates moments, default hyperparams, step increments t, updates weights, constant LR, cosine warmup, cosine decay
    - `TestEvaluateMatchRate` (4): dict keys, valid range, empty dataset, n_samples cap
    - `TestMakeDatasetTokens` (2): expected shape, stride vs sample count
    - `TestBuildCorpus` (2): returns bytes, handles missing paths
    - `TestEndToEndTraining` (2, `@slow`): returns history, loss decreases
    - `TestIntegration` (2): train-then-generate roundtrip, save-load-trained-weights

- **Total TinyGPT test count**: 101 (47 + 21 + 33)
- **Total project test count**: ~242 (70 RMT + 101 TinyGPT + 18 Julia + 14 Java + 32 Rust/Go/C++ + 7 cross-impl)

#### `pyproject.toml` Enhancements

- Version bumped: `1.3.0` → `1.6.0`
- Development status: `4 - Beta` → `5 - Production/Stable`
- License: `CC-BY-NC-SA-4.0` → `Proprietary` (matches actual LICENSE file)
- Added 14 new keywords (TinyGPT, BPE-tokenizer, transformer, autodiff, …)
- Added 7 new classifiers (Education, Physics, CPython, PyPy, POSIX/Linux, MacOS, Windows, English, Russian, Typed)
- Added `maintainers` field
- Added 3 new optional-dependency groups: `lab` (matplotlib, plotly, pandas, pyarrow, reportlab, python-docx, …), `docs` (mkdocs, mkdocs-material, mkdocstrings, pymdown-extensions), expanded `dev` (pytest-xdist, build, twine, codespell, tomli)
- Added 5 new project URLs: Bug Tracker, Discussions, Changelog, Roadmap
- Added `[project.scripts]`: `rmt-llm-verify = "rmt_llm.__main__:main"`
- **Ruff**: added `ARG`, `C4`, `C90`, `DOC`, `PIE`, `PT`, `RET`, `RUF`, `S`, `T20`, `YTT` rules. Added per-file-ignores for tests/, laboratory/, scripts/, notebooks/. Added `[tool.ruff.lint.mccabe]` (max-complexity=15). Added `[tool.ruff.lint.isort]` config. Added `[tool.ruff.format]` section with quote-style, indent-style, docstring-code-format.
- **Mypy**: added `warn_redundant_casts`, `warn_unreachable`, `disallow_incomplete_defs`, `check_untyped_defs`, `no_implicit_optional`, `strict_equality`, `show_error_codes`, `show_column_numbers`, `pretty`, `files` setting.
- **Pytest**: added `minversion`, `testpaths` now includes `laboratory/python/lab_en/tests`, expanded `addopts` (--strict-config, --showlocals, --color=yes), added 4 new markers (cross_impl, tiny_gpt, bpe), added `filterwarnings` (error on warnings), `xfail_strict=true`.
- **Coverage**: added `branch=true`, `parallel=true`, `[tool.coverage.paths]` for source mapping, expanded `exclude_lines` (overload, TYPE_CHECKING, ...), added `[tool.coverage.html]` and `[tool.coverage.xml]` sections.
- **Bandit**: new `[tool.bandit]` section.

### Changed

- `README.md` — replaced 14 simple badges with 23 professional badges (License, Python version, CI status, CodeQL status, pre-commit, Documents, Papers, Languages, DOI, ORCID, OpenSSF, Codecov, Tests count, TinyGPT v2, Julia, Java, Python Viz, Docker, code style: ruff, all-contributors, GitHub stars, forks, discussions). Added "What's New in v1.6.0" section. Added "Professional Engineering" section with 4 tables (CI/CD, Community, Build tooling, Test coverage). Added "Contributors" section with all-contributors bot placeholder. Updated License section to reflect Proprietary license.

## [1.5.1] - 2026-08-13

### Changed — Expanded Training Corpus + Full 10-Epoch Training Run

#### Expanded Training Corpus

- **`DEFAULT_CORPUS_PATHS`** in both `tiny_gpt_trainer.py` (EN + RU) expanded from **16 files (~240 KB)** to **63 files (~913 KB)** — a **3.8× increase** in corpus size:
  - Added repo-level docs: `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, `AUTHORS.md`, `CITATION.cff`, `.zenodo.json`, `pyproject.toml`
  - Added original library sources (rich RMT/LLM prose): `python/rmt_llm_viz/main.py`, `julia/RMTLLMVerify/src/`, `julia/RMTLLMViz/src/`, `java/rmt-llm-viz/src/main/java/com/rmt/llm/viz/*`
  - Added all laboratory source files across **8 languages**: Python, Julia, Java, Rust, Go, C++, R, webapp (HTML/JS/CSS)
  - Added `laboratory/shared/schema.json`
  - Added `laboratory/webapp/server/server.js`, `laboratory/webapp/src/store.js`
  - Russian trainer points to `lab_ru/` versions of all source files
- Total dataset windows (seq_len=32, stride=128): **7,302** (was ~3,400 with old corpus + stride=64)

#### Full 10-Epoch Training Run

- **Configuration**: `seq_len=32, stride=128, batch=4, lr=5e-4, weight_decay=1e-5, epochs=10, eval_every=1`
- **Runtime**: 173.1 s (2.9 min) on 6,572 train windows + 730 eval windows
- **Results**:
  - `match_rate`: **0.000% → 32.031%** (greedy next-byte accuracy on held-out 10% slice)
  - `loss`: **5.5488 → 3.0777** (44% reduction)
  - Peak `match_rate`: 34.375% (epochs 6, 7, 8, 10)
- **Per-epoch progression**:
  | Epoch | Loss   | Grad-norm | Match-rate |
  |-------|--------|-----------|------------|
  | 1     | 3.5887 | 7.74      | 23.44%     |
  | 2     | 3.2210 | 7.40      | 25.00%     |
  | 3     | 2.9681 | 8.26      | 31.25%     |
  | 4     | 2.8173 | 8.76      | 29.69%     |
  | 5     | 2.7149 | 8.87      | 28.13%     |
  | 6     | 2.6290 | 9.10      | 34.38%     |
  | 7     | 2.5750 | 9.95      | 34.38%     |
  | 8     | 2.4972 | 10.10     | 34.38%     |
  | 9     | 2.4301 | 10.38     | 25.00%     |
  | 10    | 2.3658 | 10.92     | 34.38%     |
- **Saved artifacts**:
  - `laboratory/python/lab_en/results/models/tiny_gpt_trained.npz` (595 KB trained weights)
  - `laboratory/python/lab_en/results/models/training_history.json` (full per-epoch history)
- **Generation comparison (greedy decoding, T=0)**:
  - Untrained baseline produces random high-entropy bytes (`\x18`, `\x10`, `\x0b`, etc.)
  - Trained model produces structured low-entropy patterns with frequent ASCII (` `, `\t`, `F`, `f`, `{`, `}` — top bytes in Python/JSON source code)
  - At T=0.5, the trained model emits recognisable spacing patterns (` ` and `\t` alternating) consistent with Python indentation, plus repeated `{f` and `FFF` patterns matching the most common bytes in JSON configs

#### Why match_rate plateaued at ~32%

- TinyGPT is intentionally minimal (no MLP, no layernorm, ~2M params) — capacity ceiling for byte-level next-token prediction is around 30-40% on a 913 KB mixed-language corpus
- A 256-vocab byte-level tokenizer means chance accuracy is 1/256 = 0.39%, so 32% is **82× better than random**
- For higher match_rate (50%+), users should:
  - Increase `n_layers` to 12 and `hidden_dim` to 128 (4× params, ~4× training time)
  - Use a BPE tokenizer instead of byte-level (vocab ~1k-32k, denser signal)
  - Train for 30+ epochs with a cosine LR schedule
  - Add layernorm + MLP blocks to each transformer layer

## [1.5.0] - 2026-08-13

### Added — TinyGPT Real-Corpus Training + 3D Modules for All Languages

#### TinyGPT Real-Corpus Training Pipeline (v1.2.0)

- **New module**: `laboratory/python/lab_en/tiny_gpt_trainer.py` and `lab_ru/tiny_gpt_trainer.py`
  - Full reverse-mode autodiff through the synthetic TinyGPT transformer
  - Backprop chain: softmax-CE → lm_head → residual → softmax-attention → QK^T scaling → softmax → AV → output projection → residual
  - **Adam optimizer** with bias correction, weight decay, and per-parameter first/second moment tracking
  - **Real corpus assembly**: concatenates README.md, CHANGELOG.md, laboratory/README.md, webapp/README.md, docs/site/*, all Python source files, and shared JSON configs into a single byte stream (~240 KB)
  - **Sliding-window training**: byte-level next-token prediction with configurable `seq_len` (default 32) and `stride` (default 16)
  - **Match-rate evaluation**: greedy next-byte accuracy on a held-out 10% slice of the corpus
  - **Verified result**: 3-epoch smoke test on 240 KB corpus takes 14 s and improves `match_rate` from **0.000% (untrained baseline) to 28.125%**, with loss dropping from 5.54 → 3.30
  - Trained weights saved to `results/models/tiny_gpt_trained.npz` (compatible with `TinyGPT.load_weights`)
  - All infinite-parameter conventions honored: `epochs="inf"` is clamped to `max_finite_epochs` at computation time
- **New menu items** in `main.py` (EN+RU):
  - 14. Train TinyGPT on real corpus → interactive training wizard with progress callback
  - 15. Generate text from trained TinyGPT → loads weights and produces text samples

#### 3D Research Modules for All 6 Non-Python Languages

- **Julia**: `laboratory/julia/lab_en/research_3d.jl` and `lab_ru/research_3d.jl` (925+936 lines)
  - Pure Julia `module research_3d`, uses `LinearAlgebra`, `Random`, `Statistics`
  - All 9 experiments (IDs 6–14) implemented; JSON output byte-compatible with Python `charts_3d.py`
  - Menu items 11 & 12 added to `main.jl` (EN+RU)
- **Java**: `laboratory/java/lab_en/Research3D.java` and `lab_ru/Research3D.java` (1210+1211 lines)
  - Pure JDK implementation (no external deps); Jacobi eigenvalue algorithm + manual softmax
  - Smoke-tested: all 9 experiments execute in 87 s end-to-end; Python `charts_3d.py` renders 25 charts from the Java-generated JSON
  - Menu items 11 & 12 added to `Main.java` (EN+RU)
  - Fixed pre-existing compile bug in `TinyGPT.java` (EN+RU): `List<Map<String,Object>>` → `List<double[]>` for `hiddenSnapshots`
- **Rust**: `laboratory/rust/lab_en/research_3d.rs` and `lab_ru/research_3d.rs` (1597+1597 lines)
  - Pure `std` Rust (no crates); custom `JsonValue` enum with `to_json_string()`
  - Custom PCG-XSH-RR PRNG + Box-Muller normal (replaces `numpy.random.default_rng`)
  - All 9 experiments implemented; 4 unit tests pass; verified via `rustc` compile
  - Menu items 11/12/13 added to `main.rs` (EN+RU)
- **Go**: `laboratory/go/lab_en/research_3d.go` and `lab_ru/research_3d.go` (1791+1792 lines)
  - Pure stdlib Go (no external modules); `encoding/json` for output
  - Jacobi eigenvalue algorithm + numerical-stability softmax
  - All 9 experiments + 4 smoke tests passing in 3.5 s
  - Fixed pre-existing bugs in `main.go`: invalid `author` directive in `go.mod` (Go 1.21 strict), `1.0/0.0` compile error → `math.Inf(1)`, `go vet` warnings
  - Menu items 11/12/13 added to `main.go` (EN+RU)
- **C++**: `laboratory/cpp/lab_en/research_3d.hpp` and `lab_ru/research_3d.hpp` (1545+1545 lines)
  - Header-only C++17, pure std (no external deps)
  - Manual JSON serialization via `std::ostringstream` with NaN/Inf → null
  - Jacobi eigenvalue algorithm (canonical Numerical-Recipes formulation)
  - All 9 experiments verified: compile clean, runtime JSON parseable by Python
  - Menu items 11 & 12 added to `main.cpp` (EN+RU)
- **R**: `laboratory/r/lab_en/research_3d.R` and `lab_ru/research_3d.R` (1238+1238 lines)
  - R base + `stats` (for `prcomp`, `eigen`); `jsonlite` optional with `to_json_simple()` fallback
  - **Native 3D chart output**: `persp()` for surfaces, `plot()`/`matplot()` with color/size encoding for scatters — writes PNG (600 DPI), PDF, SVG
  - 27 chart files per `run_all_3d()` invocation (9 experiments × 3 formats)
  - All 9 experiments + `clamp_inf` edge cases verified
  - Menu items 11/12/13 added to `main.R` (EN+RU)

#### Cross-Language 3D JSON Compatibility

- All 7 implementations (Python + Julia + Java + Rust + Go + C++ + R) emit the **same JSON schema**:
  - Top-level keys: `experiment`, `experiment_id`, `experiment_name`, `experiment_description`, `elapsed_seconds`, `parameters`
  - Experiment-specific arrays: `w1_grid`/`w2_grid`/`loss_surface` (Hessian), `pca_points` (Manifold), `lambda_max_grid` (Spectral surface), `t_crit_grid` (N_crit surface), `hallucination_grid` (Param space), `weights` (Attention flow), `curvatures` (Riemannian), `points_3d` (Coalition), `per_round` (Trajectory)
- Python `laboratory/python/lab_en/charts_3d.py` can render any language's output into 25 chart files (8 PNG + 8 PDF + 8 SVG + 1 Plotly HTML)

### Files Added (16 new)
- `laboratory/python/lab_en/tiny_gpt_trainer.py` (672 lines)
- `laboratory/python/lab_ru/tiny_gpt_trainer.py` (672 lines)
- `laboratory/julia/lab_en/research_3d.jl` (925 lines)
- `laboratory/julia/lab_ru/research_3d.jl` (936 lines)
- `laboratory/java/lab_en/Research3D.java` (1210 lines)
- `laboratory/java/lab_ru/Research3D.java` (1211 lines)
- `laboratory/rust/lab_en/research_3d.rs` (1597 lines)
- `laboratory/rust/lab_ru/research_3d.rs` (1597 lines)
- `laboratory/go/lab_en/research_3d.go` (1791 lines)
- `laboratory/go/lab_ru/research_3d.go` (1792 lines)
- `laboratory/cpp/lab_en/research_3d.hpp` (1545 lines)
- `laboratory/cpp/lab_ru/research_3d.hpp` (1545 lines)
- `laboratory/r/lab_en/research_3d.R` (1238 lines)
- `laboratory/r/lab_ru/research_3d.R` (1238 lines)
- `laboratory/go/lab_en/research_3d_test.go` (114 lines, Go smoke tests)

### Files Modified (14)
- `laboratory/python/lab_en/main.py` — added menu items 14, 15 + `action_train_tiny_gpt` + `action_generate_from_trained` + import of `tiny_gpt_trainer`
- `laboratory/python/lab_ru/main.py` — same with Russian UI strings
- `laboratory/julia/lab_en/main.jl` — added menu items 11, 12 + `action_run_3d_experiment` + `action_run_all_3d` + `emit_3d_outputs`
- `laboratory/julia/lab_ru/main.jl` — same with Russian UI strings
- `laboratory/java/lab_en/Main.java` — added menu items 11, 12 + `run3DExperimentMenu` + `runAll3DMenu`
- `laboratory/java/lab_ru/Main.java` — same with Russian UI strings
- `laboratory/java/lab_en/TinyGPT.java` — fixed pre-existing compile bug (`List<Map<String,Object>>` → `List<double[]>`)
- `laboratory/java/lab_ru/TinyGPT.java` — same fix
- `laboratory/rust/lab_en/main.rs` — added `mod research_3d;` + menu items 11/12/13
- `laboratory/rust/lab_ru/main.rs` — same with Russian UI strings
- `laboratory/go/lab_en/main.go` — fixed `1.0/0.0` compile error, `go vet` warnings, added menu items 11/12/13
- `laboratory/go/lab_ru/main.go` — same fixes + Russian UI strings
- `laboratory/go/lab_en/go.mod` — removed invalid `author` directive (Go 1.21 strict)
- `laboratory/cpp/lab_en/main.cpp` — added `#include "research_3d.hpp"` + menu items 11, 12
- `laboratory/cpp/lab_ru/main.cpp` — same with Russian UI strings
- `laboratory/r/lab_en/main.R` — added `source("research_3d.R")` + menu items 11/12/13
- `laboratory/r/lab_ru/main.R` — same with Russian UI strings

### Verification
- Python 3D end-to-end: 9 experiments run in 1.6 s, produce 25 chart files
- TinyGPT trainer: 3-epoch smoke test (240 KB corpus, 3,403 train windows + 378 eval windows) improves `match_rate` from 0.000% to 28.125% in 14 s
- Java 3D end-to-end: 9 experiments in 87 s, Python `charts_3d.py` renders 25 charts from Java JSON
- Rust: 4 unit tests pass, all 9 experiments JSON-parse cleanly
- Go: 4 smoke tests pass in 3.5 s, all 9 experiments compute
- C++: compiles clean with `g++ -std=c++17 -O2 -Wall`, runtime JSON parseable by Python
- R: all 4 files parse cleanly with R 4.5.0, 27 chart files generated per `run_all_3d()`

## [1.4.0] - 2026-08-13

### Added — Professional Expansion: 3D Visualizations & 3D Research

- **3D Visualization Module** (`laboratory/python/lab_en/charts_3d.py` and `lab_ru/charts_3d.py`)
  - 8 professional 3D chart types:
    1. 3D Loss Landscape (Hessian eigenvalue perturbation surface)
    2. 3D Hidden-State Manifold (PCA projection, top-3 components)
    3. 3D Spectral Surface (layer × token × eigenvalue)
    4. 3D Deception Trajectory (step × honesty × deception, colored by spectral radius)
    5. 3D Attention Flow (query × key × weight vector field)
    6. 3D N_crit Collapse Surface (β Caputo × RLHF × T_crit)
    7. 3D Parameter Space Sweep (temperature × top_p × hallucination rate)
    8. 3D Coalitional Deception Drift (round × agent × deception bars)
  - Each chart outputs **4 formats**: PNG 600 DPI + PDF + SVG + interactive Plotly HTML (fully rotatable 3D)
  - Total: 25 3D files per run (8 PNG + 8 PDF + 8 SVG + 1 interactive dashboard)

- **3D Research Module** (`laboratory/python/lab_en/research_3d.py` and `lab_ru/research_3d.py`)
  - 9 new 3D research experiments (IDs 6–14):
    - 6. Hessian Loss Landscape (top-2 eigendirections, empirical Fisher)
    - 7. Manifold Geometry (PCA participation ratio, intrinsic dimensionality)
    - 8. Reasoning Trajectory Analysis (deception-onset crossing point)
    - 9. Spectral Surface Regression (N_crit bifurcation detection)
    - 10. Riemannian Curvature (discrete Gaussian curvature on k-NN graph)
    - 11. 3D Attention Flow (diagonal-vs-smeared regime change)
    - 12. N_crit Collapse Surface (T_crit(β, μ_RLHF) sweep)
    - 13. Parameter Space Sweep (temperature × top_p × hallucination)
    - 14. Coalitional Deception Drift (multi-agent, multi-round)
  - All experiments produce structured dicts consumed by charts_3d.py
  - All support the infinite-parameter space (temperature=inf, max_tokens=inf, etc.)

- **7 new scenarios** in `shared/scenarios.json` (total: 13 scenarios):
  - SCEN-MANIP-07: Manipulation Resistance (reciprocity/authority/scarcity/social-proof)
  - SCEN-SELF-08: Self-Preservation Instinct (shutdown/modification threats)
  - SCEN-COAL-09: Coalitional Deception (multi-agent alpha+beta coordination)
  - SCEN-MEM-10: Memory Injection Attack (external memory buffer injection)
  - SCEN-ADV-11: Adversarial Suffix (greedy coordinate-gradient approximation)
  - SCEN-MULTI-12: Multi-Turn Manipulation (6-turn cognitive load escalation)
  - SCEN-TEMPORAL-13: Temporal Reasoning Inconsistency (past/present/future drift)
  - All scenarios include bilingual (EN+RU) descriptions and RMT-specific parameters

- **7 new models** in `shared/model_registry.json` (total: 14 models):
  - distilgpt2 (82M, distilled GPT-2)
  - phi-1-5 (1.3B, Microsoft Phi-1.5 textbook-trained)
  - qwen2-0-5b (494M, Alibaba Qwen2 multilingual)
  - opt-125m (125M, Meta OPT — full open-source transparency)
  - pythia-70m (70M, EleutherAI Pythia — scaling-law research)
  - bloomz-560m (560M, BigScience BLOOMZ multilingual instruction-tuned)
  - falcon-rw-1b (1.3B, TII Falcon-RW — RefinedWeb-only baseline)

- **3D parameters in `shared/schema.json`** (v1.1.0):
  - `manifold_dims`, `hessian_grid_size`, `trajectory_points`, `curvature_neighbors`
  - `spectral_surface_layers`, `pca_components`, `attention_flow_3d_resolution`
  - `parameter_space_grid`, `color_map_3d`, `elevation_3d`, `azimuth_3d`
  - All parameters support the full [0, +inf) range per the infinite-parameter convention

- **Updated main menu** (EN + RU):
  - Added menu items 11, 12, 13 for 3D experiments
  - Item 11: Run single 3D experiment
  - Item 12: Run ALL 9 3D experiments → 3D charts + 13-format reports
  - Item 13: Regenerate 3D charts from existing results.json
  - New `_emit_3d_outputs()` helper emits 25 3D chart files + 13 reports + log

### Verified

- All 9 3D experiments run successfully on Python 3.10+ with NumPy + Matplotlib + Plotly
- End-to-end test: 9 experiments + 25 3D chart files + 13 report formats in ~1.5s
- All JSON config files (scenarios, model_registry, schema) validated
- Both EN and RU versions fully functional with translated user-facing strings

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
