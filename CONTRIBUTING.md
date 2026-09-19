# 🤝 Contributing to RMT & LLM Research

First off — **thank you** for taking the time to contribute! 🎉

This document describes how to contribute to `rmt-llm-research`. It covers
setup, code style, testing, the PR workflow, and project-specific conventions
for the 8 supported languages. By participating, you agree to abide by the
[Code of Conduct](./CODE_OF_CONDUCT.md).

> **TL;DR:** fork → branch → commit → `pre-commit run --all-files` → push →
> open PR with the template filled in. We typically review within 1–3 days.

---

## 📑 Table of Contents

- [Ways to Contribute](#-ways-to-contribute)
- [Development Environment](#-development-environment)
- [Project Structure](#-project-structure)
- [Code Style & Linting](#-code-style--linting)
- [Testing](#-testing)
- [Commit Convention](#-commit-convention)
- [Pull Request Workflow](#-pull-request-workflow)
- [Language-Specific Guidelines](#-language-specific-guidelines)
- [Mathematical Conventions](#-mathematical-conventions)
- [Translation Workflow (EN ↔ RU)](#-translation-workflow-en--ru)
- [Releases & Versioning](#-releases--versioning)
- [Recognition](#-recognition)

---

## 🎯 Ways to Contribute

| Type              | Examples                                                                   | Effort     |
|-------------------|----------------------------------------------------------------------------|------------|
| 🐛 Bug fix        | Fix an off-by-one in `mp_bounds()`, patch a CI flake                       | Small      |
| 📚 Documentation  | Fix a typo, expand a section, add an example, improve docstrings           | Small      |
| 🧪 Tests          | Add a property test, increase coverage, add a cross-language regression    | Small–Med  |
| ✨ Feature         | New RMT module, new visualization, TinyGPT improvement                     | Medium     |
| 🌍 Translation    | Sync EN ↔ RU labs, translate a new doc, add a third language              | Medium     |
| 🏗️ Architecture   | Refactor a module, change the JSON schema, add a new lab language          | Large      |
| 📐 Math / proof   | Correct a formula, add a citation, propose a new theorem path             | Large      |

Not sure where to start? Look for issues labeled
[`good first issue`](https://github.com/wild8highlander/rmt-llm-research/labels/good%20first%20issue)
or
[`help wanted`](https://github.com/wild8highlander/rmt-llm-research/labels/help%20wanted).

---

## 🛠️ Development Environment

### Prerequisites

| Tool      | Min version | Required for                            |
|-----------|-------------|-----------------------------------------|
| Python    | 3.10        | `src/rmt_llm/`, `laboratory/python/`    |
| Julia     | 1.9         | `julia/`, `laboratory/julia/`           |
| Java JDK  | 21          | `java/`, `laboratory/java/`             |
| Rust      | 1.75        | `laboratory/rust/`                      |
| Go        | 1.21        | `laboratory/go/`                        |
| C++       | C++20       | `laboratory/cpp/`                       |
| R         | 4.3         | `laboratory/r/`                         |
| Node.js   | 20          | `laboratory/webapp/`                    |
| Docker    | 24          | Optional — reproducible env             |

You do **not** need all of these — only the languages you intend to work on.
Python is required for the core verification package.

### Setup

```bash
# 1. Clone your fork
git clone https://github.com/<your-username>/rmt-llm-research.git
cd rmt-llm-research

# 2. Add upstream remote
git remote add upstream https://github.com/wild8highlander/rmt-llm-research.git

# 3. Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 4. Install with dev extras
pip install --upgrade pip
pip install -e ".[dev]"

# 5. Install pre-commit hooks
pip install pre-commit
pre-commit install

# 6. Verify everything works
pytest -v
pre-commit run --all-files
```

### Optional: Docker

```bash
docker build -t rmt-llm-research .
docker run --rm -it -v "$PWD":/workspace rmt-llm-research bash
```

---

## 📁 Project Structure

Read [`ARCHITECTURE.md`](./ARCHITECTURE.md) for the full layout. The short
version:

- `src/rmt_llm/` — pure-Python RMT math, no I/O, no plotting
- `laboratory/` — 8 languages × 2 locales, interactive menus
- `python/rmt_llm_viz/` + `julia/RMTLLMViz/` + `java/rmt-llm-viz/` — 3D viz suites
- `docs/` — monographs (EN + RU), GitHub Pages site
- `.github/` — CI workflows, issue/PR templates

---

## 🎨 Code Style & Linting

### Python

We use **Ruff** for both linting and formatting. Configuration is in
`pyproject.toml` under `[tool.ruff]`.

```bash
# Check only — fast, fails CI on errors
ruff check src/ laboratory/python/

# Format in-place
ruff format src/ laboratory/python/

# Auto-fix what's safe to auto-fix
ruff check --fix src/ laboratory/python/
```

**Key rules we enforce:**

- 100-char line length
- 4-space indentation, double quotes for strings
- `import` ordering: stdlib → third-party → local, separated by blank lines
- No `print()` in `src/rmt_llm/` — use `logging` or return values
- All public functions have type hints (`def foo(x: float) -> float:`)
- All public functions have docstrings (Google style)

**Type checking** is advisory (does not fail CI), but please run `mypy src/`
before opening a PR:

```bash
mypy src/ --ignore-missing-imports
```

### Pre-commit

`.pre-commit-config.yaml` runs:

- `ruff check` + `ruff format`
- `end-of-file-fixer`, `trailing-whitespace`, `mixed-line-ending`
- `check-yaml`, `check-toml`, `check-json`, `check-merge-conflict`
- `check-added-large-files` (blocks files > 500 KB)
- `markdownlint-cli2`
- `python-safety` (advisory)

Install with `pre-commit install` (run once after cloning). Then it runs
automatically on every `git commit`. To run all hooks manually:

```bash
pre-commit run --all-files
```

---

## 🧪 Testing

### Python tests

```bash
# Run all tests
pytest

# Run a specific module's tests
pytest src/rmt_llm/tests/test_marchenko_pastur.py -v

# Run with coverage
pytest --cov=rmt_llm --cov-report=term-missing --cov-report=html

# Run only fast tests (skip slow ones)
pytest -m "not slow"

# Run a single test function
pytest src/rmt_llm/tests/test_marchenko_pastur.py::test_mp_bounds_basic -v
```

**Coverage threshold:** 70% (`pyproject.toml → [tool.coverage.report]`).
PRs that drop coverage below 70% will be flagged.

### Julia tests

```bash
cd julia/RMTLLMVerify
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

### Java tests

```bash
cd java/rmt-llm-viz
./gradlew verify        # Headless cross-implementation checks (14 tests)
./gradlew test          # JUnit tests (if any)
```

### Rust / Go / C++ tests

```bash
cd laboratory/rust/lab_en && cargo test
cd laboratory/go/lab_en  && go test ./...
cd laboratory/cpp/lab_en && make test
```

### Cross-implementation consistency

```bash
pytest src/rmt_llm/tests/test_cross.py -v
```

This compares Python's numerical output against hard-coded Julia reference
values. **If your PR changes a numerical result, update the reference
values** in `test_cross.py` and explain the change in the PR description.

---

## 📝 Commit Convention

We follow a simplified **Conventional Commits** style:

```text
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:** `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`,
`perf`, `build`, `ci`, `revert`

**Scopes:** `rmt`, `tiny-gpt`, `lab`, `viz`, `docs`, `ci`, `deps`,
`translation`, `security`

**Examples:**

```text
feat(rmt): add Dyson Brownian motion module with 3 unit tests
fix(tiny-gpt): clamp gradient norm before Adam step to prevent NaN
docs(readme): add "What's New in v1.6.0" section
ci: add CodeQL workflow for Python semantic analysis
deps(ci): bump actions/checkout from 4.1.6 to 4.1.7
translation(lab_ru): sync tiny_gpt_trainer.py with EN v2 architecture
```

**Rules:**

- Subject line ≤ 72 chars, lowercase first letter, no period at end
- Body wraps at 100 chars, explains **what + why** (not how — the diff shows how)
- Footer references issues: `Closes #123`, `Refs #456`
- Breaking changes: add `BREAKING CHANGE:` in the footer or `!` after the type (e.g. `feat(rmt)!: ...`)

---

## 🔄 Pull Request Workflow

1. **Fork** the repo and create a feature branch from `main`:

   ```bash
   git checkout -b feat/my-feature
   ```

2. **Make your changes.** Commit early and often — we'll squash on merge if needed.
3. **Run pre-commit locally:**

   ```bash
   pre-commit run --all-files
   ```

4. **Run the relevant tests** (see [Testing](#-testing) above).
5. **Update `CHANGELOG.md`** under `[Unreleased]` (or the latest version section).
6. **Push to your fork:**

   ```bash
   git push origin feat/my-feature
   ```

7. **Open a PR** against `main`. Fill in the PR template completely — the
   checklist is enforced by reviewers.
8. **Address review feedback** with new commits (do not force-push unless asked).
9. **Once approved**, a maintainer will squash-merge your PR.

### PR size guidelines

- **Ideal:** < 300 lines changed, single concern
- **Acceptable:** < 1000 lines, well-justified
- **Large:** > 1000 lines — please split into multiple PRs or pre-discuss in a Discussion

### Review SLA

- First response: ≤ 3 days
- Full review: ≤ 7 days
- Merge decision: ≤ 14 days

If your PR has been waiting longer than this, ping `@wild8highlander` in a
comment.

---

## 🌐 Language-Specific Guidelines

### Python (`src/rmt_llm/`, `laboratory/python/`)

- **NumPy only** — no PyTorch, no TensorFlow, no JAX in the verification package
- Use `np.ndarray` for arrays, not Python lists
- Vectorize — no `for` loops over arrays when a NumPy ufunc will do
- `float64` everywhere; cast to `float32` only at explicit boundaries
- Public functions return plain types (`float`, `tuple`, `np.ndarray`) — no custom classes in `src/rmt_llm/`
- Use `__all__` in every module to declare the public API

### Julia (`julia/`, `laboratory/julia/`)

- Follow [SciML style guide](https://github.com/SciML/SciMLStyle)
- 4-space indent, 92-char line limit
- Use `LinearAlgebra`, `Statistics`, `Random` from stdlib; avoid heavy deps
- All public functions in `RMTLLMVerify.jl` must match the Python signature order

### Java (`java/`, `laboratory/java/`)

- Follow [Google Java Style](https://google.github.io/styleguide/javaguide.html)
- 4-space indent, 120-char line limit
- One class per file, package-private unless explicitly public API
- No `System.out.println` in library code — use `java.util.logging`

### Rust (`laboratory/rust/`)

- Run `cargo fmt` before committing
- Run `cargo clippy -- -D warnings` — no warnings allowed
- Use `ndarray` for arrays; prefer `&[T]` slices in function signatures
- Document public functions with `///` (rustdoc)

### Go (`laboratory/go/`)

- Run `gofmt -w .` before committing
- Run `golangci-lint run` — no warnings allowed
- Use `gonum.org/v1/gonum/mat` for matrices
- Package-level docs go on the package declaration: `// Package laben ...`

### C++ (`laboratory/cpp/`)

- C++20 standard
- Use Eigen for linear algebra
- `clang-format` with the Google style (`.clang-format` in directory)
- No exceptions in numerical code — return `std::expected<T, Error>` or use status codes

### R (`laboratory/r/`)

- Follow [tidyverse style guide](https://style.tidyverse.org/)
- 2-space indent, 80-char line limit
- Use `matrix` for 2D, `array` for higher dims
- Document with `roxygen2` comments (`#'`)

### JavaScript / React (`laboratory/webapp/`)

- Use Prettier with default config + 100-char line
- Functional components only (no class components)
- Use `useReducer` + `useContext` for state, not Redux (we don't need it for this scale)

---

## 📐 Mathematical Conventions

- **Random matrix parameter:** `q = N/T` (rows/cols) — never `T/N`
- **Tracy-Widom:** F₂ (β=2) is the default; specify β explicitly otherwise
- **Caputo fractional derivative:** order `β ∈ (0, 1)` — never use Riemann-Liouville without explicit justification
- **Winding number:** `ν ∈ {0, 1}` (integer); skin strength `S ∈ ℝ⁺`
- **N_crit:** always quote the value `≈ 114.39` from the Caputo path; corrections are reported as `N_crit ± ΔN`

When adding a new formula:

1. **Cite the source** in the docstring: paper, equation number, page
2. **Add a numerical regression test** with hard-coded expected values
3. **Update `constants.py`** if the formula introduces a new constant
4. **Document units** — most of our quantities are dimensionless, but state it explicitly

---

## 🌍 Translation Workflow (EN ↔ RU)

The `_en` and `_ru` directories are kept in lock-step. To make changes:

1. **Edit the EN version first.**
2. **Run the sync script:**

   ```bash
   python scripts/sync_ru_trainer.py   # or equivalent for your module
   ```

3. **Manually review** the generated RU file — auto-translation is a starting point, not a finish line
4. **Verify both versions produce identical numerical output** — run both menus and diff the JSON

**Translation rules:**

- Translate docstrings, print statements, and user-facing strings
- Do **not** translate: variable names, function names, class names, file names
- Keep the same English-language comments for non-obvious numerical tricks (so reviewers can compare EN ↔ RU side-by-side)
- For mathematical prose, follow Russian academic style (e.g. "матрица ковариации" not "ковариационная матрица" when ambiguous)

---

## 🚀 Releases & Versioning

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR** (e.g. 1.0 → 2.0): breaking API changes
- **MINOR** (e.g. 1.5 → 1.6): new features, backward-compatible
- **PATCH** (e.g. 1.6.0 → 1.6.1): bug fixes only

**Release process (maintainers only):**

1. Update `version` in `pyproject.toml`
2. Add a new section to `CHANGELOG.md` with the version + date
3. Tag: `git tag -a v1.6.0 -m "Release v1.6.0"`
4. Push tag: `git push origin v1.6.0`
5. `.github/workflows/release.yml` creates the GitHub Release + uploads artifacts
6. `.github/workflows/zenodo.yml` archives the release to Zenodo and updates the DOI

---

## 🏆 Recognition

Contributors are recognized in three ways:

1. **`AUTHORS.md`** — alphabetical list with ORCID/affiliation
2. **`CHANGELOG.md`** — credit in the relevant version section (`Thanks @username for …`)
3. **`README.md` contributors table** — managed by
   [all-contributors](https://allcontributors.org/) bot. To add a contributor:

   ```bash
   # Install the CLI
   npm install -g all-contributors-cli

   # Add a contributor (use the right contribution type)
   all-contributors add <username> code,doc,test,translation
   ```

   Or comment on an issue: `@all-contributors please add @username for code`

**Contribution types** recognized by the bot:
`code`, `doc`, `test`, `translation`, `review`, `ideas`, `bug`, `design`,
`maintenance`, `infra`, `tool`, `security`, `content`, `data`, `mentoring`,
`projectManagement`, `fundingFinding`, `eventOrganizing`, `talk`, `tutorial`,
`video`, `audio`, `promotion`, `research`.

---

## ❓ Questions?

- **Quick question:** [GitHub Discussions](https://github.com/wild8highlander/rmt-llm-research/discussions)
- **Bug or feature:** open an issue with the right template
- **Sensitive topic:** see [`SECURITY.md`](./SECURITY.md) or `.github/SUPPORT.md`
- **Maintainer:** [@wild8highlander](https://github.com/wild8highlander) · ORCID [0009-0003-7299-0701](https://orcid.org/0009-0003-7299-0701)

---

<sub>By contributing, you agree that your contributions will be licensed
under the project's [Proprietary License](./LICENSE).</sub>
