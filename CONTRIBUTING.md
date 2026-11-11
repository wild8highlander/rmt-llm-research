# Contributing to RMT-LLM Research

Thanks for your interest! This repository combines a NumPy verification core,
multi-language laboratories and a neural lab. Here is how to get productive
fast.

## Setup

```bash
git clone https://github.com/wild8highlander/rmt-llm-research.git
cd rmt-llm-research
python -m pip install -e .            # installs the rmt-llm-verify CLI
rmt-llm-verify verify                  # self-check: all modules must pass
python -m pytest src/rmt_llm/tests -q  # unit tests of the NumPy core
```

The TinyGPT laboratory ships its own suite
(`laboratory/python/lab_en/tests/`, 101 tests); see its README for details.

## Ground rules

- The canonical math lives in `src/rmt_llm/`. Every other implementation
  (C++, Rust, Go, Java, Julia, R) is cross-verified against it — change the
  core first, then port.
- Keep dependencies minimal: the core requires only `numpy`.
- Run `pre-commit run --all-files` before committing (ruff, mypy,
  markdownlint, codespell, gitleaks and friends are configured in
  `.pre-commit-config.yaml`).
- Charts and reports land in `results/` / `laboratory/results/`; do not
  commit generated output into source directories.

## Pull requests

1. Fork / branch (`feat/...`, `fix/...`, `docs/...`).
2. Make the change, add or update a test when behaviour changes.
3. `rmt-llm-verify verify` must exit 0 — CI runs the same gate via the
   `cli-doctor` workflow.
4. Fill the checklist in the PR template.

## Repo Janitor

The `Repo Janitor` workflow is the repository's self-healing bot. On a
weekly schedule (Saturdays), on demand via *Actions → Repo Janitor → Run
workflow*, and as a self-heal whenever CI fails on `main`, it runs
`scripts/repo_janitor.py` to keep the repo tidy:

- rewrites internal markdown links whose target file was moved or renamed;
- repairs stale `directory:` entries in `.github/dependabot.yml`;
- repairs broken `nav:` paths in `mkdocs.yml`;
- updates external links that moved permanently (HTTP 301/308) and reports
  the dead ones;
- re-applies `ruff format`, so a formatting drift can never hold the Lint
  job hostage again.

Everything fixable lands in one PR from the `janitor/auto-fix` branch
(labelled `janitor`); anything that needs human judgement is filed as an
issue with the same label. Knobs live in `.github/janitor.yml`. Run it
locally before opening a docs-heavy PR:

```bash
python scripts/repo_janitor.py --fix --online   # apply fixes + probe links
python scripts/repo_janitor.py                  # report only, offline
```

## Where things live

- Roadmap: `docs/ROADMAP.md`
- Architecture overview: `ARCHITECTURE.md`
- Security reports: see `SECURITY.md` (do not open public issues for
  vulnerabilities)
- Scientific protocol for the neural lab: `neural_lab/EXPERIMENT_PROTOCOL.md`
