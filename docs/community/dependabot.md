# Automated maintenance — Dependabot, the Janitor, and the doctors

The repository is kept healthy by three automated layers. Together they
cover every dependency tree, every internal link, and every CI gate — so the
canonical tree you clone is always the reference state.

## Layer 1 — Dependabot: eleven update blocks, twelve manifests

[Dependabot](https://docs.github.com/en/code-security/dependabot) opens
pull requests when dependencies move. The configuration lives in
[`.github/dependabot.yml`](https://github.com/wild8highlander/rmt-llm-research/blob/main/.github/dependabot.yml)
and covers every package manifest the repository actually ships (the pip
block at `/` picks up both `pyproject.toml` and
`requirements-stages.txt`):

| # | Ecosystem | Directory | Manifest | Cadence |
|---|-----------|----------|----------|---------|
| 1 | pip (core) | `/` | `pyproject.toml` | weekly (Mon) |
| 2 | GitHub Actions | `/` | `.github/workflows/*.yml` | weekly (Mon) |
| 3 | Julia — verifier | `/laboratory/julia/julia/RMTLLMVerify` | `Project.toml` | monthly |
| 4 | Julia — visualizer | `/laboratory/julia/julia/RMTLLMViz` | `Project.toml` | monthly |
| 5 | npm — web dashboard | `/laboratory/webapp` | `package.json` | weekly (Mon) |
| 6 | Rust — lab EN | `/laboratory/rust/lab_en` | `Cargo.toml` | monthly |
| 7 | Rust — lab RU | `/laboratory/rust/lab_ru` | `Cargo.toml` | monthly |
| 8 | Gradle — Java visualizer | `/laboratory/java/java/rmt-llm-viz` | `build.gradle` | monthly |
| 9 | Go — lab EN | `/laboratory/go/lab_en` | `go.mod` | monthly |
| 10 | Go — lab RU | `/laboratory/go/lab_ru` | `go.mod` | monthly |
| 11 | Docker — research image | `/` | `Dockerfile` | weekly (Mon) |
| 12 | pip — stages pipeline | `/` | `requirements-stages.txt` (same block as #1) | weekly (Mon) |

Policy highlights:

- **Grouping** — dev-toolchain bumps (pytest, ruff, mypy, coverage) and
  runtime bumps arrive as one PR per group, not one per package; the same
  for the GitHub Actions pins, the Julia packages and the Go modules.
- **Labels** — every PR is labeled `dependencies` + the ecosystem tag, so
  filtering the [PR list](https://github.com/wild8highlander/rmt-llm-research/pulls?q=label%3Adependencies)
  shows the whole queue.
- **Security first** — security advisories ignore the schedule and open a PR
  immediately; `pip-audit` (pre-commit) and CodeQL cross-check the result.
- **Noise control** — `open-pull-requests-limit` caps each ecosystem
  (3–5 PRs), majors of pinned models are ignored, and everything lands
  on the reviewer's Monday morning (`timezone: Europe/Moscow`).
- **Commit prefixes** — `deps(python)`, `deps(ci)`, `deps(webapp)` … keep
  the changelog readable.

## Layer 2 — the Repo Janitor: repairs, not reports

Where Dependabot upgrades dependencies, the janitor
(`scripts/repo_janitor.py`, workflow `repo-janitor`) repairs *repository
structure*:

- rewrites internal Markdown links whose target was moved or renamed;
- repairs stale `directory:` entries in `.github/dependabot.yml`;
- repairs broken `nav:` paths in `mkdocs.yml`;
- optionally (`--online`) rewrites external links that moved permanently.

Its companion `scripts/fix_links.py` resolves the fuzzy cases — names that
differ by punctuation (`Model-Z` vs `Model — Z`, `RCDA-5` vs
`R - CDA — 5`, `Scripts` vs `Script's`), wrong `../` depth, emoji anchors —
and is idempotent: it only rewrites links that do not resolve.

Run them locally:

```bash
python scripts/check_links.py   # verify — exits 1 on any broken link
python scripts/fix_links.py     # repair the unresolvable ones
python scripts/repo_janitor.py --fix  # structural repair pass
make check-links                # the Makefile shortcut
```

## Layer 3 — the doctors: daily gates

- **cli-doctor** (daily) — smoke-tests the `rmt-llm-verify` CLI and the
  whole link graph; failures open a labeled issue automatically.
- **verify-repo** (on every Markdown/asset change) — the evidence gate:
  MANIFEST sha256 pinning, asset size budgets, alt text, CLI smoke.
- **markdown-lint + pre-commit** — style and hygiene gates.
- **stages / benchmarks / CodeQL / docker / scorecard** — the research and
  security lanes.

The invariant they protect: **the tree on `main` is the reference state** —
green README claims, green CI, one canonical layout.

## Configuration cheatsheet

| File | Role |
|------|------|
| `.github/dependabot.yml` | dependency PRs: ecosystems, grouping, schedules, labels |
| `.github/janitor.yml` | what the janitor may auto-fix and what it only reports |
| `.github/workflows/repo-janitor.yml` | when the janitor runs (weekly + on CI failure) |
| `scripts/repo_janitor.py` | the janitor itself |
| `scripts/fix_links.py` | fuzzy link repair (idempotent, verify-first) |
| `scripts/check_links.py` | the link gate used by cli-doctor |
| `scripts/verify_repo.py` | the MANIFEST / budgets / alt-text evidence gate |
