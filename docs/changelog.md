# Changelog

This page mirrors [`CHANGELOG.md`](https://github.com/wild8highlander/rmt-llm-research/blob/main/CHANGELOG.md)
for the docs site. The canonical version is at the repo root.

All notable changes to this project are documented in
[`CHANGELOG.md`](https://github.com/wild8highlander/rmt-llm-research/blob/main/CHANGELOG.md).

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## Quick links

- [Latest version (v1.7.0)](https://github.com/wild8highlander/rmt-llm-research/blob/main/CHANGELOG.md#170---2026-02-13)
- [v1.6.0](https://github.com/wild8highlander/rmt-llm-research/blob/main/CHANGELOG.md#160---2026-02-10)
- [Full history](https://github.com/wild8highlander/rmt-llm-research/blob/main/CHANGELOG.md)

---

## v1.7.0 highlights

### Added (Sprints 4-7)

- **ARCHITECTURE.md** expanded with C4 component diagram, deployment diagram,
  ADR section, performance characteristics, expanded test architecture
- **Makefile** new targets: `docs-mkdocs`, `docs-serve-mkdocs`,
  `docs-build-mkdocs`, `bench`, `test-hypothesis`, `profile`, `security`
- **Pre-commit hooks** new: vulture (dead code), interrogate (docstring
  coverage), pyupgrade, bandit, check-jsonschema, nbstripout
- **EditorConfig** expanded: ini, cfg, sql, graphql, proto sections
- **README** new sections: Quick Start, Roadmap, FAQ (15 Q&A),
  comparison table, advanced badges (PePy, CodeFactor, Snyk, Hugging Face)
- **MkDocs Material** documentation site with mkdocstrings API reference
- **4 tutorial notebooks**: quickstart, training, generation, multilingual
- **Hypothesis property-based tests** (`tests/test_hypothesis.py`)
- **pytest-benchmark performance tests** (`tests/test_benchmark.py`)
- **benchmark.yml** workflow — nightly performance regression detection
- **Dockerfile** enhanced with HEALTHCHECK, OCI labels, multi-stage build
- **docker-compose.yml** new services: `docs` (mkdocs), `bench`

### Changed

- `pyproject.toml`: added `bench` and `all` optional-dependency groups
- CI now runs hypothesis + benchmark tests in `benchmark.yml`
- Documentation now builds with MkDocs Material (was: static HTML)

See the [full v1.7.0 changelog entry](https://github.com/wild8highlander/rmt-llm-research/blob/main/CHANGELOG.md#170---2026-02-13).
