# Contributing

We welcome contributions of all sizes — from typo fixes to new RMT modules.

The full contributor guide is at
[`CONTRIBUTING.md`](https://github.com/wild8highlander/rmt-llm-research/blob/main/CONTRIBUTING.md)
(400+ lines, 8 languages covered).

This page is a quick summary for the docs site.

---

## Quick start

```bash
git clone https://github.com/<your-fork>/rmt-llm-research.git
cd rmt-llm-research
make install-dev          # installs dev deps + pre-commit hooks
make check                # quick local validation
# ...make your changes...
make ci                   # full local CI simulation
# open a PR against main
```

---

## What to work on

| Issue label | What it means | Good for beginners? |
|-------------|---------------|---------------------|
| [`good first issue`](https://github.com/wild8highlander/rmt-llm-research/labels/good%20first%20issue) | Small, self-contained, well-described | ✅ Yes |
| [`help wanted`](https://github.com/wild8highlander/rmt-llm-research/labels/help%20wanted) | Bigger task we'd love help with | ⚠️ Maybe |
| [`documentation`](https://github.com/wild8highlander/rmt-llm-research/labels/documentation) | Docs improvements | ✅ Yes |
| [`bug`](https://github.com/wild8highlander/rmt-llm-research/labels/bug) | Something is broken | ⚠️ Maybe |
| [`enhancement`](https://github.com/wild8highlander/rmt-llm-research/labels/enhancement) | New feature | ⚠️ Maybe |

---

## Code style

- **Python:** enforced by `ruff` + `ruff-format` (replaces black + isort + flake8)
- **Types:** advisory `mypy` — type hints are nice-to-have, not must-have
- **Tests:** required for new functions; property-based tests preferred for math
- **Docstrings:** Google style, required for public functions

The full style guide is in
[CONTRIBUTING.md → Code style](https://github.com/wild8highlander/rmt-llm-research/blob/main/CONTRIBUTING.md#code-style).

---

## See also

- [CONTRIBUTING.md (full)](https://github.com/wild8highlander/rmt-llm-research/blob/main/CONTRIBUTING.md)
- [Code of Conduct](code-of-conduct.md)
- [Security](security.md)
- [Roadmap](roadmap.md)
