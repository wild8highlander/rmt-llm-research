<p align="center">
  <img src="../assets/logos/root_scripts.banner.svg" width="760" alt="scripts — animated logo: terminal window with blinking cursor">
</p>

# 🛠️ scripts — Repository Utilities

<p align="center">
  <em>Repo-level tooling: chart generation and documentation validation</em><br><br>
  <a href="../README.md"><img src="https://img.shields.io/badge/folder-root%20scripts-22d3ee" alt="scripts"></a>
  <img src="https://img.shields.io/badge/python%203.10%2B-3776AB" alt="Python 3.10+">
</p>

Repository-level maintenance scripts. These are the tools the maintainer
runs to keep the visual and documentation layers of the repository
consistent — they are not part of the core library, not installed with
the package, and not required by any experiment. Think of this folder as
the repository's own little workbench: chart art for the assets gallery,
and a gate that keeps documentation links honest.

## What's inside

| Script | Purpose |
|---|---|
| `gen_charts.py` | **Black Aurora chart suite** — regenerates the 600 dpi publication-grade PNG charts in `assets/charts/` from the core library's routines (spectra, MP bounds, phase diagrams) |
| `validate_md_links.py` | walks all `*.md` files and verifies that every relative link and image target exists; prints a per-file report of missing targets |

## Usage

```bash
# regenerate all repository charts (deterministic seeds)
python3 scripts/gen_charts.py

# check documentation links (exit code reflects broken links)
python3 scripts/validate_md_links.py
```

`validate_md_links.py` is intentionally strict: it treats a renamed or
deleted file that is still referenced as an error, which is exactly the
failure mode that accumulates silently in long-lived documentation. Run
it after any file move; CI performs the equivalent check on push.

## How it fits

- The charts produced here populate [`assets/charts/`](../assets/charts/README.md)
  and are referenced by the documentation set.
- The link validator complements the [`markdown-lint`](../.github/workflows/markdown-lint.yml)
  workflow: the workflow enforces style, the validator enforces targets.
- Experiment-side tooling lives elsewhere — see
  [`neural_lab/science/experiments/`](../neural_lab/science/experiments/README.md)
  for the H1 runner and [`laboratory/`](../laboratory/README.md) for the
  per-language labs.

---

<p align="center">
  <img src="../assets/logos/root_scripts.avatar.svg" width="56" alt="scripts animated avatar"><br>
  <sub>
    <a href="../README.md">🏠 repository root</a> · <a href="../assets/README.md">🎨 branding</a> · <a href="../neural_lab/README.md">🌐 neural_lab →</a>
  </sub>
</p>
