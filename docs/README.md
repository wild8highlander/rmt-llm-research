<p align="center">
  <a href="../README.md"><img src="../assets/branding/aurora-icon.svg" width="72" alt="RMT × LLM Research — Black Aurora"></a>
</p>

# 📚 docs — Documentation Hub

<p align="center">
  <em>Everything worth reading, from a 60-second quick start to the full ADR archive</em><br><br>
  <img src="../assets/branding/aurora-divider.svg" width="420" alt=""><br><br>
  <a href="../README.md"><img src="https://img.shields.io/badge/folder-docs-526CFE" alt="repository root"></a>
  <img src="https://img.shields.io/badge/mkdocs%20material-526CFE" alt="MkDocs Material"><img src="https://img.shields.io/badge/en%20/%20ru-yellow" alt="EN/RU">
</p>

This folder is the single source of documentation truth for the project.
It is rendered into the public documentation site with **MkDocs Material**
(see `mkdocs.yml` at the repository root), and every page is plain
Markdown, so it reads just as well straight from the repository. The
materials are organized by intent: getting started, tutorials, API
reference, architecture decisions, and community governance.

| Path | Content |
|---|---|
| [`getting-started/`](getting-started/) | quick start, installation, first run, FAQ |
| [`tutorials/`](tutorials/) | 4 step-by-step tutorials mirroring the notebooks |
| [`api/`](api/) | API reference for the core package, TinyGPT, BPE, trainer |
| [`architecture/`](architecture/) | principles, request flow, ADR archive |
| [`community/`](community/) | code of conduct, contributing, roadmap, security |
| [`en/`](en/) | research papers (English, `.docx`) |
| [`ru/`](ru/) | research papers (Russian, `.docx`) |
| [`ROADMAP.md`](ROADMAP.md) | the research roadmap — including the open questions |
| [`changelog.md`](changelog.md) | documentation changelog |

> `docs/site/` is **generated output** of the MkDocs build — do not edit
> it by hand; `docs/javascripts/` and `docs/stylesheets/` hold the site
> theme glue.

## Serve locally

```bash
make docs-serve-mkdocs     # http://localhost:8000
```

The online version lives at
[wild8highlander.github.io/rmt-llm-research](https://wild8highlander.github.io/rmt-llm-research/).

---

<p align="center">
  <sub>
    <a href="../README.md">← root</a> · <a href="README.md">top</a> · <a href="../README.md">↑ parent</a> · <a href="../README.md">🏠 repository root</a> · <a href="getting-started/README.md">start reading →</a>
  </sub>
</p>
