# GitHub Pages — enabling and operating the documentation site

The repository carries a full [MkDocs Material](https://squidfunk.github.io/mkdocs-material/)
site (`docs/`, `mkdocs.yml`) and a deployment workflow
(`.github/workflows/deploy-docs.yml`). This page explains how to turn it on,
what it builds, and how to keep it green.

## TL;DR — one-time enablement (2 minutes)

GitHub Pages is a per-repository setting. It is currently **off**; to turn it
on:

1. Open **Settings → Pages** in the repository
   (`https://github.com/wild8highlander/rmt-llm-research/settings/pages`).
2. Under *Build and deployment → Source*, select **GitHub Actions**
   (not *Deploy from a branch* — the site is built by the workflow, not pushed
   as HTML).
3. Save. Nothing else: the workflow already has the required permissions
   (`pages: write`, `id-token: write`) and the `github-pages` environment.

Then trigger the first build:

- **Actions → Deploy Documentation Site to GitHub Pages → Run workflow**
  (manual `workflow_dispatch`), or
- push any change under `docs/**`, `mkdocs.yml` or `src/rmt_llm/**` to `main`.

The site appears at:

```text
https://wild8highlander.github.io/rmt-llm-research/
```

## What the workflow does

| Step | What | Notes |
|------|------|-------|
| 1 | Checkout (full history) | `fetch-depth: 0` — the `git-revision-date-localized` plugin needs real commit dates |
| 2 | Python 3.12 + pip cache | `cache-dependency-path: pyproject.toml` |
| 3 | `pip install -e ".[docs]"` | mkdocs-material, mkdocstrings, minify, exclude plugins |
| 4 | `mkdocs build --strict` | strict mode: **any warning fails the build** |
| 5 | Upload artifact + deploy | `actions/deploy-pages` with the OIDC token |

Strict mode is the point: the site cannot drift out of sync with the tree.
The gates that keep it green:

- `scripts/check_links.py` — every relative link in every Markdown file
  resolves (run daily by `cli-doctor`);
- `markdownlint-cli2` — style gates (run on Markdown changes);
- the `exclude_docs` globs in `mkdocs.yml` keep GitHub-only `README.md`
  twins out of the site build, so their repo-relative links never leak in.

## Local preview

```bash
pip install -e ".[docs]"   # or: make docs-serve-mkdocs (auto-installs)
mkdocs serve               # hot reload on http://localhost:8000
```

Build exactly like CI does (strict, into `site-build/`):

```bash
mkdocs build --strict
```

## Site features

- **Search** with EN+RU tokenizers (`plugins.search.lang`).
- **API reference** rendered by mkdocstrings from `src/rmt_llm/` and the
  TinyGPT lab sources — the docs and the code cannot diverge.
- **Dark / light palette**, instant navigation, sticky tabs, content tabs,
  MathJax (`docs/javascripts/mathjax.js`) and Mermaid diagrams.
- **Custom 404 page** (`docs/404.md`) with the aurora favicon.
- **Open Graph / Twitter cards** on every page — favicon and the pinned
  `og-image.png` ship via `docs/overrides/main.html`, so link previews on
  social platforms look right.
- **sitemap.xml** is generated automatically (`site_url` is set in
  `mkdocs.yml`); submit it to any indexer.
- **Minified HTML** (`mkdocs-minify-plugin`) and DOCX/PDF assets excluded
  from the build (the `exclude` plugin globs).

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| 404 on the Pages URL right after enabling | the first build has not finished | wait for the *Deploy Documentation Site* run to go green |
| build fails with `WARNING` lines | strict mode turned a warning into an error | read the log: usually a nav page, an escaping link, or a missing asset; re-run `mkdocs build --strict` locally |
| deploy job fails with 403 on `deploy-pages` | Pages source is set to *Deploy from a branch* | Settings → Pages → Source: **GitHub Actions** |
| pages build is stale after a big refactor | path filters did not trigger | run the workflow manually (`workflow_dispatch`) |

## Reference

- Workflow: [`.github/workflows/deploy-docs.yml`](https://github.com/wild8highlander/rmt-llm-research/blob/main/.github/workflows/deploy-docs.yml)
- Site config: [`mkdocs.yml`](https://github.com/wild8highlander/rmt-llm-research/blob/main/mkdocs.yml)
- Docs extras: `[project.optional-dependencies].docs` in
  [`pyproject.toml`](https://github.com/wild8highlander/rmt-llm-research/blob/main/pyproject.toml)
- GitHub docs: [Configuring a publishing source for your site](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-site)
