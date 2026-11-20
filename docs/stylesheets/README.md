# 🎨 stylesheets — Material theme overrides

CSS layered on top of MkDocs Material. Loaded via `mkdocs.yml` →
`extra_css`, so the aurora palette, tables and code blocks keep their
identity across light/dark modes.

| File | Role |
|------|------|
| [`extra.css`](extra.css) | the overrides: brand colors, typography tweaks, table and admonition polish |
| [`README.md`](README.md) | this file |

Guidelines:

- Only load-bearing overrides belong here — anything Material already does
  well should stay in `mkdocs.yml` config, not in CSS.
- The site is built with `mkdocs build --strict`; a missing file referenced
  here fails the build, so keep `mkdocs.yml` and this directory in sync.
- Related: [docs/javascripts/](../javascripts/README.md) (the JS twin).
