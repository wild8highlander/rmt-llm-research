# 📜 javascripts — site-level scripts

Scripts loaded by the built MkDocs Material site. Referenced from
`mkdocs.yml` → `extra_javascript` (after the MathJax CDN entry), so every
page that needs equations gets them.

| File | Role |
|------|------|
| [`mathjax.js`](mathjax.js) | configures MathJax v3: inline `$...$` and display `$$...$$` delimiters, safe chattrs, no-junk menu |
| [`README.md`](README.md) | this file |

Notes:

- The docs build (`mkdocs build --strict`) copies this directory into the
  site verbatim — keep it minimal, plain and dependency-free.
- The MathJax core itself comes from the CDN; only the configuration is
  local, so the site never breaks on a bad local vendored copy.
- Related: [docs/stylesheets/](../stylesheets/README.md) (the CSS twin) and
  the monograph pages that lean on the math renderer
  ([docs/monographs/](../monographs/README.md)).
