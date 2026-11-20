# 🕰️ site — the legacy static demo (pre-MkDocs)

A frozen, dependency-free demo page from before the MkDocs Material site
existed. Kept for reference and for the raw-HTML experiment gallery it
still hosts; it is **not** part of the MkDocs build (the `exclude` plugin
skips `site/*`).

| Path | Role |
|------|------|
| [`index.html`](index.html) | the demo landing page (self-contained) |
| [`style.css`](style.css) | its stylesheet |
| [`demo/`](demo/) | the small interactive demos + assets |
| [`README.md`](README.md) | this file |

Prefer the live site for anything new:

```bash
make docs-serve-mkdocs    # http://localhost:8000 — hot reload
```

The current documentation lives in [docs/](../index.md) (MkDocs Material,
built by the `deploy-docs.yml` workflow).
