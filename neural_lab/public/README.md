# 🌐 public — static assets of the neural-lab app

Files served verbatim by Next.js at the site root (no bundling, no
processing) — see `next.config.ts` in the app root.

| File | Role |
|------|------|
| [`robots.txt`](robots.txt) | crawler rules for the neural-lab preview |
| [`README.md`](README.md) | this file |

Use it only for files that must keep their exact bytes/URLs (favicons,
verification tokens, manifests). Everything else belongs under the app's
component tree (`src/`) so it gets hashed URLs and proper caching.
