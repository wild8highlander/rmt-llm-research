# Security

This page mirrors [`SECURITY.md`](https://github.com/wild8highlander/rmt-llm-research/blob/main/SECURITY.md)
for the docs site. The canonical version is at the repo root.

---

## Reporting a vulnerability

**Do NOT open a public issue for security vulnerabilities.**

Instead, please report vulnerabilities privately:

1. Go to [github.com/wild8highlander/rmt-llm-research/security/advisories/new](https://github.com/wild8highlander/rmt-llm-research/security/advisories/new)
2. Click "Report a vulnerability"
3. Fill in the template with:
   - Description of the vulnerability
   - Steps to reproduce
   - Affected versions
   - Potential impact
   - Suggested fix (if any)

### Response time

| Severity | First response | Fix target |
|----------|----------------|------------|
| Critical (RCE, data leak) | < 24 hours | < 72 hours |
| High (privilege escalation) | < 48 hours | < 7 days |
| Medium (DoS, info leak) | < 7 days | < 30 days |
| Low (theoretical) | < 30 days | Next release |

---

## Threat model

### What we protect

1. **User-supplied corpora** — the BPE tokenizer processes untrusted text
2. **User-supplied model weights** — the trainer loads `.npz` files
3. **CI secrets** — `CODECOV_TOKEN`, `ZENODO_TOKEN`

### What we don't protect

1. **The model output** — TinyGPT is a 2.5M-param toy model; don't use it
   for security-critical applications
2. **The webapp** — the React dashboard is for local research use only,
   not exposed to the internet
3. **Trained weights** — they're in the repo, anyone can read them

### Security boundaries

- **Untrusted model weights** — `model_downloader.py` only pulls from a
  hard-coded registry. The trainer never deserializes user-supplied `.npz`
  files. Loading external weights requires `--trust-weights` flag and
  prints a warning.
- **Untrusted corpora** — BPE tokenizer operates on raw bytes; no `eval`,
  no `pickle`, no YAML `unsafe_load`.
- **CI secrets** — only `CODECOV_TOKEN` and `ZENODO_TOKEN` exist; both are
  scoped read-only. No write tokens in workflows.
- **GitHub Actions permissions** — every workflow has `permissions: {
  contents: read }` by default; only `release.yml` escalates to `contents:
  write`, and only on tagged commits.

---

## Supported versions

| Version | Supported | Until |
|---------|-----------|-------|
| 1.6.x   | ✅ Yes    | Current |
| 1.5.x   | ⚠️ Critical fixes only | 2026-06-30 |
| < 1.5   | ❌ No     | End-of-life |

---

## Security measures in CI

| Measure | Tool | Workflow |
|---------|------|----------|
| Static analysis | CodeQL | `codeql.yml` |
| Dependency audit | pip-audit, Dependabot | `pre-commit.yml`, Dependabot |
| Secret scanning | gitleaks | `pre-commit.yml` |
| Supply-chain scoring | OpenSSF Scorecard | `scorecard.yml` |
| Container scan | Trivy | `docker.yml` |
| Adversary-in-the-loop | GitHub Security Advisories | manual |

---

## See also

- [`SECURITY.md`](https://github.com/wild8highlander/rmt-llm-research/blob/main/SECURITY.md) (canonical)
- [`CONTRIBUTING.md`](https://github.com/wild8highlander/rmt-llm-research/blob/main/CONTRIBUTING.md)
- [Architecture: Security Boundaries](../architecture/index.md#security-boundaries)
