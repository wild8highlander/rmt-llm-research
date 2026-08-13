# 🛡️ Security Policy

## 🔒 Supported Versions

We publish security fixes for the latest major release only. Older versions
receive patches on a best-effort basis — please upgrade to the latest `main`
before reporting a vulnerability.

| Version | Supported          | Notes                                  |
|---------|--------------------|----------------------------------------|
| 1.6.x   | ✅ Active           | Current release line                   |
| 1.5.x   | ⚠️ Best-effort      | Critical fixes only                    |
| < 1.5   | ❌ Unsupported      | Please upgrade                         |

## 📣 Reporting a Vulnerability

**Please DO NOT open a public GitHub issue for security vulnerabilities.**

Instead, choose one of the following private channels:

1. **GitHub Security Advisory (preferred)**
   - Go to <https://github.com/wild8highlander/rmt-llm-research/security/advisories/new>
   - Click **"Report a vulnerability"**
   - Fill in the template — GitHub will notify the maintainer privately

2. **Email**
   - Send to **`aslan08_05@mail.ru`**
   - Subject: `[SECURITY] rmt-llm-research — <short description>`
   - PGP-encrypt if possible (key fingerprint published in `AUTHORS.md`)

### What to include in the report

- **Affected component** — module, file, or workflow (e.g. `src/rmt_llm/nhse.py`, `Dockerfile`, `.github/workflows/ci.yml`)
- **Commit / release** — `git rev-parse --short HEAD`
- **Reproduction** — minimal steps or PoC script
- **Impact** — what an attacker could achieve (RCE, data leak, supply-chain compromise)
- **Suggested fix** — if you have one in mind

### Response timeline

| Step                    | Target SLA     |
|-------------------------|----------------|
| Acknowledgment          | ≤ 72 hours     |
| Initial assessment      | ≤ 7 days       |
| Fix or mitigation       | ≤ 30 days      |
| Public disclosure (CVE) | After fix is released, coordinated with reporter |

We follow a **coordinated disclosure** model: we will not publish details
until a fix is available, and we will credit you in the release notes unless
you prefer to remain anonymous.

## 🎯 Threat Model & Scope

This project is a **research codebase**, not a production service. The
following are **in scope** for security reports:

| In scope                                              | Out of scope                                       |
|-------------------------------------------------------|----------------------------------------------------|
| Remote Code Execution in any module                   | "Theoretical" issues without a working PoC         |
| Path traversal in file-loading code (`model_downloader.py`) | Denial of service via large inputs            |
| Supply-chain risks (CI secret leaks, poisoned deps)   | Issues in dependencies (report to upstream)        |
| Prompt injection in TinyGPT generation paths          | Hallucinations as designed behavior (see papers)   |
| Pickle / npz deserialization of untrusted weights     | Self-DoS by running training on huge inputs        |
| SQL injection in `reports.py` SQLite export           | "Behavior differs from ChatGPT" complaints         |

If you are unsure whether your finding is in scope, **report it anyway** —
we'd rather see a false positive than miss a real issue.

## 🔐 CI/CD Security Posture

- **OpenSSF Scorecard** — automated supply-chain scoring (see `.github/workflows/scorecard.yml`).
- **CodeQL** — semantic code analysis for Python (see `.github/workflows/codeql.yml`).
- **Minimal permissions** — every workflow uses `permissions: { contents: read }` by default.
- **Pinned actions** — third-party actions are pinned to commit SHAs (via `dependabot.yml`).
- **No secrets in logs** — sensitive values are masked via `::add-mask::`.

## 📦 Dependency Security

- Dependabot monitors **9 ecosystems** (pip, github-actions, julia, npm, cargo, gradle, gomod, docker).
- Security advisories trigger **immediate** PRs, regardless of the weekly schedule.
- We run `pip-audit` and `npm audit` as part of the pre-commit hook chain.

## 🚨 Incident Response

In the event of a confirmed, exploited vulnerability:

1. Maintainer publishes a **GitHub Security Advisory** with a CVE request.
2. A **patch release** (e.g. `1.6.0 → 1.6.1`) is cut within 72 hours.
3. The **CHANGELOG** entry references the CVE without disclosing exploit details.
4. After 14 days (or once 80% of users have upgraded — whichever comes first), full technical details are published.

## 📜 Credit

We gratefully credit security researchers in:

- The release notes (`CHANGELOG.md`)
- The `AUTHORS.md` file (with consent)

If you would like to remain anonymous, just let us know in your report.

---

<sub>Maintainer: [Iskhak Hamzatovich Isaev](https://github.com/wild8highlander) · ORCID: [0009-0003-7299-0701](https://orcid.org/0009-0003-7299-0701)</sub>
