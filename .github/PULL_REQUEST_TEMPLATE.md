<!--
  Thanks for opening a Pull Request! 👋
  Please fill in every section below. PRs that skip sections will be returned for revision.

  💡 Pro tip: link the issue this PR closes with "Closes #123" or "Resolves #123"
     so GitHub auto-closes the issue on merge.
-->

## 📌 Summary

<!-- 1–3 sentences: what does this PR do and why? -->

## 🔗 Related Issue

<!-- "Closes #123" / "Refs #456" / "Part of roadmap item R-3" -->

## 🧪 Changes

<!-- Tick everything that applies. Delete irrelevant lines. -->

- [ ] **Source code** — `src/rmt_llm/*` or `laboratory/**`
- [ ] **Tests** — added/updated pytest / Julia / Java / Rust tests
- [ ] **Documentation** — README, CHANGELOG, docs/, docstrings
- [ ] **CI/CD** — `.github/workflows/*`, `pre-commit`, `Dockerfile`
- [ ] **Build / packaging** — `pyproject.toml`, `Cargo.toml`, `go.mod`, `Project.toml`
- [ ] **Translation** — EN ↔ RU sync
- [ ] **Other** — explain below

### Change Details

<!-- For each file you touched, give a 1-line summary of what changed and why. -->

```
src/rmt_llm/marchenko_pastur.py  — fixed off-by-one in mp_bounds() when q > 1
tests/test_mp.py                 — added regression test for q > 1 case
```

## ✅ Checklist

<!-- Tick every box that applies. Required boxes are marked with ⚠️. -->

- ⚠️ [ ] My code follows the project's style (`ruff check`, `ruff format --check`)
- ⚠️ [ ] I ran `pre-commit run --all-files` and all hooks pass
- ⚠️ [ ] I added / updated tests for my changes
- ⚠️ [ ] `pytest -v` passes locally (or `julia --project=. -e 'using Pkg; Pkg.test()'` for Julia)
- ⚠️ [ ] I updated `CHANGELOG.md` under an `[Unreleased]` or versioned section
- [ ] I updated `README.md` and `docs/` if user-facing behavior changed
- [ ] I added type hints and ran `mypy src/`
- [ ] For RMT/math changes: I added a property test or numerical regression value
- [ ] For new language ports: I added cross-implementation consistency check
- [ ] I documented my design decisions in the PR description below

## 🔬 Mathematical / Algorithmic Changes (if applicable)

<!-- If your PR touches equations, formulas, or numerical constants, paste them here
     with a citation to the source (paper, textbook, or upstream reference). -->

```latex
% Example:
% \lambda_{\pm} = (1 \pm \sqrt{q})^2, \quad q = N/T
% Ref: Marchenko & Pastur (1967), eq. (1.1)
```

## 🖼️ Screenshots / Visual Diff (if applicable)

<!-- Drop PNG/GIF for UI / visualization / dashboard changes. -->

## 📝 Design Notes

<!-- Why did you choose this approach over alternatives? Any trade-offs?
     Any follow-up work this PR enables? Any technical debt it introduces? -->

## 🚨 Breaking Changes

- [ ] This PR introduces breaking changes
- [ ] I have updated the version in `pyproject.toml` per SemVer
- [ ] I have documented the migration path in `CHANGELOG.md`

<!-- If breaking: describe what users need to change. -->

---

<sub>By opening this PR, you agree to the
[Code of Conduct](https://github.com/wild8highlander/rmt-llm-research/blob/main/CODE_OF_CONDUCT.md)
and confirm that you have the right to license your contribution under the project's
[Proprietary License](https://github.com/wild8highlander/rmt-llm-research/blob/main/LICENSE).</sub>
