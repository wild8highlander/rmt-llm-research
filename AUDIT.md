# AUDIT.md — reference revision of wild8highlander/rmt-llm-research

Audit performed on `main` @ `6bbea30` ("RMT - LLM", includes the earlier
CLI/README/link cleanup), working branch: `chore/repo-elevation`.
Scope: map the repository, inventory every code file, list CLI entry
points, verify links, find missing READMEs and dependency manifests.
Scientific logic and computed results were **not** touched.

## 1. Repository map (depth ≤ 3, annotated)

```text
rmt-llm-research/
├── src/rmt_llm/            # 19 py — canonical NumPy core + CLI (12 modules, tests/)
├── laboratory/             # 133 MB — 8-language verification labs (EN + RU siblings)
│   ├── python/{lab_en,lab_ru,python/}   # 58 py; python/ = rmt_llm_viz (3D suite)
│   ├── {cpp,go,java,julia,r,rust}/{lab_en,lab_ru}
│   ├── julia/julia/{RMTLLMVerify,RMTLLMViz}  # 24 jl across the repo
│   ├── webapp/{src,server}  # Node server.js + Vite front-end
│   ├── scripts/             # gen_charts, render_branding, sync_ru_trainer, train_v2_chunk
│   ├── shared/              # cross-language fixtures
│   └── results/             # charts, charts_3d, reports, models (artifacts)
├── neural_lab/             # 556 MB — Next.js playground + research programs
│   ├── MNN-Pocket-0.3B/     # 556 MB total: kit Code/Docs/Graphic/Model ZIP + rcda*.zip history
│   ├── research/tinygpt_formula/  # TinyGPT v3 + scaling (7 py, Colab notebooks)
│   ├── science/             # predictive core: 38 py (H1 protocol, predictors, tests)
│   ├── stages/              # 44 py — stage1..5 pipeline + common/, results_schema/
│   ├── src, public, assets  # Next.js app (56 tsx, 12 ts), h1_weights/
├── formal_proof/           # rmt_fp/ (8 py formal library), experiments/ (9 py), monograph/
│   └── monograph/chapters_{EN,RU}/  # 14 md chapters of the written proof
├── docs/                   # MkDocs site: api, architecture, community, getting-started,
│   ├── monographs/          # M01–M06 (EN+RU descriptions; docx git-ignored, built on release)
│   ├── papers/              # merged analysis paper (PDF distributed with releases)
│   └── {tutorials,stylesheets,javascripts,site}
├── notebooks/              # 6 Jupyter: 01_quickstart … 05_tinygpt_v3 + rmt_llm_verification
├── results/                # top-level run artifacts (charts/, reports/)
├── assets/                 # branding (aurora), charts, covers, logos, scripts
├── scripts/                # repo-level maintenance: check_links.py, gen_banner_gif.py
└── .github/                # 12 workflows, dependabot (11 ecosystems), templates, CODEOWNERS
```

## 2. Code inventory by language

| Language | Files | Where |
|---|---|---|
| Python | 198 | src/rmt_llm (19), laboratory/python (58), neural_lab (89 across stages/science/research/MNN), formal_proof (17), scripts (2), assets (2) |
| Julia | 24 | laboratory/julia/{lab_en,lab_ru}/ (runners, scenarios, charts), julia/julia/{RMTLLMVerify,RMTLLMViz}/ (packages with src/test) |
| TypeScript/TSX | 68 | neural_lab/src (Next.js app + API), components |
| JavaScript | 7 | laboratory/webapp (server.js + Vite front-end), docs/javascripts/mathjax.js |
| Jupyter | 8 | notebooks/01–05 + verification, 2 Colab (tinygpt_formula) |
| HTML | 24 | docs/site (legacy demo), neural_lab public assets |
| TOML | 7 | pyproject.toml + 4 Project.toml + 2 packaging |

Dependency manifests: `pyproject.toml` (PEP 621, `numpy>=1.24` core, dev/lab/docs/bench/notebooks extras), `requirements-stages.txt`, `formal_proof/requirements.txt`, `laboratory/python/{lab_en,lab_ru}/requirements.txt`, 4 × Julia `Project.toml`, 2 × `package.json` (neural_lab Next.js, laboratory/webapp), `Dockerfile` + `docker-compose.yml` (Python + Julia + Java + Node).

## 3. CLI entry points

| Entry point | Status |
|---|---|
| `rmt-llm-verify` → `rmt_llm.__main__:main` (`pyproject.toml [project.scripts]`) | ✅ works; 7 subcommands (`info`, `verify`, `mp`, `bbp`, `tw`, `nhse`, `spacing`); self-check gate 5/5 |
| `python -m rmt_llm` | ✅ same entry via `__main__.py` |
| `make` — 69 targets | ✅ test/docs/docker/ci bundles |
| `python scripts/gen_banner_gif.py` | ✅ renders the README banner GIF from package functions |
| `python scripts/check_links.py` | ✅ 0 broken relative links (593 checked) |
| `python assets/scripts/validate_md_links.py` | ✅ stricter per-file validator (7 files in its allow-list) |
| `laboratory/scripts/{gen_charts,render_branding,sync_ru_trainer,train_v2_chunk}.py` | ✅ lab tooling |
| `neural_lab` npm (`dev/build/start/db:*`), `laboratory/webapp` npm (`dev/build/preview/server`) | present; not exercised in this audit (Next.js app, heavy deps) |
| Julia runners (`run_full_lab.jl`, `research_3d.jl`) | present; Julia not installed in audit sandbox — flagged, not run |

## 4. Documentation map

- 120 `README.md` files, 209 `.md` total. Every meaningful directory is
  covered; deliberately skipped: timestamped artifact dirs
  (`results/charts/<RUN_ID>`, `laboratory/results/charts/<RUN_ID>`) and
  Next.js source subfolders.
- `assets/charts/bbp_transition_live.gif` — animated README banner,
  rendered by the package itself (`scripts/gen_banner_gif.py`).
- Community files: CONTRIBUTING / CODE_OF_CONDUCT / CHANGELOG / SECURITY /
  SUPPORT / ARCHITECTURE / AUTHORS / CITATION.cff — present and cross-linked.

## 5. Link health

- `scripts/check_links.py`: **593 relative links, 0 broken** (after
  restoring the two `laboratory/results/{charts,charts_3d}/README.md`
  files lost in the squash merge — their folders are `.gitignore`d under
  the "keep selectively via `git add -f`" policy; restored exactly that
  way).
- `assets/scripts/validate_md_links.py`: passes (7 md in scope).
- External URLs (badges, DOI, ORCID, docs site) are **not** verified by
  these tools — the two link-checking workflows (`markdown-lint.yml`,
  `cli-doctor.yml`) cover relative paths only.

## 6. Hygiene findings

1. `neural_lab/MNN-Pocket-0.3B/` is 556 MB (rcda2/3/4.zip history + split
   model archive). Decision pending owner: keep in git vs. move to
   Releases/Zenodo.
2. `docs/site/` — legacy static demo predating MkDocs; documented as
   reference, candidate for removal at owner's call.
3. Deliberate nesting quirks kept for path stability:
   `laboratory/python/python/`, `laboratory/{java,java}/`,
   `laboratory/julia/julia/` — each level documented.
4. `.gitignore` mixes generated-artifact ignores with "keep selectively
   via `git add -f`" — READMEs inside ignored artifact dirs must be
   force-added (this bit the squash merge; now fixed).
5. README badge claims "tests-171+" while `src/rmt_llm/tests` currently
   runs 101 pytest tests (the rest live in the labs); flagged, badge text
   left to owner (PR block will verify the exact number).

## 7. Work plan ↔ master-prompt blocks

| Block | State after this audit |
|---|---|
| 1 Audit | ✅ this file |
| 2 Code deep-read → CODE.md | in progress: 3 full module samples (MP, BBP, TW) + map; remaining modules after owner's OK |
| 3 Root README landing | partially in place (badges, TOC, structure, CLI sections); headline-results table + reading paths proposed for next block |
| 4 Subfolder READMEs | 120 present; enrichment with file→role tables for key dirs after OK |
| 5 Links | ✅ 0 broken relative |
| 6 Dependabot | already configured (11 ecosystems, weekly/monthly); grouping/labels/auto-merge workflow proposed for next block |
| 7 CLI | ✅ restored + smoke-tested; `tests/test_cli.py` proposed for next block |
| 8 Hygiene | community files present; editorconfig/pre-commit present |
| 9 CI | 12 workflows; dedicated `docs-link-check.yml` proposed (currently covered by cli-doctor link job) |
