# rmt-llm-research — tidy-ci delivery

Branch: `repo-tidy-ci` (one commit on top of `main` @ 863015c)
Patch:  `rmt-llm-research-tidy-ci.patch` (git am or git apply)

## What changed (108 files)

1. Broken links & paths — 0 problems left (validator: scripts/validate_md_links.py)
   - root README monograph links retargeted docs/{en,ru} -> monographs/M0X
   - docs hub, laboratory nested twins, neural_lab, papers, src depths fixed
   - assets/branding/covers/ (12 plates) added — were referenced but missing
2. CI — all six failing workflows fixed + pipeline extended
   - ci.yml: phantom lint dirs removed, Julia project paths fixed
   - deploy-docs.yml: mkdocs --strict green (exclude_docs, nav, anchors, links)
   - markdown-lint.yml + pre-commit: markdownlint 0 issues in 98 files
   - codeql-config.yml: phantom paths removed
   - Dockerfile: real nested julia COPY path
   - NEW workflows: links.yml, schema.yml, labels.yml, greetings.yml,
     dependency-review.yml
   - stale.yml MOVED to .github/workflows/ (was inert outside workflows/)
3. Debranding — "Aurora edition" wording removed everywhere, visuals kept
   - all markdown texts + alt attributes
   - 12 monograph DOCX (text, ribbons, footers, embedded cover plates)
   - papers/LLM_Analysis_Merged.pdf rebuilt
4. scripts/ — validate_md_links.py (CI link validator) and gen_charts.py
   (600 dpi chart generator, portable paths) are now part of the repo

## How to apply

Option A (patch):
    git checkout main && git pull
    git checkout -b repo-tidy-ci
    git am rmt-llm-research-tidy-ci.patch   # or: git apply the .patch
    git push -u origin repo-tidy-ci

Option B (files): copy the tree below over the repository root, then
    git add -A && git commit -m "fix: repository tidy-up"

NOTE: this archive intentionally does NOT contain file deletions; the only
move is .github/stale.yml -> .github/workflows/stale.yml (apply via patch,
or `git rm .github/stale.yml` manually if copying files).

Validation run before shipping:
- links 0/96 md files; reports schema 90/90; markdownlint 0/98;
- mkdocs --strict exit 0; ruff clean; pytest 454 + 353 passed;
- all 12 workflows valid against the GitHub workflow schema.
