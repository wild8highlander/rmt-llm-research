#!/usr/bin/env python3
"""Validate every markdown file in the repository: relative links, image and
<source> src paths, and blank-line hygiene.

Runs in CI (see .github/workflows/links.yml) and locally:
    python scripts/validate_md_links.py
Exits non-zero when any relative target does not exist.
"""

import re
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]

LINK = re.compile(r"\[[^\]]*\]\(([^)#\s]+)\)")
IMG = re.compile(r"<img[^>]+src=\"([^\"#]+)\"")
SOURCE = re.compile(r"<source[^>]+src=\"([^\"#]+)\"")

SKIP_DIRS = ("docs/site", "node_modules", ".git", "laboratory/webapp")
SKIP_FILES = {"CHANGELOG.md"}

problems: list[str] = []
md_files: list[Path] = []

for md in sorted(REPO.rglob("*.md")):
    rel = md.relative_to(REPO)
    parts = rel.as_posix()
    if parts in SKIP_FILES:
        continue
    if any(parts.startswith(s) or f"/{s}/" in f"/{parts}" for s in SKIP_DIRS):
        continue
    if "/results/" in f"/{parts}":
        continue
    md_files.append(md)

for md in md_files:
    text = md.read_text(encoding="utf-8")
    rel = md.relative_to(REPO)
    if "\n\n\n" in text:
        problems.append(f"{rel}: 3+ consecutive blank lines")
    targets = set(LINK.findall(text)) | set(IMG.findall(text)) | set(SOURCE.findall(text))
    for t in sorted(targets):
        if re.match(r"^[a-z]+://", t) or t.startswith("mailto:"):
            continue
        if t.startswith("https://img.shields.io"):
            continue
        if not (md.parent / t).resolve().exists():
            problems.append(f"{rel}: MISSING target '{t}'")

if problems:
    print(f"{len(problems)} problem(s):")
    for p in problems:
        print(" -", p)
    sys.exit(1)

print(
    f"All markdown files valid: links, images and blank lines OK ({len(md_files)} files checked)."
)
