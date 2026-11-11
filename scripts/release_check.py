#!/usr/bin/env python3
"""release_check — pre-flight gate for release metadata.

Artifacts are pinned by `verify_repo.py`; this gate checks the *metadata*
that a release must keep consistent:

1. the version agrees across pyproject.toml, CITATION.cff and .zenodo.json
   (pyproject is the source of truth — `--fix` syncs the other two),
2. CHANGELOG.md has a section for the current version,
3. CITATION.cff is citable: DOI identifier, date-released and license,
4. .zenodo.json is uploadable: title, license, creators with ORCID.

Usage:
    python scripts/release_check.py            # report, exit 1 on failures
    python scripts/release_check.py --fix      # sync drifted versions first

Exit codes: 0 = release metadata consistent, 1 = failures found.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SEMVER_RE = re.compile(r"\d+\.\d+\.\d+")


def _read(repo: str, rel: str) -> str:
    with open(os.path.join(repo, rel), encoding="utf-8") as fh:
        return fh.read()


def _versions_in(repo: str) -> dict[str, str | None]:
    """Read the release version from every file that carries one."""
    pyproject = _read(repo, "pyproject.toml")
    m = re.search(r'^version\s*=\s*"(\d+\.\d+\.\d+)"', pyproject, re.M)
    versions = {"pyproject.toml": m.group(1) if m else None}

    cff = _read(repo, "CITATION.cff")
    m = re.search(r'^version:\s*"(\d+\.\d+\.\d+)"', cff, re.M)
    versions["CITATION.cff"] = m.group(1) if m else None

    zenodo = json.loads(_read(repo, ".zenodo.json"))
    versions[".zenodo.json"] = str(zenodo.get("version")) if zenodo.get("version") else None
    return versions


def check_versions(repo: str) -> list[str]:
    """Compare every version against pyproject (the source of truth)."""
    versions = _versions_in(repo)
    truth = versions["pyproject.toml"]
    problems: list[str] = []
    if not truth:
        problems.append("pyproject.toml has no readable SemVer version")
        return problems
    for name, version in versions.items():
        if version is None:
            problems.append(f"{name}: no version field found")
        elif version != truth:
            problems.append(f"{name}: version {version} != pyproject {truth}")
    return problems


def fix_versions(repo: str) -> list[str]:
    """Sync CITATION.cff and .zenodo.json versions to pyproject; return changes."""
    versions = _versions_in(repo)
    truth = versions["pyproject.toml"]
    if not truth:
        return []
    changed: list[str] = []
    for rel, pattern in [
        ("CITATION.cff", re.compile(r'(^version:\s*")(\d+\.\d+\.\d+)(")', re.M)),
        (".zenodo.json", re.compile(r'("version"\s*:\s*")(\d+\.\d+\.\d+)(")')),
    ]:
        if versions[rel] == truth:
            continue
        text = _read(repo, rel)
        new_text, n = pattern.subn(rf"\g<1>{truth}\g<3>", text, count=1)
        if n:
            with open(os.path.join(repo, rel), "w", encoding="utf-8", newline="") as fh:
                fh.write(new_text)
            changed.append(f"{rel}: version -> {truth}")
    return changed


def check_changelog(repo: str, version: str) -> list[str]:
    """CHANGELOG must mention the current version (Unreleased counts)."""
    if not os.path.exists(os.path.join(repo, "CHANGELOG.md")):
        return ["CHANGELOG.md is missing"]
    text = _read(repo, "CHANGELOG.md")
    headings = re.findall(r"^#+\s*\[?([Uu]nreleased|\d+\.\d+\.\d+)\]?", text, re.M)
    if version in headings or "Unreleased" in headings:
        return []
    return [
        f"CHANGELOG.md has no section for {version} (only {headings[:5]}...)"
        if headings
        else "CHANGELOG.md has no version sections at all"
    ]


def check_citation(repo: str) -> list[str]:
    """CITATION.cff must be citable: DOI, date, license."""
    problems: list[str] = []
    cff = _read(repo, "CITATION.cff")
    if "type: doi" not in cff:
        problems.append("CITATION.cff: no DOI identifier")
    if not re.search(r'^date-released:\s*"\d{4}-\d{2}-\d{2}"', cff, re.M):
        problems.append("CITATION.cff: no date-released")
    if "license:" not in cff:
        problems.append("CITATION.cff: no license field")
    return problems


def check_zenodo(repo: str) -> list[str]:
    """.zenodo.json must be uploadable: title, license, creators."""
    problems: list[str] = []
    data = json.loads(_read(repo, ".zenodo.json"))
    if not data.get("title"):
        problems.append(".zenodo.json: no title")
    if not data.get("license"):
        problems.append(".zenodo.json: no license")
    creators = data.get("creators") or []
    if not creators:
        problems.append(".zenodo.json: no creators")
    elif not any(c.get("orcid") for c in creators):
        problems.append(".zenodo.json: creators have no ORCID")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pre-flight gate for release metadata.")
    parser.add_argument("--repo", default=REPO_ROOT)
    parser.add_argument(
        "--fix", action="store_true", help="sync drifted versions to pyproject first"
    )
    args = parser.parse_args(argv)

    fixed: list[str] = []
    if args.fix:
        fixed = fix_versions(args.repo)
        for change in fixed:
            print(f"FIXED   {change}")

    problems: list[str] = []
    versions = _versions_in(args.repo)
    truth = versions["pyproject.toml"] or "?.?.?"
    problems += check_versions(args.repo)
    if versions["pyproject.toml"]:
        problems += check_changelog(args.repo, versions["pyproject.toml"])
    problems += check_citation(args.repo)
    problems += check_zenodo(args.repo)

    print(f"release-check: version {truth} — {len(problems)} problem(s)")
    for problem in problems:
        print(f"PROBLEM  {problem}")
    if not problems and not fixed:
        print("all release metadata consistent (pyproject = CITATION.cff = .zenodo.json)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
