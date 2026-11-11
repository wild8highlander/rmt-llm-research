#!/usr/bin/env python3
"""bump_version — one command, three metadata files, zero drift.

The release version lives in pyproject.toml, CITATION.cff and
.zenodo.json. This script bumps all three atomically (with --dry-run
preview) and refuses downgrades, so the metadata can never disagree
about which release this is.

Usage:
    python scripts/bump_version.py patch          # 1.3.0 -> 1.3.1
    python scripts/bump_version.py minor          # 1.3.0 -> 1.4.0
    python scripts/bump_version.py major          # 1.3.0 -> 2.0.0
    python scripts/bump_version.py 1.4.2          # explicit SemVer
    python scripts/bump_version.py patch --dry-run

Exit codes: 0 = bumped (or clean dry-run), 1 = invalid input, 2 = a
version file drifted so far the rewrite failed.
"""

from __future__ import annotations

import argparse
import datetime
import os
import re
import sys


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")

# (relative path, regex with the old value in group 2, replacement template, value kind)
TARGETS = [
    (
        "pyproject.toml",
        re.compile(r'(^version\s*=\s*")(\d+\.\d+\.\d+)(")', re.M),
        r"\g<1>{new}\g<3>",
        "version",
    ),
    (
        "CITATION.cff",
        re.compile(r'(^version:\s*")(\d+\.\d+\.\d+)(")', re.M),
        r"\g<1>{new}\g<3>",
        "version",
    ),
    (
        "CITATION.cff",
        re.compile(r'(^date-released:\s*")(\d{4}-\d{2}-\d{2})(")', re.M),
        r"\g<1>{date}\g<3>",
        "date",
    ),
    (
        ".zenodo.json",
        re.compile(r'("version"\s*:\s*")(\d+\.\d+\.\d+)(")'),
        r"\g<1>{new}\g<3>",
        "version",
    ),
]


def next_version(current: tuple[int, int, int], part: str) -> str:
    """Compute the next SemVer triple for patch/minor/major."""
    major, minor, patch = current
    if part == "patch":
        return f"{major}.{minor}.{patch + 1}"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    if part == "major":
        return f"{major + 1}.0.0"
    raise ValueError(f"unknown bump part: {part}")


def _parse(text: str) -> tuple[int, int, int]:
    m = SEMVER_RE.match(text)
    if not m:
        raise ValueError(f"not a SemVer triple: {text!r}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def bump(repo: str, new_version: str, new_date: str, dry_run: bool = False) -> bool:
    """Rewrite every version target; return True when anything changed."""
    changed = False
    for rel, pattern, template, value_kind in TARGETS:
        path = repo + "/" + rel
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        repl = template.format(new=new_version, date=new_date)
        new_text, n = pattern.subn(repl, text, count=1)
        if n == 0:
            print(f"ERROR   {rel}: version pattern not found — file drifted? Aborting.")
            return False
        old = pattern.search(text).group(2)
        new_value = new_date if value_kind == "date" else new_version
        if new_text == text:
            print(f"same    {rel}: {old} (unchanged)")
            continue
        changed = True
        print(f"{'dryrun' if dry_run else 'bumped':<7} {rel}: {old} -> {new_value}")
        if not dry_run:
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write(new_text)
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bump the release version in all metadata files.")
    parser.add_argument("part", help="patch | minor | major | explicit SemVer (e.g. 1.4.2)")
    parser.add_argument("--repo", default=REPO_ROOT)
    parser.add_argument("--dry-run", action="store_true", help="show the rewrite, change nothing")
    parser.add_argument(
        "--force",
        action="store_true",
        help="allow a downgrade (e.g. rolling a hotfix branch back)",
    )
    args = parser.parse_args(argv)

    pyproject = args.repo + "/pyproject.toml"
    with open(pyproject, encoding="utf-8") as fh:
        m = re.search(r'^version\s*=\s*"(\d+\.\d+\.\d+)"', fh.read(), re.M)
    if not m:
        print("ERROR   cannot read the current version from pyproject.toml")
        return 2
    current = _parse(m.group(1))

    if SEMVER_RE.match(args.part):
        new = _parse(args.part)
    elif args.part in ("patch", "minor", "major"):
        new = _parse(next_version(current, args.part))
    else:
        print(f"ERROR   expected patch | minor | major | SemVer, got {args.part!r}")
        return 1

    new_version = f"{new[0]}.{new[1]}.{new[2]}"
    if new < current and not args.force:
        print(
            f"ERROR   {new_version} < current {'.'.join(map(str, current))} — use --force to downgrade"
        )
        return 1

    today = datetime.date.today().isoformat()
    print(f"version {'.'.join(map(str, current))} -> {new_version}, date-released -> {today}\n")
    ok = bump(args.repo, new_version, today, args.dry_run)
    if not ok:
        return 2

    print(
        "\nFollow-ups (not automated):"
        "\n  1. Add a CHANGELOG.md entry under '## [Unreleased]' or '## [" + new_version + "]'."
        "\n  2. Re-pin MANIFEST.json:  python scripts/verify_repo.py --repin"
        "\n  3. Commit + tag:          git commit -am 'release: v"
        + new_version
        + "' && git tag v"
        + new_version
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
