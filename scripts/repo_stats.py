#!/usr/bin/env python3
"""Repository statistics — counts, report tables and README badge sync.

The honest-numbers engine behind the main README's badge wall. Computes
live counts (tests, monographs, papers, code, languages, assets) from the
working tree and can rewrite the README badges so the "Tests 171+" style
claims never drift from reality.

Usage:
    python scripts/repo_stats.py                   # human table
    python scripts/repo_stats.py --markdown        # markdown table (docs)
    python scripts/repo_stats.py --json            # machine-readable
    python scripts/repo_stats.py --check           # exit 1 if README badges are stale
    python scripts/repo_stats.py --sync-badges     # rewrite README badges to live counts

Exit codes: 0 = ok (or badges in sync), 1 = stale badges under --check.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LAB_LANGUAGES = ["cpp", "go", "java", "julia", "python", "r", "rust", "webapp"]

# README badge patterns: (label, regex with the live number in group 2).
# The whole suffix after the digits is consumed by the regex, so a patch
# applied twice (or a hand-edit) can never grow a second "+".
BADGE_PATTERNS = [
    (
        "tests",
        re.compile(r"(img\.shields\.io/badge/tests-)(\d+)(\+*)(-green)"),
    ),
    (
        "documents",
        re.compile(r"(img\.shields\.io/badge/Documents-)(\d+)(-green)"),
    ),
    (
        "papers",
        re.compile(r"(img\.shields\.io/badge/Papers-)(\d+)(-blue)"),
    ),
]


def _loc(path: str) -> int:
    """Total line count of a text file (binary-safe: skip undecodable)."""
    try:
        with open(path, encoding="utf-8", errors="ignore") as fh:
            return sum(1 for _ in fh)
    except OSError:
        return 0


def _count_pattern(root: str, exts: tuple[str, ...], pattern: str) -> tuple[int, int]:
    """Count occurrences of pattern across files with the given extensions."""
    hits, files = 0, 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames if d not in (".git", "__pycache__", ".mypy_cache", ".ruff_cache")
        ]
        for fn in filenames:
            if not fn.endswith(exts):
                continue
            path = os.path.join(dirpath, fn)
            try:
                with open(path, encoding="utf-8", errors="ignore") as fh:
                    n = len(re.findall(pattern, fh.read(), re.MULTILINE))
            except OSError:
                continue
            if n:
                hits += n
                files += 1
    return hits, files


def _count_monographs(docs_dir: str) -> int:
    """Monograph documents = M* directories x render languages (en, ru).

    The DOCX/PDF renderings ship via Zenodo, not the git tree — the git
    tree carries one Markdown source set per monograph.
    """
    mono_dir = os.path.join(docs_dir, "monographs")
    if not os.path.isdir(mono_dir):
        return 0
    threads = [
        d
        for d in os.listdir(mono_dir)
        if d.startswith("M") and os.path.isdir(os.path.join(mono_dir, d))
    ]
    languages = sum(os.path.isdir(os.path.join(mono_dir, lang)) for lang in ("en", "ru"))
    return len(threads) * (languages or 1)


def _dir_size_mb(root: str) -> float:
    total = 0
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, fn))
            except OSError:
                continue
    return total / (1024 * 1024)


def _count_by_ext(root: str, exts: tuple[str, ...]) -> int:
    n = 0
    for _dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__")]
        for fn in filenames:
            if fn.endswith(exts):
                n += 1
    return n


def compute_stats(repo: str = REPO_ROOT) -> dict:
    """Collect the live repository numbers used by reports and badges."""
    pkg = os.path.join(repo, "src", "rmt_llm")
    labs = os.path.join(repo, "laboratory")
    docs = os.path.join(repo, "docs")

    py_files = {
        "src": _count_by_ext(pkg, (".py",)),
        "laboratory_python": _count_by_ext(os.path.join(labs, "python"), (".py",)),
        "scripts": _count_by_ext(os.path.join(repo, "scripts"), (".py",)),
        "neural_lab": _count_by_ext(os.path.join(repo, "neural_lab"), (".py",)),
    }
    py_loc = {
        "src": sum(
            _loc(os.path.join(dp, fn))
            for dp, _, fns in os.walk(pkg)
            for fn in fns
            if fn.endswith(".py")
        ),
        "laboratory_python": sum(
            _loc(os.path.join(dp, fn))
            for dp, _, fns in os.walk(os.path.join(labs, "python"))
            for fn in fns
            if fn.endswith(".py")
        ),
        "scripts": sum(
            _loc(os.path.join(dp, fn))
            for dp, _, fns in os.walk(os.path.join(repo, "scripts"))
            for fn in fns
            if fn.endswith(".py")
        ),
    }

    src_tests, _ = _count_pattern(pkg, (".py",), r"^\s*def test_")
    lab_tests, _ = _count_pattern(os.path.join(labs, "python"), (".py",), r"^\s*def test_")

    languages = [lang for lang in LAB_LANGUAGES if os.path.isdir(os.path.join(labs, lang))]

    visuals = _count_by_ext(os.path.join(repo, "assets", "visuals"), (".avif", ".webp", ".png"))
    anims = _count_by_ext(os.path.join(repo, "assets", "anim"), (".gif", ".webm", ".mp4"))

    return {
        "python": {
            "files": py_files,
            "loc": py_loc,
            "loc_total": sum(py_loc.values()),
        },
        "tests": {"src": src_tests, "laboratory": lab_tests, "total": src_tests + lab_tests},
        "docs": {
            "markdown": _count_by_ext(docs, (".md",)),
            "monographs": _count_monographs(docs),
            "papers": _count_by_ext(os.path.join(docs, "papers"), (".md", ".pdf")),
            "root_markdown": _count_by_ext(repo, (".md",)),
        },
        "laboratories": {"languages": languages, "count": len(languages)},
        "scripts": {"python_files": py_files["scripts"], "loc": py_loc["scripts"]},
        "assets": {
            "visuals": visuals,
            "anims": anims,
            "size_mb": round(_dir_size_mb(os.path.join(repo, "assets")), 1),
        },
    }


def _fmt_table(stats: dict) -> str:
    py = stats["python"]
    lines = [
        "rmt-llm-research — repository statistics",
        "",
        f"  Python code        {py['loc_total']:>8} LOC  (src {py['loc']['src']} · lab {py['loc']['laboratory_python']} · scripts {py['loc']['scripts']})",
        f"  Python files       {sum(py['files'].values()):>8}      (src {py['files']['src']} · lab {py['files']['laboratory_python']} · scripts {py['files']['scripts']} · neural_lab {py['files']['neural_lab']})",
        f"  Tests              {stats['tests']['total']:>8}      (src {stats['tests']['src']} · laboratory {stats['tests']['laboratory']})",
        f"  Languages          {stats['laboratories']['count']:>8}      {', '.join(stats['laboratories']['languages'])}",
        f"  Markdown docs      {stats['docs']['markdown']:>8}      (+ {stats['docs']['root_markdown']} at repo root)",
        f"  Monographs         {stats['docs']['monographs']:>8}      (M-threads x EN/RU renderings)",
        f"  Papers             {stats['docs']['papers']:>8}      docs/papers/",
        f"  Visual assets      {stats['assets']['visuals']:>8}      statics",
        f"  Animation assets   {stats['assets']['anims']:>8}      loops ({stats['assets']['size_mb']} MB assets/)",
    ]
    return "\n".join(lines)


def _fmt_markdown(stats: dict) -> str:
    py = stats["python"]
    rows = [
        ("Python LOC", str(py["loc_total"]), "src + laboratory/python + scripts"),
        ("Python files", str(sum(py["files"].values())), "package, labs, tooling"),
        (
            "Tests",
            str(stats["tests"]["total"]),
            f"{stats['tests']['src']} core + {stats['tests']['laboratory']} lab",
        ),
        (
            "Lab languages",
            str(stats["laboratories"]["count"]),
            ", ".join(stats["laboratories"]["languages"]),
        ),
        ("Markdown documents", str(stats["docs"]["markdown"]), "docs/ tree"),
        ("Monographs", str(stats["docs"]["monographs"]), "M-threads x EN/RU renderings"),
        ("Papers", str(stats["docs"]["papers"]), "docs/papers/"),
        ("Visual assets", str(stats["assets"]["visuals"]), "statics in assets/visuals/"),
        ("Animation assets", str(stats["assets"]["anims"]), "loops in assets/anim/"),
    ]
    lines = ["| Metric | Value | Notes |", "|---|---:|---|"]
    lines += [f"| {m} | {v} | {note} |" for m, v, note in rows]
    return "\n".join(lines)


def sync_badges(stats: dict, repo: str = REPO_ROOT, apply: bool = True) -> list[str]:
    """Rewrite the README badge numbers to the live counts; return changes."""
    readme = os.path.join(repo, "README.md")
    with open(readme, encoding="utf-8") as fh:
        text = fh.read()
    live = {
        "tests": f"{stats['tests']['total']}+",
        "documents": str(stats["docs"]["monographs"]),
        "papers": str(stats["docs"]["papers"]),
    }
    changes: list[str] = []
    for name, pattern in BADGE_PATTERNS:
        # The live value carries its own "+" where the badge format needs
        # one, and the regex consumes any suffix (including a stray extra
        # "+" from a double-applied patch), so the result is idempotent.
        def repl(m: re.Match[str], name: str = name) -> str:
            return f"{m.group(1)}{live[name]}{m.group(m.lastindex)}"

        new_text, n = pattern.subn(repl, text)
        if n and new_text != text:
            old = pattern.search(text)
            changes.append(f"{name}: {old.group(2)}{old.group(old.lastindex)} -> {live[name]}")
            text = new_text
    if apply and changes:
        with open(readme, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
    return changes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Repository statistics and README badge sync.")
    parser.add_argument("--repo", default=REPO_ROOT, help="repository root")
    parser.add_argument("--markdown", action="store_true", help="emit a markdown table")
    parser.add_argument("--json", action="store_true", help="emit raw JSON")
    parser.add_argument(
        "--sync-badges", action="store_true", help="rewrite README badge numbers to live counts"
    )
    parser.add_argument(
        "--check", action="store_true", help="exit 1 when README badges are stale (no writes)"
    )
    args = parser.parse_args(argv)

    stats = compute_stats(args.repo)

    if args.json:
        print(json.dumps(stats, indent=2))
        return 0
    if args.markdown:
        print(_fmt_markdown(stats))
        return 0
    print(_fmt_table(stats))

    if args.sync_badges:
        changes = sync_badges(stats, args.repo, apply=True)
        if changes:
            print("\nREADME badges updated:")
            for change in changes:
                print(f"  • {change}")
        else:
            print("\nREADME badges already in sync.")
        return 0

    changes = sync_badges(stats, args.repo, apply=False)
    if args.check and changes:
        print("\nREADME badges are stale — rerun with --sync-badges:")
        for change in changes:
            print(f"  • {change}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
