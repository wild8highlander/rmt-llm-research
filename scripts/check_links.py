#!/usr/bin/env python3
"""Check markdown links and anchors across the repository; report breakage.

Used by the `cli-doctor` CI workflow and locally:

    python scripts/check_links.py [repo_root] [--no-anchors]

Validates `[text](relative/path)`, reference-style definitions and inline
images against the filesystem. Since v2 it also verifies `#anchor`
fragments against the headings of the target document (GitHub-style slugs).
Skips `.git`, build output and external http(s) links. URL-escapes like
`Model%20ZIP/` are decoded before checking. Exit code 1 when anything
is broken.
"""

import os
import re
import sys
from urllib.parse import unquote


DEFAULT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = DEFAULT_ROOT
CHECK_ANCHORS = True
for arg in sys.argv[1:]:
    if arg == "--no-anchors":
        CHECK_ANCHORS = False
    elif not arg.startswith("-"):
        ROOT = arg

MD = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
REF = re.compile(r"^\s*\[[^\]]+\]:\s+(\S+)\s*$", re.M)
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$", re.M)
NAMED_ANCHOR = re.compile(r'<a\s+name="([^"]+)"')
SKIP_DIRS = (".git", "node_modules", "__pycache__", "site-build", ".venv", "docs/site")

_slug_cache: dict[str, set[str]] = {}


def github_slug(title: str) -> str:
    """Convert a markdown heading to the GitHub anchor slug."""
    title = title.strip().strip("`")
    title = re.sub(r"[^\w\- ]", "", title.lower())
    return re.sub(r" ", "-", title)


def headings_of(path: str) -> set[str]:
    """All anchor slugs available in a markdown document (cached)."""
    if path in _slug_cache:
        return _slug_cache[path]
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        _slug_cache[path] = set()
        return set()
    slugs = {github_slug(title) for _, title in HEADING.findall(text)}
    slugs |= set(NAMED_ANCHOR.findall(text))
    _slug_cache[path] = slugs
    return slugs


broken = []  # (source file, raw target, reason)
checked = 0
anchors_checked = 0
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
    for fn in filenames:
        if not fn.endswith(".md"):
            continue
        path = os.path.join(dirpath, fn)
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        for t in MD.findall(text) + REF.findall(text):
            if t.startswith(("http://", "https://", "mailto:", "<")):
                continue
            if t.startswith("#"):
                target_path, fragment = path, t[1:]
            else:
                head, _, frag = t.partition("#")
                target_path = os.path.normpath(os.path.join(dirpath, unquote(head.split("?")[0])))
                fragment = frag if frag else None
            if not target_path:
                continue
            checked += 1
            if not os.path.exists(target_path):
                broken.append((os.path.relpath(path, ROOT), t, "target not found"))
                continue
            if fragment and CHECK_ANCHORS and target_path.endswith(".md"):
                # mkdocstrings API anchors (#rmt_llm.module.item) are generated
                # at docs-build time — they don't exist in the git tree and are
                # verified by `mkdocs build --strict` in deploy-docs.yml instead.
                if re.fullmatch(r"[A-Za-z_][\w.]*", fragment) and "." in fragment:
                    continue
                anchors_checked += 1
                if fragment not in headings_of(target_path):
                    broken.append((os.path.relpath(path, ROOT), t, "anchor not found"))

print(f"checked {checked} relative links ({anchors_checked} anchors) in {ROOT}")
if broken:
    for src, t, reason in sorted(broken):
        print(f"BROKEN  {src}  ->  {t}  [{reason}]")
    print(f"\n{len(broken)} broken links/anchors")
    sys.exit(1)
print("all relative links OK")
