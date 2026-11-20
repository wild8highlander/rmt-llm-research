#!/usr/bin/env python3
"""Fix broken relative markdown links by resolving renamed/moved targets.

The repair counterpart to ``check_links.py``: where the checker reports a
broken relative link, the fixer rewrites it to the closest existing target.

Resolution strategy, applied per broken link:

1. **Tail-name matching.** Every path segment is normalized (case-folded,
   whitespace/quotes/dashes/underscores dropped) so that ``RCDA-5`` matches
   ``R - CDA — 5``, ``Model-ZIP`` matches ``Model ZIP``, and ``Scripts``
   matches ``Script's``. The longest normalized tail of the broken target
   that exists in the tree wins.
2. **Proximity ranking.** Candidates that share the longest directory
   prefix with the source file (and the longest character prefix, and the
   shallowest depth) win, so ``README.md`` referenced from a
   ``R - CDA — 5`` file resolves to the sibling ``R - CDA — 5/README.md``
   and not to a same-named file in a different subtree.
3. **Anchor repair.** For ``file.md#fragment`` links whose file exists but
   the fragment does not, the fragment is matched (same normalization)
   against the target's heading slugs and rewritten to the exact slug.

Only relative links are touched; external http(s) links, ``mailto:`` and
``<...>`` autolinks are left alone.

Usage:
    python scripts/fix_links.py            # report + fix
    python scripts/fix_links.py --dry-run  # report only
    python scripts/check_links.py           # verify (exit 0 = healthy)

Exit codes: 0 = all broken links fixed (or none), 1 = nothing to run on.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Iterable
from urllib.parse import quote, unquote


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MD_LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)\s]+)\)")
MD_REF_RE = re.compile(r"^(\s*\[[^\]]+\]:\s+)(\S+)(\s*)$", re.M)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$", re.M)
SKIP_DIRS = (".git", "node_modules", "__pycache__", "site-build", ".venv", "docs/site")
# Characters dropped when comparing names: whitespace, quotes, hyphens of
# every flavor, underscores, and the emoji variation selector.
_DROP = re.compile(r"[\s'’`\-—–‐_‑\ufe0f]")


def norm(name: str) -> str:
    """Normalize a path segment for fuzzy name comparison."""
    return _DROP.sub("", name.lower())


def slugify(title: str) -> str:
    """GitHub-style heading slug (mirrors scripts/check_links.py)."""
    title = title.strip().strip("`")
    title = re.sub(r"[^\w\- ]", "", title.lower())
    return re.sub(r" ", "-", title)


def slugs_of(path: str) -> set[str]:
    """All heading slugs available in a markdown file."""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return set()
    return {slugify(t) for _, t in HEADING_RE.findall(text)}


class TreeIndex:
    """Indexed view of every file and directory in the repository."""

    def __init__(self, root: str) -> None:
        self.root = root
        entries: list[str] = []  # repo-relative posix paths
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
            rel = os.path.relpath(dirpath, root).replace(os.sep, "/")
            if rel != ".":
                entries.append(rel)
            for fn in sorted(filenames):
                entries.append(f"{rel}/{fn}" if rel != "." else fn)
        # normalized path suffix -> actual paths (tail matching at any depth)
        self._suffix: dict[str, list[str]] = {}
        for p in entries:
            segs = p.split("/")
            for j in range(len(segs)):
                key = "/".join(norm(s) for s in segs[j:])
                self._suffix.setdefault(key, []).append(p)

    def resolve_tail(self, norm_tail: list[str]) -> list[str]:
        """Actual paths whose normalized last segments equal ``norm_tail``."""
        return list(self._suffix.get("/".join(norm_tail), []))

    def exists(self, rel_path: str) -> bool:
        return os.path.exists(os.path.join(self.root, rel_path))


def score(candidate: str, source_rel: str) -> tuple[int, int, int]:
    """Proximity ranking: shared dirs, then shared chars, then shallower."""
    c_segs = candidate.split("/")
    s_segs = source_rel.split("/")
    shared = 0
    for a, b in zip(c_segs, s_segs):
        if a != b:
            break
        shared += 1
    nc, ns = norm(candidate), norm(source_rel)
    prefix = 0
    for a, b in zip(nc, ns):
        if a != b:
            break
        prefix += 1
    return (shared, prefix, -len(c_segs))


def encode(rel_from_dir: str, fragment: str | None) -> str:
    """Percent-encode a relative path (plus optional fragment)."""
    enc = quote(rel_from_dir, safe="/-._~()")
    return f"{enc}#{fragment}" if fragment else enc


def resolve(target: str, file_dir: str, rel: str, tree: TreeIndex) -> str | None:
    """Rewrite target to an existing path, or None if unresolvable."""
    head, _, frag = target.partition("#")
    decoded = unquote(head.split("?")[0])
    raw = os.path.normpath(os.path.join(file_dir, decoded)).replace(os.sep, "/")
    if tree.exists(raw):
        return None
    segs = raw.split("/")
    for k in range(len(segs), 0, -1):  # longest tail first
        cands = tree.resolve_tail([norm(s) for s in segs[-k:]])
        cands = [c for c in cands if tree.exists(c)]
        if not cands:
            continue
        best = max(cands, key=lambda c: score(c, rel))
        new_rel = os.path.relpath(best, file_dir or ".").replace(os.sep, "/")
        return encode(new_rel, frag) if frag else encode(new_rel, None)
    return None


def fix_anchor(target: str, file_dir: str, tree: TreeIndex) -> str | None:
    """Repair a fragment-only mismatch against the target's headings."""
    head, _, frag = target.partition("#")
    if not frag:
        return None
    decoded = unquote(head.split("?")[0])
    raw = os.path.normpath(os.path.join(file_dir, decoded)).replace(os.sep, "/")
    if not (raw.endswith(".md") and tree.exists(raw)):
        return None
    want = norm(unquote(frag))
    if not want:
        return None
    for slug in slugs_of(os.path.join(REPO_ROOT, raw)):
        if norm(slug) == want:
            new_head = encode(os.path.relpath(raw, file_dir or ".").replace(os.sep, "/"), None)
            candidate = f"{new_head}#{slug}"
            if candidate != target:
                return candidate
    return None


def fix_file(path: str, tree: TreeIndex, dry: bool) -> list[str]:
    """Fix broken relative links in one markdown file. Returns report lines."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    rel = os.path.relpath(path, REPO_ROOT).replace(os.sep, "/")
    file_dir = os.path.dirname(rel)
    notes: list[str] = []

    def new_target(target: str) -> str | None:
        """The replacement for one link target, or None if it is healthy."""
        if target.startswith(("http://", "https://", "mailto:", "<")):
            return None
        new = resolve(target, file_dir, rel, tree)
        if new is None and target.partition("#")[2]:
            new = fix_anchor(target, file_dir, tree)
        return None if (new is None or new == target) else new

    def sub_link(m: re.Match) -> str:
        target = m.group(2)
        if target.startswith("#"):  # same-file anchor
            for slug in slugs_of(path):
                if norm(slug) == norm(unquote(target[1:])) and slug != target[1:]:
                    notes.append(f"{rel}: {target} -> #{slug}")
                    return f"[{m.group(1)}](#{slug})"
            return m.group(0)
        new = new_target(target)
        if new is None:
            return m.group(0)
        notes.append(f"{rel}: {target} -> {new}")
        return f"[{m.group(1)}]({new})"

    def sub_ref(m: re.Match) -> str:
        new = new_target(m.group(2))
        if new is None:
            return m.group(0)
        notes.append(f"{rel}: {m.group(2)} -> {new}")
        return f"{m.group(1)}{new}{m.group(3)}"

    new_text = MD_REF_RE.sub(sub_ref, MD_LINK_RE.sub(sub_link, text))
    if not dry and new_text != text:
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(new_text)
    return notes


def main(argv: Iterable[str] | None = None) -> int:
    """Entry point: walk every .md file and repair broken relative links."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="report only, fix nothing")
    args = parser.parse_args(list(argv) if argv is not None else None)

    tree = TreeIndex(REPO_ROOT)
    all_notes: list[str] = []
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if fn.endswith(".md"):
                all_notes.extend(fix_file(os.path.join(dirpath, fn), tree, args.dry_run))

    for note in all_notes:
        print(f"FIXED  {note}")
    print(f"fix_links: {len(all_notes)} link(s) {'would be ' if args.dry_run else ''}fixed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
