#!/usr/bin/env python3
"""Check relative markdown links across the repository; report broken targets.

Used by the `cli-doctor` CI workflow and locally:

    python scripts/check_links.py [repo_root]

Validates `[text](relative/path)`, reference-style definitions and inline
images against the filesystem. Skips `.git`, external http(s) links and
pure anchors. URL-escapes like `Model%20ZIP/` are decoded before checking.
Exit code 1 when any link is broken.
"""
import os
import re
import sys
from urllib.parse import unquote


DEFAULT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROOT
MD = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
REF = re.compile(r"^\s*\[[^\]]+\]:\s+(\S+)\s*$", re.M)

broken = []
checked = 0
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d not in (".git", "node_modules",
                                                    "__pycache__")]
    for fn in filenames:
        if not fn.endswith(".md"):
            continue
        path = os.path.join(dirpath, fn)
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        for t in MD.findall(text) + REF.findall(text):
            if t.startswith(("http://", "https://", "mailto:", "#", "<")):
                continue
            t_clean = unquote(t.split("#")[0].split("?")[0])
            if not t_clean:
                continue
            checked += 1
            target = os.path.normpath(os.path.join(dirpath, t_clean))
            if not os.path.exists(target):
                broken.append((os.path.relpath(path, ROOT), t))

print(f"checked {checked} relative links in {ROOT}")
if broken:
    for src, t in sorted(broken):
        print(f"BROKEN  {src}  ->  {t}")
    print(f"\n{len(broken)} broken links")
    sys.exit(1)
print("all relative links OK")
