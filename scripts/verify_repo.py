#!/usr/bin/env python3
"""Evidence gate for rmt-llm-research.

Verifies the machine-checkable claims of the README's
"Evidence & reproducibility" section:

  1. every path pinned in ``MANIFEST.json`` exists and matches its sha256;
  2. every visual slot is present in the promised format;
  3. every file respects its size budget (AVIF/WebP 400 KB, GIF 2 MB,
     WebM 3 MB, MP4 4 MB, ``assets/visuals`` 5 MB total);
  4. every poster promised by ``assets/prompts.yaml`` exists;
  5. every ``<img>``/``<video>`` in a README carries alt text;
  6. (unless ``--skip-cli``) the ``rmt-llm-verify`` self-check exits 0.

``--repin`` recomputes ``MANIFEST.json`` after assets are regenerated.

Usage::

    python scripts/verify_repo.py --all        # the full gate
    python scripts/verify_repo.py --repin      # re-pin hashes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(REPO, "MANIFEST.json")
VISUALS = os.path.join(REPO, "assets", "visuals")
ANIM = os.path.join(REPO, "assets", "anim")

PINNED_GLOBS = [
    "assets/prompts.yaml",
    "assets/visuals/*",
    "assets/anim/*",
    "results/reports/*.json",
    "CITATION.cff",
]

STATIC_SLOTS = {
    "hero": ("avif", "webp", "png"),
    "thread-1-spectral": ("avif", "webp"),
    "thread-2-hallucination": ("avif", "webp"),
    "thread-3-rlhf": ("avif", "webp"),
    "architecture": ("avif", "webp", "png"),
    "favicon": ("avif", "webp", "png"),
    "divider-thread-1": ("avif", "webp", "png"),
    "divider-thread-2": ("avif", "webp", "png"),
    "divider-thread-3": ("avif", "webp", "png"),
    "og-image": ("png",),
}
ANIM_SLOTS = ("hero", "thread-1", "thread-2", "thread-3", "dataflow")
POSTERS = (
    "hero-poster",
    "thread-1-poster",
    "thread-2-poster",
    "thread-3-poster",
    "dataflow-poster",
)

BUDGET_KB = {
    ".avif": 400,
    ".webp": 400,
    ".png": 2048,
    ".gif": 2048,
    ".webm": 3072,
    ".mp4": 4096,
}
VISUALS_DIR_BUDGET_MB = 5

TAG_RE = re.compile(r"<(img|video)\b[^>]*\s[^>]*>", re.IGNORECASE)
ALT_RE = re.compile(r"\balt\s*=\s*\"[^\"]*\"|\balt\s*=\s*'[^']*'", re.IGNORECASE)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def repin() -> None:
    import glob

    files: dict[str, dict[str, object]] = {}
    for pattern in PINNED_GLOBS:
        for path in sorted(glob.glob(os.path.join(REPO, pattern))):
            rel = os.path.relpath(path, REPO)
            files[rel] = {
                "sha256": sha256(path),
                "bytes": os.path.getsize(path),
            }
    manifest = {
        "schema": "rmt-llm-manifest/v1",
        "generated_by": "scripts/verify_repo.py --repin",
        "note": (
            "Pinned sha256 of the evidence layer: visual slots, "
            "animation loops, prompts, result reports, citation."
        ),
        "files": files,
    }
    with open(MANIFEST, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"MANIFEST.json re-pinned: {len(files)} files")


def check_manifest() -> list[str]:
    errors: list[str] = []
    with open(MANIFEST, encoding="utf-8") as fh:
        data = json.load(fh)
    for rel, meta in data["files"].items():
        path = os.path.join(REPO, rel)
        if not os.path.exists(path):
            errors.append(f"missing pinned file: {rel}")
        elif sha256(path) != meta["sha256"]:
            errors.append(f"sha256 mismatch: {rel}")
    return errors


def check_slots() -> list[str]:
    errors = []
    for slot, exts in STATIC_SLOTS.items():
        for ext in exts:
            p = os.path.join(VISUALS, f"{slot}-2x.{ext}")
            if slot == "og-image":
                p = os.path.join(VISUALS, f"{slot}.{ext}")
            if not os.path.exists(p):
                errors.append(f"missing static slot: {os.path.relpath(p, REPO)}")
    for slot in ANIM_SLOTS:
        for ext in ("webm", "mp4", "gif"):
            p = os.path.join(ANIM, f"{slot}-loop.{ext}")
            if not os.path.exists(p):
                errors.append(f"missing anim slot: {os.path.relpath(p, REPO)}")
    for poster in POSTERS:
        p = os.path.join(VISUALS, f"{poster}.avif")
        if not os.path.exists(p):
            errors.append(f"missing poster: {os.path.relpath(p, REPO)}")
    if not os.path.exists(os.path.join(ANIM, "cli-demo.gif")):
        errors.append("missing anim slot: assets/anim/cli-demo.gif")
    return errors


def check_budgets() -> list[str]:
    errors = []
    for base in (VISUALS, ANIM):
        for name in os.listdir(base):
            path = os.path.join(base, name)
            size_kb = os.path.getsize(path) / 1024
            ext = os.path.splitext(name)[1]
            limit = BUDGET_KB.get(ext)
            if limit and size_kb > limit:
                errors.append(
                    f"over budget: {os.path.relpath(path, REPO)} {size_kb:.0f} KB > {limit} KB"
                )
    total_mb = sum(os.path.getsize(os.path.join(VISUALS, n)) for n in os.listdir(VISUALS)) / (
        1024 * 1024
    )
    if total_mb > VISUALS_DIR_BUDGET_MB:
        errors.append(f"assets/visuals total {total_mb:.1f} MB > {VISUALS_DIR_BUDGET_MB} MB")
    return errors


def check_alt() -> list[str]:
    errors = []
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in (".git", "node_modules", "__pycache__")]
        for name in files:
            if not name.lower().endswith(".md"):
                continue
            path = os.path.join(root, name)
            with open(path, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
            for match in TAG_RE.finditer(text):
                tag = match.group(0)
                if not ALT_RE.search(tag):
                    rel = os.path.relpath(path, REPO)
                    line = text[: match.start()].count("\n") + 1
                    errors.append(f"{rel}:{line}: <{match.group(1)}> without alt text")
    return errors


def check_cli() -> list[str]:
    exe = shutil.which("rmt-llm-verify")
    if exe is None:
        return ["rmt-llm-verify not found on PATH — pip install -e . first"]
    try:
        proc = subprocess.run(  # noqa: S603 — entry point, resolved via which
            [exe, "verify"], capture_output=True, text=True, timeout=120, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [f"CLI smoke failed: {exc}"]
    if proc.returncode != 0:
        return [f"rmt-llm-verify verify exited {proc.returncode}"]
    if "5/5 checks passed" not in proc.stdout:
        return ["rmt-llm-verify verify did not report 5/5 checks passed"]
    return []


def main() -> int:
    ap = argparse.ArgumentParser(description="Evidence gate for the repo")
    ap.add_argument("--all", action="store_true", help="run every check")
    ap.add_argument("--repin", action="store_true", help="recompute MANIFEST.json hashes and exit")
    ap.add_argument("--budgets", action="store_true")
    ap.add_argument("--alt", action="store_true")
    ap.add_argument("--skip-cli", action="store_true")
    args = ap.parse_args()

    if args.repin:
        repin()
        return 0

    groups: list[tuple[str, list[str]]] = []
    if args.all or not (args.budgets or args.alt):
        groups.append(("manifest", check_manifest()))
        groups.append(("slots", check_slots()))
        groups.append(("budgets", check_budgets()))
        groups.append(("alt text", check_alt()))
        if not args.skip_cli:
            groups.append(("cli smoke", check_cli()))
    else:
        if args.budgets:
            groups.append(("budgets", check_budgets()))
        if args.alt:
            groups.append(("alt text", check_alt()))

    failed = 0
    for name, errors in groups:
        status = "OK" if not errors else "FAIL"
        print(f"  [{status}] {name}")
        for err in errors:
            print(f"      - {err}")
        failed += len(errors)
    total = len(groups)
    if failed:
        print(f"verify-repo: {failed} problem(s) in {total} group(s)")
        return 1
    print(f"MANIFEST.json OK — evidence gate passed ({total} groups)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
