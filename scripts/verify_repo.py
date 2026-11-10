#!/usr/bin/env python3
"""Evidence gate for rmt-llm-research.

Verifies the machine-checkable claims of the README's
"Evidence & reproducibility" section:

  1. every path pinned in ``MANIFEST.json`` exists and matches its sha256;
  2. every visual slot is present in the promised format;
  3. every file respects its size budget (AVIF/WebP 400 KB, GIF 2 MB,
     WebM 3 MB, MP4 4 MB; ``assets/visuals`` ≤ 5 MB, ``assets/anim`` ≤ 12 MB
     per directory);
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
    "neural_lab/assets/visuals/*",
    "neural_lab/assets/anim/*",
    "neural_lab/MNN-Pocket-0.3B/assets/visuals/*",
    "neural_lab/MNN-Pocket-0.3B/assets/anim/*",
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

# Folder-level visual layers (see the `folders:` section of prompts.yaml).
# Each entry: (visuals dir, anim dir, static slots, anim slots, posters).
FOLDER_LAYERS = (
    (
        os.path.join(REPO, "neural_lab", "assets", "visuals"),
        os.path.join(REPO, "neural_lab", "assets", "anim"),
        ("hero", "divider"),
        ("hero",),
        ("neural-lab-hero-poster",),
    ),
    (
        os.path.join(REPO, "neural_lab", "MNN-Pocket-0.3B", "assets", "visuals"),
        os.path.join(REPO, "neural_lab", "MNN-Pocket-0.3B", "assets", "anim"),
        ("hero", "divider", "og-image"),
        ("hero",),
        ("mnn-hero-poster",),
    ),
)
FOLDER_ANIM_GIFS = (
    os.path.join(REPO, "neural_lab", "MNN-Pocket-0.3B", "assets", "anim", "cli-demo.gif"),
    os.path.join(REPO, "neural_lab", "assets", "anim", "lab-tour-loop.gif"),
    # animated counterparts of the RCDA figures, drawn from the committed JSONs
    os.path.join(REPO, "neural_lab", "MNN-Pocket-0.3B", "assets", "anim", "scans-loop.gif"),
    os.path.join(REPO, "neural_lab", "MNN-Pocket-0.3B", "assets", "anim", "dose-loop.gif"),
    os.path.join(REPO, "neural_lab", "MNN-Pocket-0.3B", "assets", "anim", "training-loop.gif"),
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
ANIM_DIR_BUDGET_MB = 12

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


def _check_folder_layers() -> list[str]:
    errors = []
    for vis_dir, anim_dir, static_slots, anims, posters in FOLDER_LAYERS:
        for slot in static_slots:
            exts = ("png",) if slot == "og-image" else ("avif", "webp", "png")
            for ext in exts:
                name = f"{slot}-2x.{ext}" if slot != "og-image" else f"{slot}.{ext}"
                p = os.path.join(vis_dir, name)
                if not os.path.exists(p):
                    errors.append(f"missing folder static: {os.path.relpath(p, REPO)}")
        for slot in anims:
            for ext in ("webm", "mp4", "gif"):
                p = os.path.join(anim_dir, f"{slot}-loop.{ext}")
                if not os.path.exists(p):
                    errors.append(f"missing folder anim: {os.path.relpath(p, REPO)}")
        for poster in posters:
            p = os.path.join(vis_dir, f"{poster}.avif")
            if not os.path.exists(p):
                errors.append(f"missing folder poster: {os.path.relpath(p, REPO)}")
    for gif in FOLDER_ANIM_GIFS:
        if not os.path.exists(gif):
            errors.append(f"missing folder anim: {os.path.relpath(gif, REPO)}")
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
    errors += _check_folder_layers()
    return errors


def check_budgets() -> list[str]:
    errors = []
    layer_dirs: list[str] = []
    for vis_dir, anim_dir, _st, _an, _po in FOLDER_LAYERS:
        layer_dirs += [vis_dir, anim_dir]
    for base in (VISUALS, ANIM, *layer_dirs):
        if not os.path.isdir(base):
            continue
        for name in os.listdir(base):
            path = os.path.join(base, name)
            size_kb = os.path.getsize(path) / 1024
            ext = os.path.splitext(name)[1]
            limit = BUDGET_KB.get(ext)
            if limit and size_kb > limit:
                errors.append(
                    f"over budget: {os.path.relpath(path, REPO)} {size_kb:.0f} KB > {limit} KB"
                )
    for base in (
        VISUALS,
        *(d for d in layer_dirs if d.endswith("visuals")),
    ):
        total_mb = sum(os.path.getsize(os.path.join(base, n)) for n in os.listdir(base)) / (
            1024 * 1024
        )
        if total_mb > VISUALS_DIR_BUDGET_MB:
            errors.append(
                f"{os.path.relpath(base, REPO)} total {total_mb:.1f} MB > {VISUALS_DIR_BUDGET_MB} MB"
            )
    anim_dirs = [ANIM, *(d for d in layer_dirs if d.endswith("anim"))]
    for base in anim_dirs:
        total_mb = sum(os.path.getsize(os.path.join(base, n)) for n in os.listdir(base)) / (
            1024 * 1024
        )
        if total_mb > ANIM_DIR_BUDGET_MB:
            errors.append(
                f"{os.path.relpath(base, REPO)} total {total_mb:.1f} MB > {ANIM_DIR_BUDGET_MB} MB"
            )
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
