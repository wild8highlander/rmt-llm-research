#!/usr/bin/env python3
"""Environment doctor — what does this machine need to run the repo?

One command answers "why doesn't X run on my machine": Python version
against `requires-python`, the core dependency, the dev tooling, the
optional build tools (ffmpeg for animations, Julia/Go/Cargo/Gradle for
the cross-language labs) and the package itself.

Usage:
    python scripts/env_doctor.py            # human table
    python scripts/env_doctor.py --json     # machine-readable
    python scripts/env_doctor.py --core     # only hard requirements (CI-friendly)

Exit codes: 0 = all hard requirements satisfied, 1 = something required
is missing (optional tooling only lowers the exit code with --strict).
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import re
import shutil
import sys


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (name, kind, required, hint)
# kind: "python" (module import), "binary" (PATH lookup), "meta" (info only)
CHECKS: list[tuple[str, str, bool, str]] = [
    ("numpy", "python", True, "pip install numpy"),
    ("yaml", "python", True, "pip install pyyaml"),
    ("matplotlib", "python", False, "pip install matplotlib (renderers)"),
    ("PIL", "python", False, "pip install pillow (renderers)"),
    ("rmt_llm", "python", False, "pip install -e .  (the package itself)"),
    ("pytest", "binary", True, "pip install pytest"),
    ("ruff", "binary", False, "pip install ruff"),
    ("mypy", "binary", False, "pip install mypy"),
    ("codespell", "binary", False, "pip install codespell"),
    ("pre-commit", "binary", False, "pip install pre-commit"),
    ("git", "binary", True, "install git"),
    ("ffmpeg", "binary", False, "needed by gen_anims.py / gen_banner_gif.py"),
    ("julia", "binary", False, "needed by make test-julia"),
    ("go", "binary", False, "needed by make test-go"),
    ("cargo", "binary", False, "needed by make test-rust"),
    ("g++", "binary", False, "needed by make test-cpp"),
    ("node", "binary", False, "needed by the webapp dashboard"),
    ("docker", "binary", False, "needed by make docker"),
]


def _requires_python(repo: str) -> tuple[int, int]:
    """Parse requires-python from pyproject.toml into a comparable tuple."""
    with open(os.path.join(repo, "pyproject.toml"), encoding="utf-8") as fh:
        text = fh.read()
    m = re.search(r"requires-python\s*=\s*[\"']>=\s*(\d+)\.(\d+)", text)
    if not m:
        return (3, 10)
    return (int(m.group(1)), int(m.group(2)))


def _module_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        module = importlib.import_module(name)
        return str(getattr(module, "__version__", "unknown"))


def run_checks(repo: str = REPO_ROOT, core_only: bool = False) -> list[dict]:
    """Execute every check and return the result rows."""
    rows: list[dict] = []
    need_major, need_minor = _requires_python(repo)

    version = sys.version_info
    ok = (version.major, version.minor) >= (need_major, need_minor)
    rows.append(
        {
            "name": "python",
            "required": True,
            "ok": ok,
            "detail": f"{version.major}.{version.minor}.{version.micro} "
            f"(needs >= {need_major}.{need_minor})",
        }
    )

    for name, kind, required, hint in CHECKS:
        if core_only and not required:
            continue
        if kind == "binary":
            found = shutil.which(name)
            rows.append(
                {
                    "name": name,
                    "required": required,
                    "ok": bool(found),
                    "detail": os.path.basename(found) if found else f"not on PATH — {hint}",
                }
            )
        else:
            try:
                rows.append(
                    {
                        "name": name,
                        "required": required,
                        "ok": True,
                        "detail": _module_version(name),
                    }
                )
            except Exception:
                rows.append(
                    {"name": name, "required": required, "ok": False, "detail": f"missing — {hint}"}
                )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Environment doctor for the rmt-llm toolbelt.")
    parser.add_argument("--repo", default=REPO_ROOT, help="repository root")
    parser.add_argument("--core", action="store_true", help="only hard requirements")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--strict", action="store_true", help="optional failures also exit 1")
    args = parser.parse_args(argv)

    rows = run_checks(args.repo, args.core)
    failed_required = [r for r in rows if r["required"] and not r["ok"]]
    failed_optional = [r for r in rows if not r["required"] and not r["ok"]]

    if args.json:
        print(json.dumps({"checks": rows, "ok": not failed_required}, indent=2))
    else:
        mark = {True: "OK  ", False: "MISS"}
        for row in rows:
            tag = "required" if row["required"] else "optional"
            print(f"[{mark[bool(row['ok'])]}] {row['name']:<12} {tag:<8} {row['detail']}")
        print(
            f"\nenv-doctor: {len(rows) - len(failed_required) - len(failed_optional)}/{len(rows)} available, "
            f"{len(failed_required)} required missing, {len(failed_optional)} optional missing"
        )

    if failed_required or (args.strict and failed_optional):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
