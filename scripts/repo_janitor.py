#!/usr/bin/env python3
"""Repo Janitor — find and auto-fix repository-health issues.

The self-healing companion to `check_links.py`: where the doctor only
reports, the janitor repairs. It

1. rewrites internal markdown links whose target file or directory was
   moved or renamed (resolved by basename + trailing-path scoring),
2. repairs stale `directory:` entries in `.github/dependabot.yml`,
3. repairs broken `nav:` paths in `mkdocs.yml`,
4. with `--online`, rewrites external http(s) links that moved permanently
   (301/308 redirect chains) and reports dead ones.

Usage:
    python scripts/repo_janitor.py                     # report only, offline
    python scripts/repo_janitor.py --fix               # apply fixes, offline
    python scripts/repo_janitor.py --fix --online --report janitor-report

Exit codes: 0 = clean or all issues fixed, 1 = unfixable issues remain
(suppressed with --no-fail), 2 = bad invocation.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import dataclasses
import json
import os
import posixpath
import re
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from urllib.parse import quote, unquote

import yaml


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FENCE_RE = re.compile(r"(```.*?```|~~~.*?~~~)", re.S)
MD_LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)\s]+)\)")
MD_REF_RE = re.compile(r"^(\s*\[[^\]]+\]:\s+)(\S+)(\s*)$", re.M)
TARGET_SPLIT_RE = re.compile(r"^([^#?]+)([#?].*)?$")

# Manifest files that prove a package-ecosystem directory is real.
MANIFESTS: dict[str, tuple[str, ...]] = {
    "pip": ("pyproject.toml", "setup.py", "setup.cfg"),
    "npm": ("package.json",),
    "cargo": ("Cargo.toml",),
    "gradle": ("build.gradle", "build.gradle.kts", "settings.gradle", "settings.gradle.kts"),
    "maven": ("pom.xml",),
    "gomod": ("go.mod",),
    "docker": ("Dockerfile",),
    "julia": ("Project.toml",),
}

DEFAULT_SKIP_DIRS = [
    ".git",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "docs/site",
    "laboratory/results",
    "node_modules",
    "results",
    "site-build",
]

EXTERNAL_DEFAULTS: dict[str, object] = {
    "timeout_seconds": 12,
    "max_workers": 8,
    # 403/405/429 mean "bot-blocked", not dead — treat them as alive.
    "alive_status_codes": [200, 206, 301, 302, 303, 307, 308, 403, 405, 429],
    "moved_status_codes": [301, 308],
    "dead_status_codes": [404, 410],
    "skip_url_patterns": [
        r"^https?://(localhost|127\.0\.0\.1)",
        r"^https://doi\.org/10\.5281/zenodo",
        r"^https://img\.shields\.io",
        r"^https://api\.securityscorecards\.dev",
        r"^https://codecov\.io",
        r"^https://(www\.)?linkedin\.com",
        r"^https://(www\.)?twitter\.com",
        r"^https://x\.com",
        r"^https://orcid\.org",
    ],
    "user_agent": "rmt-llm-repo-janitor/1.0 (repository link-health sweep)",
}


# ─── Data model ──────────────────────────────────────────────────────────────
@dataclasses.dataclass
class Fix:
    """A concrete rewrite the janitor applied (or proposes)."""

    kind: str
    file: str
    old: str
    new: str
    note: str = ""

    def line(self) -> str:
        suffix = f" ({self.note})" if self.note else ""
        return f"- **{self.kind}** `{self.file}`: `{self.old}` -> `{self.new}`{suffix}"


@dataclasses.dataclass
class Problem:
    """An issue the janitor detected but cannot safely fix itself."""

    kind: str
    file: str
    detail: str
    reason: str = ""

    def line(self) -> str:
        return f"- UNFIXABLE: **{self.kind}** `{self.file}`: {self.detail} ({self.reason})"


@dataclasses.dataclass
class Stats:
    internal_links: int = 0
    external_occurrences: int = 0
    external_unique: int = 0
    external_ok: int = 0
    external_moved: int = 0
    external_dead: int = 0
    nav_entries: int = 0
    dependabot_dirs: int = 0


# ─── Configuration ───────────────────────────────────────────────────────────
class _PermissiveLoader(yaml.SafeLoader):
    """SafeLoader that tolerates mkdocs `!!python/name:` tags (values become None)."""


def _unknown_tag(loader: yaml.Loader, suffix: str, node: yaml.Node) -> None:  # noqa: ARG001
    return None


_PermissiveLoader.add_multi_constructor("tag:yaml.org,2002:python", _unknown_tag)


def load_yaml(path: str) -> tuple[dict | None, str]:
    """Parse a YAML file permissively; return (data, error_message)."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = yaml.load(fh, Loader=_PermissiveLoader) or {}  # noqa: S506 - loader extends SafeLoader; python tags resolve to None
        return data, ""
    except (yaml.YAMLError, OSError) as exc:
        return None, str(exc)


def load_config(path: str) -> dict:
    """Load .github/janitor.yml over the built-in defaults."""
    cfg: dict = {"skip_dirs": list(DEFAULT_SKIP_DIRS), "external": copy.deepcopy(EXTERNAL_DEFAULTS)}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            user = yaml.safe_load(fh) or {}
        for key, value in user.items():
            if isinstance(value, dict) and isinstance(cfg.get(key), dict):
                cfg[key].update(value)
            else:
                cfg[key] = value
    return cfg


# ─── Repository index ────────────────────────────────────────────────────────
class RepoIndex:
    """Map of basename -> repo-relative paths, built once per run."""

    def __init__(self, root: str, skip_dirs: list[str]) -> None:
        self.root = root
        self.files: dict[str, list[str]] = {}
        self.dirs: dict[str, list[str]] = {}
        for dirpath, dirnames, filenames in os.walk(root):
            rel = os.path.relpath(dirpath, root).replace(os.sep, "/")
            rel = "" if rel == "." else rel
            keep: list[str] = []
            for d in dirnames:
                child = f"{rel}/{d}" if rel else d
                if d in skip_dirs or child in skip_dirs:
                    continue
                keep.append(d)
                self.dirs.setdefault(d, []).append(child)
            dirnames[:] = keep
            for fn in filenames:
                child = f"{rel}/{fn}" if rel else fn
                self.files.setdefault(fn, []).append(child)

    def markdown_files(self) -> list[str]:
        return sorted(p for paths in self.files.values() for p in paths if p.endswith(".md"))


def _trail_overlap(old_parts: list[str], cand_parts: list[str]) -> int:
    """Number of path components shared at the tail of old and candidate."""
    n = 0
    for a, b in zip(reversed(old_parts), reversed(cand_parts)):
        if a != b:
            break
        n += 1
    return n


def pick_candidate(old_rel: str, candidates: list[str]) -> str | None:
    """Choose the most plausible new location for a moved path.

    Prefers the candidate sharing the most trailing path components with the
    old path; ties are rejected so the janitor never guesses.
    """
    if not candidates:
        return None
    old_parts = [p for p in old_rel.split("/") if p]
    scored = sorted(
        (-_trail_overlap(old_parts, c.split("/")), abs(len(c.split("/")) - len(old_parts)), c)
        for c in candidates
    )
    if len(scored) >= 2 and scored[0][:2] == scored[1][:2]:
        return None
    return scored[0][2]


# ─── File helpers ────────────────────────────────────────────────────────────
def read_text(path: str) -> str:
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)


def map_outside_fences(text: str, transform: Callable[[str], str]) -> str:
    """Apply transform to markdown text but never inside fenced code blocks."""
    parts = FENCE_RE.split(text)
    return "".join(p if i % 2 else transform(p) for i, p in enumerate(parts))


def _segment_transform(
    repo: str,
    md: str,
    index: RepoIndex,
    fixes: list[Fix],
    problems: list[Problem],
    stats: Stats,
) -> Callable[[str], str]:
    md_dir = os.path.dirname(md)

    def transform(seg: str) -> str:
        def sub_link(m: re.Match[str]) -> str:
            # re.sub replaces the whole match, so rebuild "[text](target)".
            target = _fix_internal_target(
                m.group(2), repo, md_dir, md, index, fixes, problems, stats
            )
            return f"[{m.group(1)}]({target})"

        def sub_ref(m: re.Match[str]) -> str:
            return (
                m.group(1)
                + _fix_internal_target(m.group(2), repo, md_dir, md, index, fixes, problems, stats)
                + m.group(3)
            )

        seg = MD_LINK_RE.sub(sub_link, seg)
        return MD_REF_RE.sub(sub_ref, seg)

    return transform


def _fix_internal_target(
    raw: str,
    repo: str,
    md_dir: str,
    md: str,
    index: RepoIndex,
    fixes: list[Fix],
    problems: list[Problem],
    stats: Stats,
) -> str:
    """Return the repaired link target, or the original when untouched."""
    if raw.startswith(("http://", "https://", "mailto:", "#", "<")):
        return raw
    split = TARGET_SPLIT_RE.match(raw)
    if not split:
        return raw
    path_part, suffix = split.group(1) or "", split.group(2) or ""
    if not path_part:
        return raw
    decoded = unquote(path_part)
    if decoded.startswith(("http://", "https://", "mailto:")):
        return raw
    stats.internal_links += 1

    if decoded.startswith("/"):
        current = os.path.join(repo, decoded.lstrip("/"))
        old_rel = decoded.lstrip("/")
    else:
        current = os.path.normpath(os.path.join(repo, md_dir, decoded))
        old_rel = posixpath.normpath(
            posixpath.join(md_dir.replace(os.sep, "/"), decoded.replace(os.sep, "/"))
        )
    if os.path.exists(current):
        return raw

    base = posixpath.basename(decoded)
    candidates = index.files.get(base, []) + index.dirs.get(base, [])
    cand = pick_candidate(old_rel, candidates)
    if cand is None:
        problems.append(
            Problem(
                "internal-link", md, f"`{raw}` -> target not found", "no unambiguous relocation"
            )
        )
        return raw
    rel = os.path.relpath(os.path.join(repo, cand), os.path.join(repo, md_dir)).replace(os.sep, "/")
    new_link = quote(rel, safe="/._~") + suffix
    fixes.append(Fix("internal-link", md, raw, new_link))
    return new_link


def fix_internal_links(
    repo: str,
    md_files: list[str],
    index: RepoIndex,
    apply: bool,
    fixes: list[Fix],
    problems: list[Problem],
    stats: Stats,
) -> None:
    """Rewrite internal markdown links whose target moved."""
    for md in md_files:
        text = read_text(os.path.join(repo, md))
        new_text = map_outside_fences(
            text, _segment_transform(repo, md, index, fixes, problems, stats)
        )
        if new_text != text and apply:
            write_text(os.path.join(repo, md), new_text)


# ─── External links ──────────────────────────────────────────────────────────
class _RedirectRecorder(urllib.request.HTTPRedirectHandler):
    """HTTPRedirectHandler that remembers which redirect codes it saw."""

    def __init__(self) -> None:
        self.statuses: list[int] = []

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        self.statuses.append(int(code))
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _probe(url: str, timeout: int, ua: str) -> tuple[int | None, str, list[int]]:
    """Return (final status, final URL, redirect codes seen) for a URL."""
    opener = urllib.request.build_opener(_RedirectRecorder())
    recorder = next(h for h in opener.handlers if isinstance(h, _RedirectRecorder))
    code: int | None = None
    final_url = url
    for method in ("HEAD", "GET"):
        recorder.statuses.clear()
        req = urllib.request.Request(url, method=method, headers={"User-Agent": ua})  # noqa: S310 - http/https enforced by caller
        try:
            with opener.open(req, timeout=timeout) as resp:
                return int(resp.getcode() or 0), str(resp.geturl()), list(recorder.statuses)
        except urllib.error.HTTPError as exc:
            code = int(exc.code)
        except (urllib.error.URLError, OSError, ValueError):
            code = None
        # HEAD is often rejected (403/404/405); retry once with GET before
        # declaring the URL dead.
        if method == "HEAD" and code in (400, 401, 403, 404, 405, 429):
            continue
        return code, final_url, list(recorder.statuses)
    return code, final_url, []


def _external_url_collector(
    found: dict[str, list[str]],
    md: str,
    counter: dict[str, int],
) -> Callable[[str], str]:
    def collect(seg: str) -> str:
        for regex in (MD_LINK_RE, MD_REF_RE):
            for m in regex.finditer(seg):
                raw = m.group(2)
                if raw.startswith(("http://", "https://")):
                    counter["n"] += 1
                    files = found.setdefault(raw, [])
                    if md not in files:
                        files.append(md)
        return seg

    return collect


def collect_external_urls(repo: str, md_files: list[str]) -> tuple[dict[str, list[str]], int]:
    """Map raw URL -> markdown files referencing it (outside code fences)."""
    found: dict[str, list[str]] = {}
    occurrences = 0
    for md in md_files:
        text = read_text(os.path.join(repo, md))
        counter = {"n": 0}
        map_outside_fences(text, _external_url_collector(found, md, counter))
        occurrences += counter["n"]
    return found, occurrences


def check_external_urls(
    repo: str,
    md_files: list[str],
    cfg: dict,
    apply: bool,
    fixes: list[Fix],
    problems: list[Problem],
    stats: Stats,
) -> None:
    """Probe external links; rewrite permanently moved ones, report dead ones."""
    ext_cfg = cfg["external"]
    urls, occurrences = collect_external_urls(repo, md_files)
    stats.external_occurrences += occurrences
    stats.external_unique += len(urls)
    skip = [re.compile(p) for p in ext_cfg["skip_url_patterns"]]
    alive = set(ext_cfg["alive_status_codes"])
    moved_codes = set(ext_cfg["moved_status_codes"])
    dead_codes = set(ext_cfg["dead_status_codes"])

    targets = {u: files for u, files in urls.items() if not any(p.search(u) for p in skip)}
    results: dict[str, tuple[int | None, str, list[int]]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(ext_cfg["max_workers"])) as pool:
        futures = {
            pool.submit(_probe, u, int(ext_cfg["timeout_seconds"]), str(ext_cfg["user_agent"])): u
            for u in targets
        }
        for fut in concurrent.futures.as_completed(futures):
            results[futures[fut]] = fut.result()

    rewrites: dict[str, dict[str, str]] = {}  # md -> {raw -> new_raw}
    for url, (code, final_url, chain) in sorted(results.items()):
        if code in alive:
            stats.external_ok += 1
        elif code in dead_codes:
            stats.external_dead += 1
            problems.append(
                Problem(
                    "external-link",
                    ", ".join(sorted(set(targets[url]))),
                    f"`{url}` returns HTTP {code}",
                    "site gone, needs a human decision",
                )
            )
        if code is None:
            continue
        if not (set(chain) & moved_codes):
            continue
        final = final_url.split("#")[0]
        if not final.startswith(("http://", "https://")) or final == url.split("#")[0]:
            continue
        fragment = "#" + url.split("#", 1)[1] if "#" in url else ""
        new_raw = final + fragment
        stats.external_moved += 1
        for md in sorted(set(targets[url])):
            rewrites.setdefault(md, {})[url] = new_raw
            fixes.append(
                Fix("external-link", md, url, new_raw, f"HTTP {'/'.join(map(str, chain))} redirect")
            )

    if not apply:
        return
    for md, mapping in rewrites.items():
        path = os.path.join(repo, md)
        text = read_text(path)
        write_text(path, map_outside_fences(text, _url_substitutor(mapping)))


def _url_substitutor(mapping: dict[str, str]) -> Callable[[str], str]:
    def substitute(seg: str) -> str:
        for old, new in mapping.items():
            seg = seg.replace(old, new)
        return seg

    return substitute


# ─── Dependabot paths ────────────────────────────────────────────────────────
def _has_manifest(dir_path: str, manifests: tuple[str, ...]) -> bool:
    if not manifests:
        return os.path.isdir(dir_path)
    return any(os.path.exists(os.path.join(dir_path, m)) for m in manifests)


def fix_dependabot(
    repo: str,
    cfg: dict,
    index: RepoIndex,
    apply: bool,
    fixes: list[Fix],
    problems: list[Problem],
    stats: Stats,
) -> None:
    """Repair `directory:` entries in dependabot.yml that point nowhere."""
    rel_cfg = str(cfg.get("dependabot_config", ".github/dependabot.yml"))
    path = os.path.join(repo, rel_cfg)
    if not os.path.exists(path):
        return
    data, err = load_yaml(path)
    if data is None:
        problems.append(Problem("dependabot", rel_cfg, "unparseable YAML", err))
        return
    rewrites: list[tuple[str, str]] = []
    for entry in data.get("updates") or []:
        eco = str(entry.get("package-ecosystem", ""))
        directory = str(entry.get("directory", ""))
        stats.dependabot_dirs += 1
        if not directory or directory in ("/", "."):
            continue
        rel = directory.strip("/")
        if os.path.exists(os.path.join(repo, rel)):
            continue
        last = rel.split("/")[-1]
        cand_dirs = [
            c
            for c in index.dirs.get(last, [])
            if _has_manifest(os.path.join(repo, c), MANIFESTS.get(eco, ()))
        ]
        cand = pick_candidate(rel, cand_dirs)
        if cand is None:
            problems.append(
                Problem(
                    "dependabot",
                    rel_cfg,
                    f"`{eco}` directory `{directory}` does not exist",
                    "no matching manifest directory found",
                )
            )
            continue
        new_dir = "/" + cand
        rewrites.append((directory, new_dir))
        fixes.append(
            Fix("dependabot", rel_cfg, directory, new_dir, f"{eco} manifest found in `{cand}`")
        )
    if not (apply and rewrites):
        return
    text = read_text(path)
    for old, new in rewrites:
        pattern = re.compile(r"(directory:\s*)([\"'])" + re.escape(old) + r"\2")
        text, n = pattern.subn(f'\\g<1>"{new}"', text, count=1)
        if n == 0:
            problems.append(
                Problem(
                    "dependabot",
                    rel_cfg,
                    f"cannot rewrite `{old}`",
                    "entry not found verbatim in YAML",
                )
            )
    write_text(path, text)


# ─── MkDocs navigation ───────────────────────────────────────────────────────
def _collect_nav_strings(node: object, acc: list[str]) -> None:
    if isinstance(node, str):
        acc.append(node)
    elif isinstance(node, list):
        for item in node:
            _collect_nav_strings(item, acc)
    elif isinstance(node, dict):
        for value in node.values():
            _collect_nav_strings(value, acc)


def fix_mkdocs_nav(
    repo: str,
    cfg: dict,
    index: RepoIndex,
    apply: bool,
    fixes: list[Fix],
    problems: list[Problem],
    stats: Stats,
) -> None:
    """Repair `nav:` entries in mkdocs.yml whose target file moved."""
    rel_cfg = str(cfg.get("mkdocs_config", "mkdocs.yml"))
    path = os.path.join(repo, rel_cfg)
    if not os.path.exists(path):
        return
    data, err = load_yaml(path)
    if data is None:
        problems.append(Problem("mkdocs-nav", rel_cfg, "unparseable YAML", err))
        return
    docs_dir = str(data.get("docs_dir") or "docs").strip("/")
    strings: list[str] = []
    _collect_nav_strings(data.get("nav") or [], strings)
    rewrites: list[tuple[str, str]] = []
    for entry in strings:
        if entry.startswith(("!", "http://", "https://")) or not entry.endswith(".md"):
            continue
        stats.nav_entries += 1
        decoded = unquote(entry)
        if os.path.exists(os.path.join(repo, docs_dir, decoded)):
            continue
        base = posixpath.basename(decoded)
        cand_files = [c for c in index.files.get(base, []) if c.startswith(docs_dir + "/")]
        cand = pick_candidate(f"{docs_dir}/{decoded}", cand_files)
        if cand is None:
            problems.append(
                Problem(
                    "mkdocs-nav",
                    rel_cfg,
                    f"nav entry `{entry}` not found under `{docs_dir}/`",
                    "no unambiguous relocation found",
                )
            )
            continue
        new_entry = quote(
            os.path.relpath(os.path.join(repo, cand), os.path.join(repo, docs_dir)).replace(
                os.sep, "/"
            ),
            safe="/._~",
        )
        rewrites.append((entry, new_entry))
        fixes.append(Fix("mkdocs-nav", rel_cfg, entry, new_entry))
    if not (apply and rewrites):
        return
    text = read_text(path)
    for old, new in rewrites:
        if f"'{old}'" in text or f'"{old}"' in text:
            text = text.replace(f"'{old}'", f"'{new}'").replace(f'"{old}"', f'"{new}"')
        elif text.count(old) == 1:
            text = text.replace(old, new)
        else:
            problems.append(
                Problem(
                    "mkdocs-nav",
                    rel_cfg,
                    f"cannot safely rewrite `{old}`",
                    "value appears multiple times in YAML",
                )
            )
    write_text(path, text)


# ─── Reporting ───────────────────────────────────────────────────────────────
def write_reports(
    fixes: list[Fix],
    problems: list[Problem],
    stats: Stats,
    apply: bool,
    report_base: str,
) -> None:
    """Write a human-readable .md and a machine-readable .json report."""
    now = time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime())
    status = "applied" if apply else "proposed (rerun with --fix)"
    by_kind: dict[str, int] = {}
    for fix in fixes:
        by_kind[fix.kind] = by_kind.get(fix.kind, 0) + 1
    prob_by_kind: dict[str, int] = {}
    for problem in problems:
        prob_by_kind[problem.kind] = prob_by_kind.get(problem.kind, 0) + 1

    rows = [
        (
            "Internal links",
            stats.internal_links,
            by_kind.get("internal-link", 0),
            prob_by_kind.get("internal-link", 0),
        ),
        (
            "External links",
            stats.external_unique,
            by_kind.get("external-link", 0),
            prob_by_kind.get("external-link", 0),
        ),
        (
            "Dependabot paths",
            stats.dependabot_dirs,
            by_kind.get("dependabot", 0),
            prob_by_kind.get("dependabot", 0),
        ),
        (
            "MkDocs nav entries",
            stats.nav_entries,
            by_kind.get("mkdocs-nav", 0),
            prob_by_kind.get("mkdocs-nav", 0),
        ),
    ]
    lines = [
        f"# Repo Janitor report - {now}",
        "",
        f"Mode: fixes **{status}**.",
        "",
        "## Summary",
        "",
        "| Category | Checked | Fixes | Unfixable |",
        "|---|---:|---:|---:|",
    ]
    lines += [
        f"| {name} | {checked} | {fixed} | {broken} |" for name, checked, fixed, broken in rows
    ]
    lines += [
        f"| External URLs probed (unique) | {stats.external_unique} | {by_kind.get('external-link', 0)} | {prob_by_kind.get('external-link', 0)} |",
        "",
        "## Fixes",
        "",
    ]
    if fixes:
        lines += [fix.line() for fix in fixes]
    else:
        lines.append("No fixes were needed.")
    lines += ["", "## Unfixable (needs a human)", ""]
    if problems:
        lines += [problem.line() for problem in problems]
    else:
        lines.append("Nothing unfixable - repository is healthy.")

    write_text(report_base + ".md", "\n".join(lines) + "\n")
    write_text(
        report_base + ".json",
        json.dumps(
            {
                "generated": now,
                "applied": apply,
                "stats": dataclasses.asdict(stats),
                "fixes": [dataclasses.asdict(f) for f in fixes],
                "problems": [dataclasses.asdict(p) for p in problems],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
    )


# ─── Entry point ─────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Repo Janitor - find and auto-fix repository-health issues."
    )
    parser.add_argument(
        "--repo", default=REPO_ROOT, help="repository root (default: this checkout)"
    )
    parser.add_argument(
        "--config", default=None, help="janitor.yml path (default: <repo>/.github/janitor.yml)"
    )
    parser.add_argument("--fix", action="store_true", help="apply fixes (default: report only)")
    parser.add_argument(
        "--online", action="store_true", help="probe external http(s) links over the network"
    )
    parser.add_argument("--report", metavar="BASE", help="write BASE.md and BASE.json report files")
    parser.add_argument(
        "--no-fail", action="store_true", help="always exit 0, even with unfixable issues"
    )
    args = parser.parse_args(argv)

    repo = os.path.abspath(args.repo)
    if not os.path.isdir(repo):
        print(f"repo-janitor: repository root not found: {repo}")
        return 2
    cfg = load_config(args.config or os.path.join(repo, ".github", "janitor.yml"))
    index = RepoIndex(repo, list(cfg["skip_dirs"]))
    md_files = index.markdown_files()

    fixes: list[Fix] = []
    problems: list[Problem] = []
    stats = Stats()

    fix_internal_links(repo, md_files, index, args.fix, fixes, problems, stats)
    if args.online:
        check_external_urls(repo, md_files, cfg, args.fix, fixes, problems, stats)
    fix_dependabot(repo, cfg, index, args.fix, fixes, problems, stats)
    fix_mkdocs_nav(repo, cfg, index, args.fix, fixes, problems, stats)

    if args.report:
        write_reports(fixes, problems, stats, args.fix, os.path.abspath(args.report))

    print(
        f"repo-janitor: {len(md_files)} markdown files, {stats.internal_links} internal links checked"
    )
    if args.online:
        print(
            f"repo-janitor: {stats.external_unique} external URLs probed "
            f"({stats.external_ok} ok, {stats.external_moved} moved, {stats.external_dead} dead)"
        )
    for fix in fixes:
        print(f"FIX       {fix.line()}")
    for problem in problems:
        print(f"UNFIXABLE {problem.line()}")
    verdict = (
        f"{len(fixes)} fixes {'applied' if args.fix else 'proposed'}, {len(problems)} unfixable"
    )
    print(f"repo-janitor: {verdict}")
    if problems and not args.no_fail:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
