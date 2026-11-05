"""compare_reports — differential reports between two runs (P2.8).

After a protocol or code change (e.g. the P0.1 log-probability fix) the
question is: what did the numbers DO? This tool aligns two report.json
files by section/table and prints the deltas:

    python neural_lab/stages/compare_reports.py \
        results/stage2/run_before/report.json \
        results/stage2/run_after/report.json

It is also the regression instrument for the patch itself: a fix that
changes measurement semantics must show up here as a real delta, and a
pure-refactor commit must show deltas of exactly 0.00.
"""

from __future__ import annotations

import argparse
import json
import sys


def _auroc_rows(report: dict) -> dict[str, tuple[float, tuple[float, float] | None]]:
    """detector -> (auroc, ci) from the first section with a detector table."""
    for section in report.get("sections", []):
        table = section.get("table") or {}
        headers = [str(h).lower() for h in table.get("headers", [])]
        if not headers or headers[0] != "detector" or "auroc" not in headers:
            continue
        i_ci = headers.index("95% ci") if "95% ci" in headers else None
        out: dict[str, tuple[float, tuple[float, float] | None]] = {}
        for row in table.get("rows", []):
            if len(row) < 2:
                continue
            try:
                auroc = float(row[1])
            except (TypeError, ValueError):
                continue
            ci = None
            if i_ci is not None and len(row) > i_ci:
                try:
                    lo, hi = str(row[i_ci]).strip("[]").split(",")
                    ci = (float(lo), float(hi))
                except ValueError:
                    ci = None
            out[str(row[0])] = (auroc, ci)
        return out
    return {}


def compare(report_a: dict, report_b: dict) -> list[dict]:
    """detector | AUROC_a | AUROC_b | Δ | CI_a | CI_b (Δ = b − a)."""
    a, b = _auroc_rows(report_a), _auroc_rows(report_b)
    rows = []
    for det in sorted(set(a) | set(b)):
        au_a, ci_a = a.get(det, (None, None))
        au_b, ci_b = b.get(det, (None, None))
        delta = (au_b - au_a) if (au_a is not None and au_b is not None) else None
        rows.append({"detector": det, "auroc_a": au_a, "auroc_b": au_b,
                     "delta": delta, "ci_a": ci_a, "ci_b": ci_b})
    return rows


def format_rows(rows: list[dict]) -> str:
    headers = ["detector", "AUROC_a", "AUROC_b", "Δ", "CI_a", "CI_b"]

    def fmt(v, spec=".3f"):
        if v is None:
            return "—"
        if isinstance(v, tuple):
            return f"[{v[0]:+.3f}, {v[1]:+.3f}]"
        return format(v, spec)

    lines = [" | ".join(headers), "-" * 72]
    for r in rows:
        lines.append(" | ".join([
            r["detector"], fmt(r["auroc_a"]), fmt(r["auroc_b"]),
            fmt(r["delta"], "+.3f"), fmt(r["ci_a"]), fmt(r["ci_b"]),
        ]))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Compare two stage reports (P2.8)")
    ap.add_argument("report_a")
    ap.add_argument("report_b")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = ap.parse_args(argv)

    with open(args.report_a, encoding="utf-8") as f:
        a = json.load(f)
    with open(args.report_b, encoding="utf-8") as f:
        b = json.load(f)
    rows = compare(a, b)
    if not rows:
        print("no detector table found in either report", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        print(f"a = {args.report_a}")
        print(f"b = {args.report_b}")
        print(format_rows(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
