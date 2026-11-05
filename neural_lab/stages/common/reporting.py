"""
common.reporting — report.json + REPORT.md generation from run data.

The report text is GENERATED from the result dictionary (same rule as
science/results): numbers in the .md are formatted from the same values
that were written to the .json — no hand-written numbers.
"""

from __future__ import annotations

import json
import os


def save_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def save_jsonl(path: str, rows) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def fmt_ci(lo: float, hi: float) -> str:
    return f"[{lo:.3f}, {hi:.3f}]"


def write_report(out_dir: str, title: str, passport: dict, sections: list[dict]) -> str:
    """sections: list of {name, summary, table: {headers, rows} | None, notes}.

    Writes passport.json, report.json and REPORT.md. Returns the .md path.
    """
    os.makedirs(out_dir, exist_ok=True)
    save_json(os.path.join(out_dir, "passport.json"), passport)
    save_json(os.path.join(out_dir, "report.json"), {"passport": passport, "sections": sections})

    lines = [f"# {title}", "", f"*Run*: `{passport['experiment_id']}` · "
             f"hypothesis `{passport['hypothesis_id']}` · protocol v{passport['protocol_version']}", ""]
    m = passport.get("model", {})
    if m:
        env = passport.get("environment", {})
        passport_block = [
            "## Passport",
            "",
            f"- model: `{m.get('model_name', 'n/a')}`",
            f"- config_hash: `{m.get('config_hash')}`"
            + (f", weight_sha256: `{m['weight_sha256'][:16]}…`" if m.get("weight_sha256") else ""),
        ]
        # conditional lines are appended only when present: an empty string
        # here would become a stray blank line in the markdown
        if m.get("n_params"):
            passport_block.append(f"- n_params: {m.get('n_params'):,}")
        if m.get("weights_not_hashed"):
            passport_block.append(f"- weights_not_hashed: {m['weights_not_hashed']}")
        passport_block.append(
            f"- transformers {env.get('transformers')}, torch {env.get('torch')}"
        )
        lines += passport_block + [""]
    code = passport.get("code", {})
    if code.get("git_commit"):
        lines += [f"*code*: `{code['git_commit'][:12]}{' (dirty)' if code.get('git_dirty') else ''}`", ""]
    for s in sections:
        lines += [f"## {s['name']}", ""]
        if s.get("summary"):
            lines += [s["summary"], ""]
        if s.get("table"):
            t = s["table"]
            lines.append("| " + " | ".join(t["headers"]) + " |")
            lines.append("|" + "|".join(["---"] * len(t["headers"])) + "|")
            for row in t["rows"]:
                lines.append("| " + " | ".join(str(c) for c in row) + " |")
            lines.append("")
        if s.get("notes"):
            for n in s["notes"]:
                lines.append(f"- {n}")
            lines.append("")
    path = os.path.join(out_dir, "REPORT.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")
    return path
