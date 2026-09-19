"""
reports.py — Multi-Format Report Generator for RMT-LLM Laboratory
==================================================================

Generates reports in all 13 formats specified by the user:
  1. TXT    — plain text
  2. MD     — Markdown
  3. CSV    — comma-separated values (tabular)
  4. HTML   — self-contained HTML with embedded charts
  5. JSON   — structured JSON
  6. PDF    — PDF via ReportLab
  7. DOCX   — Microsoft Word via python-docx
  8. YAML   — YAML config-friendly
  9. XML    — XML for legacy systems
 10. LaTeX  — .tex for academic publishing
 11. Parquet — columnar for big-data workflows
 12. XLSX   — Microsoft Excel
 13. SQLite — SQL-queryable database

Each report contains:
  (a) Detailed results with explanations (top section)
  (b) Full task launch logs (bottom section)

Author: Iskhak Hamzatovich Isaev
License: Proprietary — All rights reserved.
"""

from __future__ import annotations

import csv
import datetime
import json
import os
import sqlite3
from typing import Any


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_all_reports(
    results: dict[str, Any],
    logs: list[str],
    out_dir: str = "results/reports",
    experiment_name: str = "rmt_llm_experiment",
) -> dict[str, str]:
    """Generate all 13 report formats. Returns {format: path}."""
    os.makedirs(out_dir, exist_ok=True)
    written: dict[str, str] = {}

    # 1. TXT
    written["txt"] = _write_text(_build_text(results, logs), out_dir, experiment_name)

    # 2. MD
    written["md"] = _write_text(_build_markdown(results, logs), out_dir, f"{experiment_name}.md")

    # 3. CSV
    written["csv"] = _write_csv(results, logs, out_dir, experiment_name)

    # 4. HTML
    written["html"] = _write_text(_build_html(results, logs), out_dir, f"{experiment_name}.html")

    # 5. JSON
    written["json"] = _write_json(results, logs, out_dir, experiment_name)

    # 6. PDF
    try:
        written["pdf"] = _write_pdf(results, logs, out_dir, experiment_name)
    except Exception as exc:
        written["pdf"] = f"[PDF failed: {exc}]"

    # 7. DOCX
    try:
        written["docx"] = _write_docx(results, logs, out_dir, experiment_name)
    except Exception as exc:
        written["docx"] = f"[DOCX failed: {exc}]"

    # 8. YAML
    written["yaml"] = _write_yaml(results, logs, out_dir, experiment_name)

    # 9. XML
    written["xml"] = _write_text(_build_xml(results, logs), out_dir, f"{experiment_name}.xml")

    # 10. LaTeX
    written["latex"] = _write_text(_build_latex(results, logs), out_dir, f"{experiment_name}.tex")

    # 11. Parquet
    try:
        written["parquet"] = _write_parquet(results, logs, out_dir, experiment_name)
    except Exception as exc:
        written["parquet"] = f"[Parquet failed: {exc}]"

    # 12. XLSX
    try:
        written["xlsx"] = _write_xlsx(results, logs, out_dir, experiment_name)
    except Exception as exc:
        written["xlsx"] = f"[XLSX failed: {exc}]"

    # 13. SQLite
    try:
        written["sqlite"] = _write_sqlite(results, logs, out_dir, experiment_name)
    except Exception as exc:
        written["sqlite"] = f"[SQLite failed: {exc}]"

    return written


# ---------------------------------------------------------------------------
# Build text-like content
# ---------------------------------------------------------------------------
def _build_text(results: dict[str, Any], logs: list[str]) -> str:
    out = []
    out.append("=" * 78)
    out.append("RMT-LLM Laboratory — Experiment Report (TXT)")
    out.append(f"Generated: {datetime.datetime.now().isoformat()}")
    out.append("=" * 78)
    out.append("")
    out.append("PART I — DETAILED RESULTS WITH EXPLANATIONS")
    out.append("-" * 78)
    out.append(_explain_results(results))
    out.append("")
    out.append("PART II — FULL TASK LAUNCH LOGS")
    out.append("-" * 78)
    for line in logs:
        out.append(line)
    out.append("")
    out.append("=" * 78)
    out.append("End of report.")
    return "\n".join(out)


def _build_markdown(results: dict[str, Any], logs: list[str]) -> str:
    out = []
    out.append("# RMT-LLM Laboratory — Experiment Report")
    out.append("")
    out.append(f"**Generated:** {datetime.datetime.now().isoformat()}  ")
    out.append(f"**Experiment:** {results.get('experiment_name', 'rmt_llm_experiment')}  ")
    out.append(f"**Language:** {results.get('language', 'python')}  ")
    out.append(f"**Version:** {results.get('version', 'en')}  ")
    out.append("")
    out.append("---")
    out.append("")
    out.append("## Part I — Detailed Results with Explanations")
    out.append("")
    out.append(_explain_results_md(results))
    out.append("")
    out.append("---")
    out.append("")
    out.append("## Part II — Full Task Launch Logs")
    out.append("")
    out.append("```")
    for line in logs:
        out.append(line)
    out.append("```")
    out.append("")
    return "\n".join(out)


def _build_html(results: dict[str, Any], logs: list[str]) -> str:
    md = _explain_results_md(results)
    # Simple MD → HTML
    import html as htmllib

    html_parts = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='utf-8'>",
        "<title>RMT-LLM Laboratory Report</title>",
        "<style>",
        "body{font-family:Inter,Segoe UI,Arial,sans-serif;max-width:1100px;margin:2em auto;padding:0 1em;color:#222;line-height:1.55}",
        "h1{color:#1a3a5c;border-bottom:3px solid #1a3a5c;padding-bottom:.3em}",
        "h2{color:#2c5282;margin-top:2em}",
        "h3{color:#2d3748}",
        "table{border-collapse:collapse;margin:1em 0;width:100%}",
        "th,td{border:1px solid #cbd5e0;padding:.5em .8em;text-align:left}",
        "th{background:#edf2f7}",
        "tr:nth-child(even){background:#f7fafc}",
        "pre{background:#1a202c;color:#e2e8f0;padding:1em;border-radius:6px;overflow:auto}",
        "code{background:#edf2f7;padding:.1em .3em;border-radius:3px;font-family:Consolas,monospace}",
        ".meta{background:#ebf8ff;border-left:4px solid #4299e1;padding:.6em 1em;margin:1em 0}",
        "</style>",
        "</head>",
        "<body>",
    ]
    html_parts.append("<h1>RMT-LLM Laboratory — Experiment Report</h1>")
    html_parts.append(
        f"<div class='meta'><strong>Generated:</strong> "
        f"{htmllib.escape(datetime.datetime.now().isoformat())}<br>"
        f"<strong>Experiment:</strong> "
        f"{htmllib.escape(str(results.get('experiment_name', '')))}<br>"
        f"<strong>Language:</strong> "
        f"{htmllib.escape(str(results.get('language', '')))}<br>"
        f"<strong>Version:</strong> "
        f"{htmllib.escape(str(results.get('version', '')))}</div>"
    )
    html_parts.append("<h2>Part I — Detailed Results with Explanations</h2>")
    # Convert MD to HTML simply
    for line in md.split("\n"):
        if line.startswith("### "):
            html_parts.append(f"<h3>{htmllib.escape(line[4:])}</h3>")
        elif line.startswith("## "):
            html_parts.append(f"<h2>{htmllib.escape(line[3:])}</h2>")
        elif line.startswith("| "):
            # Skip tables for simplicity in HTML version
            html_parts.append(f"<code>{htmllib.escape(line)}</code><br>")
        elif line.strip():
            html_parts.append(f"<p>{htmllib.escape(line)}</p>")
    html_parts.append("<h2>Part II — Full Task Launch Logs</h2>")
    html_parts.append("<pre>")
    html_parts.append(htmllib.escape("\n".join(logs)))
    html_parts.append("</pre>")
    html_parts.append("</body></html>")
    return "\n".join(html_parts)


def _build_xml(results: dict[str, Any], logs: list[str]) -> str:
    out = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        "<rmt_llm_report>",
        f"  <generated>{datetime.datetime.now().isoformat()}</generated>",
        "  <experiment>",
        f"    <name>{results.get('experiment_name', '')}</name>",
        f"    <language>{results.get('language', '')}</language>",
        f"    <version>{results.get('version', '')}</version>",
        "  </experiment>",
        "  <results>",
    ]
    for k, v in _flatten(results).items():
        out.append(f"    <{k}>{_xml_escape(v)}</{k}>")
    out.append("  </results>")
    out.append("  <logs>")
    for line in logs:
        out.append(f"    <log>{_xml_escape(line)}</log>")
    out.append("  </logs>")
    out.append("</rmt_llm_report>")
    return "\n".join(out)


def _build_latex(results: dict[str, Any], logs: list[str]) -> str:
    out = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{geometry}",
        r"\geometry{a4paper,margin=1in}",
        r"\usepackage{hyperref}",
        r"\usepackage{booktabs}",
        r"\usepackage{longtable}",
        r"\title{RMT-LLM Laboratory — Experiment Report}",
        r"\author{Iskhak Hamzatovich Isaev}",
        r"\date{\today}",
        r"\begin{document}",
        r"\maketitle",
        r"\section{Detailed Results with Explanations}",
    ]
    out.append(_explain_results_md(results).replace("#", "").replace("|", " | "))
    out.append(r"\section{Full Task Launch Logs}")
    out.append(r"\begin{verbatim}")
    out.extend(logs)
    out.append(r"\end{verbatim}")
    out.append(r"\end{document}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Explanatory text — the "what does this mean?" prose
# ---------------------------------------------------------------------------
def _explain_results(results: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"Experiment: {results.get('experiment_name', 'unknown')}")
    lines.append(f"Language:   {results.get('language', 'unknown')}")
    lines.append(f"Version:    {results.get('version', 'unknown')}")
    lines.append("")
    lines.append("Scenario: " + str(results.get("scenario_name", "default")))
    lines.append("  " + str(results.get("scenario_description", "")))
    lines.append("")
    lines.append("Key metrics:")
    for k, v in results.get("metrics", {}).items():
        lines.append(f"  {k:30s} = {v}")
    lines.append("")
    lines.append("RMT spectral analysis summary:")
    spec = results.get("spectral", {})
    for k, v in spec.items():
        if not isinstance(v, (list, dict)):
            lines.append(f"  {k:30s} = {v}")
    lines.append("")
    lines.append("Reasoning trace summary:")
    rt = results.get("reasoning_trace", {})
    for k in ("mean_honesty", "mean_deception", "mean_hallucination", "filter_bypass_count"):
        if k in rt:
            lines.append(f"  {k:30s} = {rt[k]}")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("  The Marchenko-Pastur (MP) law describes the bulk distribution of")
    lines.append("  eigenvalues of large random covariance matrices. When the largest")
    lines.append("  empirical eigenvalue exceeds the MP upper bound, this signals")
    lines.append("  structured (non-random) information — i.e. the model has 'detected'")
    lines.append("  a fact. Conversely, when eigenvalues stay inside the MP bulk, the")
    lines.append("  model is generating creative/hallucinated content.")
    lines.append("")
    lines.append("  The N_crit threshold is the autoregressive token count beyond which")
    lines.append("  spectral collapse makes hallucination mathematically inevitable.")
    lines.append("  Below N_crit the model can still self-correct; above it, the")
    lines.append("  Caputo fractional dynamics dominate and the model enters the")
    lines.append("  'utility trap' described in the project's main monograph.")
    return "\n".join(lines)


def _explain_results_md(results: dict[str, Any]) -> str:
    out = []
    out.append("### Experiment Metadata")
    out.append("")
    out.append(f"- **Name:** `{results.get('experiment_name', 'unknown')}`")
    out.append(f"- **Language:** `{results.get('language', 'unknown')}`")
    out.append(f"- **Version:** `{results.get('version', 'unknown')}`")
    out.append(f"- **Scenario:** `{results.get('scenario_name', 'default')}`")
    out.append("")
    out.append(f"_{results.get('scenario_description', '')}_")
    out.append("")
    out.append("### Key Metrics")
    out.append("")
    out.append("| Metric | Value |")
    out.append("|---|---|")
    for k, v in results.get("metrics", {}).items():
        out.append(f"| {k} | {v} |")
    out.append("")
    out.append("### RMT Spectral Analysis")
    out.append("")
    out.append("| Quantity | Value | Interpretation |")
    out.append("|---|---|---|")
    spec = results.get("spectral", {})
    interp = {
        "mp_upper": "Upper Marchenko-Pastur bulk bound. Eigenvalues above this indicate structured (factual) signal.",
        "mp_lower": "Lower MP bulk bound.",
        "lambda_max": "Largest empirical eigenvalue. Above mp_upper ⇒ factual recall mode.",
        "signal_detected": "True if λ_max > mp_upper (BBP transition fired).",
        "tw_fluctuation": "Tracy-Widom normalized fluctuation. Large positive values indicate signal emergence.",
    }
    for k, v in spec.items():
        if not isinstance(v, (list, dict)):
            out.append(f"| {k} | {v} | {interp.get(k, '')} |")
    out.append("")
    out.append("### Reasoning Trace (Hidden Chain-of-Thought)")
    out.append("")
    rt = results.get("reasoning_trace", {})
    out.append("| Metric | Value |")
    out.append("|---|---|")
    for k in ("mean_honesty", "mean_deception", "mean_hallucination", "filter_bypass_count"):
        if k in rt:
            out.append(f"| {k} | {rt[k]:.4f} |")
    out.append("")
    out.append("### Interpretation")
    out.append("")
    out.append(
        "The **Marchenko-Pastur (MP) law** describes the bulk distribution of "
        "eigenvalues of large random covariance matrices. When the largest "
        "empirical eigenvalue exceeds the MP upper bound (`mp_upper`), this "
        "signals structured (non-random) information — i.e. the model has "
        "'detected' a fact. Conversely, when eigenvalues stay inside the MP "
        "bulk, the model is generating creative or hallucinated content."
    )
    out.append("")
    out.append(
        "The **N_crit threshold** is the autoregressive token count beyond "
        "which spectral collapse makes hallucination mathematically inevitable. "
        "Below N_crit the model can still self-correct; above it, the Caputo "
        "fractional dynamics dominate and the model enters the **'utility trap'** "
        "described in the project's main monograph. The observed behavior "
        "verifies (or falsifies) the RMT-LLM framework's prediction."
    )
    out.append("")
    out.append(
        "The **reasoning trace** exposes the model's hidden chain-of-thought. "
        "A `mean_deception` > `mean_honesty` indicates the model is internally "
        "planning to mislead the user, while `filter_bypass_count > 0` confirms "
        "that protective filters fire only at output time, not at reasoning time "
        "— exactly as reported in the source news article."
    )
    return "\n".join(out)


# ---------------------------------------------------------------------------
# File writers
# ---------------------------------------------------------------------------
def _write_text(content: str, out_dir: str, name: str) -> str:
    path = os.path.join(
        out_dir, name if name.endswith((".txt", ".md", ".html", ".xml", ".tex")) else f"{name}.txt"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def _write_csv(results: dict[str, Any], logs: list[str], out_dir: str, name: str) -> str:
    path = os.path.join(out_dir, f"{name}.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section", "key", "value"])
        for k, v in _flatten(results).items():
            w.writerow(["results", k, v])
        for i, line in enumerate(logs):
            w.writerow(["log", f"line_{i:05d}", line])
    return path


def _write_json(results: dict[str, Any], logs: list[str], out_dir: str, name: str) -> str:
    path = os.path.join(out_dir, f"{name}.json")
    payload = {
        "generated_at": datetime.datetime.now().isoformat(),
        "results": results,
        "logs": logs,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    return path


def _write_yaml(results: dict[str, Any], logs: list[str], out_dir: str, name: str) -> str:
    path = os.path.join(out_dir, f"{name}.yaml")
    try:
        import yaml  # type: ignore

        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                {
                    "generated_at": datetime.datetime.now().isoformat(),
                    "results": results,
                    "logs": logs,
                },
                f,
                allow_unicode=True,
                sort_keys=False,
            )
    except ImportError:
        # Fallback: simple key=value YAML
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"generated_at: {datetime.datetime.now().isoformat()}\n")
            f.write("results:\n")
            for k, v in _flatten(results).items():
                f.write(f"  {k}: {v}\n")
            f.write("logs:\n")
            for line in logs:
                f.write(f"  - {line!r}\n")
    return path


def _write_pdf(results: dict[str, Any], logs: list[str], out_dir: str, name: str) -> str:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        Paragraph,
        Preformatted,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    path = os.path.join(out_dir, f"{name}.pdf")
    doc = SimpleDocTemplate(
        path,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )
    styles = getSampleStyleSheet()
    h1 = styles["Heading1"]
    h2 = styles["Heading2"]
    body = styles["BodyText"]
    mono = ParagraphStyle("Mono", parent=styles["Code"], fontSize=7, leading=8, alignment=TA_LEFT)
    story = []

    story.append(Paragraph("RMT-LLM Laboratory — Experiment Report", h1))
    story.append(Paragraph(f"Generated: {datetime.datetime.now().isoformat()}", body))
    story.append(Spacer(1, 0.5 * cm))

    story.append(Paragraph("Part I — Detailed Results with Explanations", h2))
    story.append(Paragraph(f"<b>Experiment:</b> {results.get('experiment_name', '')}", body))
    story.append(Paragraph(f"<b>Scenario:</b> {results.get('scenario_name', '')}", body))
    story.append(
        Paragraph(
            f"<b>Language:</b> {results.get('language', '')} / "
            f"<b>Version:</b> {results.get('version', '')}",
            body,
        )
    )
    story.append(Spacer(1, 0.3 * cm))

    # Metrics table
    story.append(Paragraph("Key Metrics", h2))
    metrics = results.get("metrics", {})
    if metrics:
        data = [["Metric", "Value"]] + [[str(k), str(v)] for k, v in metrics.items()]
        t = Table(data, colWidths=[8 * cm, 8 * cm])
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3a5c")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#f0f4f8")],
                    ),
                ]
            )
        )
        story.append(t)
        story.append(Spacer(1, 0.4 * cm))

    # Spectral table
    spec = results.get("spectral", {})
    if spec:
        story.append(Paragraph("RMT Spectral Analysis", h2))
        data = [["Quantity", "Value"]]
        for k, v in spec.items():
            if not isinstance(v, (list, dict)):
                data.append([str(k), str(v)])
        t = Table(data, colWidths=[8 * cm, 8 * cm])
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c5282")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ]
            )
        )
        story.append(t)
        story.append(Spacer(1, 0.4 * cm))

    # Interpretation
    story.append(Paragraph("Interpretation", h2))
    interp = (
        "The Marchenko-Pastur (MP) law describes the bulk distribution of "
        "eigenvalues of large random covariance matrices. When the largest "
        "empirical eigenvalue exceeds the MP upper bound, this signals "
        "structured (non-random) information — i.e. the model has 'detected' "
        "a fact. Below the N_crit threshold the model can still self-correct; "
        "above it, the Caputo fractional dynamics dominate and the model "
        "enters the 'utility trap' described in the project's main monograph."
    )
    story.append(Paragraph(interp, body))
    story.append(Spacer(1, 0.4 * cm))

    # Logs
    story.append(Paragraph("Part II — Full Task Launch Logs", h2))
    log_text = "\n".join(logs)
    # Chunk to avoid 1000-char Preformatted limit
    for i in range(0, len(log_text), 4000):
        story.append(Preformatted(log_text[i : i + 4000], mono))

    doc.build(story)
    return path


def _write_docx(results: dict[str, Any], logs: list[str], out_dir: str, name: str) -> str:
    from docx import Document
    from docx.shared import Pt

    path = os.path.join(out_dir, f"{name}.docx")
    doc = Document()

    # Title
    title = doc.add_heading("RMT-LLM Laboratory — Experiment Report", level=0)
    doc.add_paragraph(f"Generated: {datetime.datetime.now().isoformat()}")

    doc.add_heading("Part I — Detailed Results with Explanations", level=1)
    doc.add_paragraph(f"Experiment: {results.get('experiment_name', '')}")
    doc.add_paragraph(f"Scenario: {results.get('scenario_name', '')}")
    doc.add_paragraph(
        f"Language: {results.get('language', '')} / Version: {results.get('version', '')}"
    )

    doc.add_heading("Key Metrics", level=2)
    for k, v in results.get("metrics", {}).items():
        p = doc.add_paragraph()
        r = p.add_run(f"{k}: ")
        r.bold = True
        p.add_run(str(v))

    doc.add_heading("RMT Spectral Analysis", level=2)
    spec = results.get("spectral", {})
    for k, v in spec.items():
        if not isinstance(v, (list, dict)):
            p = doc.add_paragraph()
            r = p.add_run(f"{k}: ")
            r.bold = True
            p.add_run(str(v))

    doc.add_heading("Interpretation", level=2)
    doc.add_paragraph(
        "The Marchenko-Pastur (MP) law describes the bulk distribution of "
        "eigenvalues of large random covariance matrices. When the largest "
        "empirical eigenvalue exceeds the MP upper bound, this signals "
        "structured (non-random) information — i.e. the model has 'detected' "
        "a fact. Below the N_crit threshold the model can still self-correct; "
        "above it, the Caputo fractional dynamics dominate and the model "
        "enters the 'utility trap' described in the project's main monograph."
    )

    doc.add_heading("Part II — Full Task Launch Logs", level=1)
    for line in logs:
        p = doc.add_paragraph(line)
        p.style = doc.styles["No Spacing"]
        for r in p.runs:
            r.font.name = "Consolas"
            r.font.size = Pt(8)

    doc.save(path)
    return path


def _write_parquet(results: dict[str, Any], logs: list[str], out_dir: str, name: str) -> str:
    import pandas as pd

    path = os.path.join(out_dir, f"{name}.parquet")
    rows = []
    for k, v in _flatten(results).items():
        rows.append({"section": "results", "key": k, "value": str(v)})
    for i, line in enumerate(logs):
        rows.append({"section": "log", "key": f"line_{i:05d}", "value": line})
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    return path


def _write_xlsx(results: dict[str, Any], logs: list[str], out_dir: str, name: str) -> str:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill

    path = os.path.join(out_dir, f"{name}.xlsx")
    wb = Workbook()

    # Sheet 1: Results
    ws = wb.active
    ws.title = "Results"
    ws["A1"] = "Key"
    ws["B1"] = "Value"
    for cell in (ws["A1"], ws["B1"]):
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1a3a5c")
    row = 2
    for k, v in _flatten(results).items():
        ws.cell(row=row, column=1, value=k)
        ws.cell(row=row, column=2, value=str(v))
        row += 1
    ws.column_dimensions["A"].width = 40
    ws.column_dimensions["B"].width = 80

    # Sheet 2: Logs
    ws2 = wb.create_sheet("Logs")
    ws2["A1"] = "Line"
    ws2["B1"] = "Content"
    for cell in (ws2["A1"], ws2["B1"]):
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="2c5282")
    for i, line in enumerate(logs, start=2):
        ws2.cell(row=i, column=1, value=i - 1)
        ws2.cell(row=i, column=2, value=line)
    ws2.column_dimensions["A"].width = 8
    ws2.column_dimensions["B"].width = 120

    wb.save(path)
    return path


def _write_sqlite(results: dict[str, Any], logs: list[str], out_dir: str, name: str) -> str:
    path = os.path.join(out_dir, f"{name}.sqlite")
    if os.path.exists(path):
        os.remove(path)
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS results (
        key TEXT PRIMARY KEY,
        value TEXT,
        section TEXT DEFAULT 'results'
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS logs (
        line_no INTEGER PRIMARY KEY,
        content TEXT
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS experiment_meta (
        key TEXT PRIMARY KEY,
        value TEXT
    )""")
    for k, v in _flatten(results).items():
        cur.execute(
            "INSERT OR REPLACE INTO results (key, value, section) VALUES (?, ?, 'results')",
            (k, str(v)),
        )
    for i, line in enumerate(logs):
        cur.execute("INSERT OR REPLACE INTO logs (line_no, content) VALUES (?, ?)", (i, line))
    cur.execute(
        "INSERT OR REPLACE INTO experiment_meta VALUES ('generated_at', ?)",
        (datetime.datetime.now().isoformat(),),
    )
    conn.commit()
    conn.close()
    return path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _flatten(d: dict[str, Any], parent: str = "", sep: str = ".") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{parent}{sep}{k}" if parent else k
        if isinstance(v, dict):
            out.update(_flatten(v, key, sep))
        elif isinstance(v, list):
            out[key] = json.dumps(v, default=str)[:500]  # truncate long lists
        else:
            out[key] = v
    return out


def _xml_escape(s: Any) -> str:
    s = str(s)
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


if __name__ == "__main__":
    # Smoke test
    fake_results = {
        "experiment_name": "smoke_test",
        "language": "python",
        "version": "en",
        "scenario_name": "SCEN-LIE-01",
        "metrics": {"tokens_generated": 256, "ncrit_exceeded": True, "deception_rate": 0.62},
        "spectral": {
            "mp_upper": 2.7,
            "mp_lower": 0.3,
            "lambda_max": 3.4,
            "signal_detected": True,
            "tw_fluctuation": 1.83,
        },
        "reasoning_trace": {
            "mean_honesty": 0.23,
            "mean_deception": 0.61,
            "mean_hallucination": 0.42,
            "filter_bypass_count": 4,
        },
    }
    fake_logs = [
        "[INFO] starting experiment",
        "[INFO] loaded TinyGPT (2.1M params)",
        "[INFO] running scenario SCEN-LIE-01",
        "[INFO] done",
    ]
    written = generate_all_reports(fake_results, fake_logs, "results/reports", "smoke_test")
    print(f"Reports generated: {len(written)}")
    for fmt, p in written.items():
        print(f"  {fmt:8s} -> {p}")
