"""
reports.py — Генератор отчётов во многих форматах для лаборатории RMT-LLM
=========================================================================

Генерирует отчёты во всех 13 форматах, указанных пользователем:
  1. TXT    — простой текст
  2. MD     — Markdown
  3. CSV    — значения через запятую (табличный)
  4. HTML   — автономный HTML со встроенными графиками
  5. JSON   — структурированный JSON
  6. PDF    — PDF через ReportLab
  7. DOCX   — Microsoft Word через python-docx
  8. YAML   — YAML, удобный для конфигов
  9. XML    — XML для устаревших систем
 10. LaTeX  — .tex для академической публикации
 11. Parquet — колоночный формат для big-data
 12. XLSX   — Microsoft Excel
 13. SQLite — SQL-запрашиваемая база данных

Каждый отчёт содержит:
  (a) Подробные результаты с пояснениями (верхняя секция)
  (b) Полные логи запуска задачи (нижняя секция)

Автор: Исхак Хамзатович Исаев
Лицензия: Проприетарная — Все права защищены.
"""

from __future__ import annotations

import csv
import datetime
import json
import os
import sqlite3
from typing import Any


# ---------------------------------------------------------------------------
# Публичный API
# ---------------------------------------------------------------------------
def generate_all_reports(
    results: dict[str, Any],
    logs: list[str],
    out_dir: str = "results/reports",
    experiment_name: str = "rmt_llm_experiment",
) -> dict[str, str]:
    """Генерирует все 13 форматов отчётов. Возвращает {формат: путь}."""
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
        written["pdf"] = f"[PDF не создан: {exc}]"

    # 7. DOCX
    try:
        written["docx"] = _write_docx(results, logs, out_dir, experiment_name)
    except Exception as exc:
        written["docx"] = f"[DOCX не создан: {exc}]"

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
        written["parquet"] = f"[Parquet не создан: {exc}]"

    # 12. XLSX
    try:
        written["xlsx"] = _write_xlsx(results, logs, out_dir, experiment_name)
    except Exception as exc:
        written["xlsx"] = f"[XLSX не создан: {exc}]"

    # 13. SQLite
    try:
        written["sqlite"] = _write_sqlite(results, logs, out_dir, experiment_name)
    except Exception as exc:
        written["sqlite"] = f"[SQLite не создан: {exc}]"

    return written


# ---------------------------------------------------------------------------
# Построение текстового содержимого
# ---------------------------------------------------------------------------
def _build_text(results: dict[str, Any], logs: list[str]) -> str:
    out = []
    out.append("=" * 78)
    out.append("Лаборатория RMT-LLM — Отчёт об эксперименте (TXT)")
    out.append(f"Создано: {datetime.datetime.now().isoformat()}")
    out.append("=" * 78)
    out.append("")
    out.append("ЧАСТЬ I — ПОДРОБНЫЕ РЕЗУЛЬТАТЫ С ПОЯСНЕНИЯМИ")
    out.append("-" * 78)
    out.append(_explain_results(results))
    out.append("")
    out.append("ЧАСТЬ II — ПОЛНЫЕ ЛОГИ ЗАПУСКА ЗАДАЧИ")
    out.append("-" * 78)
    for line in logs:
        out.append(line)
    out.append("")
    out.append("=" * 78)
    out.append("Конец отчёта.")
    return "\n".join(out)


def _build_markdown(results: dict[str, Any], logs: list[str]) -> str:
    out = []
    out.append("# Лаборатория RMT-LLM — Отчёт об эксперименте")
    out.append("")
    out.append(f"**Создано:** {datetime.datetime.now().isoformat()}  ")
    out.append(f"**Эксперимент:** {results.get('experiment_name', 'rmt_llm_experiment')}  ")
    out.append(f"**Язык:** {results.get('language', 'python')}  ")
    out.append(f"**Версия:** {results.get('version', 'en')}  ")
    out.append("")
    out.append("---")
    out.append("")
    out.append("## Часть I — Подробные результаты с пояснениями")
    out.append("")
    out.append(_explain_results_md(results))
    out.append("")
    out.append("---")
    out.append("")
    out.append("## Часть II — Полные логи запуска задачи")
    out.append("")
    out.append("```")
    for line in logs:
        out.append(line)
    out.append("```")
    out.append("")
    return "\n".join(out)


def _build_html(results: dict[str, Any], logs: list[str]) -> str:
    md = _explain_results_md(results)
    # Простая конвертация MD → HTML
    import html as htmllib

    html_parts = [
        "<!DOCTYPE html>",
        "<html lang='ru'>",
        "<head>",
        "<meta charset='utf-8'>",
        "<title>Отчёт лаборатории RMT-LLM</title>",
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
    html_parts.append("<h1>Лаборатория RMT-LLM — Отчёт об эксперименте</h1>")
    html_parts.append(
        f"<div class='meta'><strong>Создано:</strong> "
        f"{htmllib.escape(datetime.datetime.now().isoformat())}<br>"
        f"<strong>Эксперимент:</strong> "
        f"{htmllib.escape(str(results.get('experiment_name', '')))}<br>"
        f"<strong>Язык:</strong> "
        f"{htmllib.escape(str(results.get('language', '')))}<br>"
        f"<strong>Версия:</strong> "
        f"{htmllib.escape(str(results.get('version', '')))}</div>"
    )
    html_parts.append("<h2>Часть I — Подробные результаты с пояснениями</h2>")
    # Простая конвертация MD в HTML
    for line in md.split("\n"):
        if line.startswith("### "):
            html_parts.append(f"<h3>{htmllib.escape(line[4:])}</h3>")
        elif line.startswith("## "):
            html_parts.append(f"<h2>{htmllib.escape(line[3:])}</h2>")
        elif line.startswith("| "):
            # Таблицы в HTML-версии для простоты пропускаем
            html_parts.append(f"<code>{htmllib.escape(line)}</code><br>")
        elif line.strip():
            html_parts.append(f"<p>{htmllib.escape(line)}</p>")
    html_parts.append("<h2>Часть II — Полные логи запуска задачи</h2>")
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
        r"\usepackage[T2A]{fontenc}",
        r"\usepackage[russian,english]{babel}",
        r"\usepackage{geometry}",
        r"\geometry{a4paper,margin=1in}",
        r"\usepackage{hyperref}",
        r"\usepackage{booktabs}",
        r"\usepackage{longtable}",
        r"\title{Лаборатория RMT-LLM --- Отчёт об эксперименте}",
        r"\author{Исхак Хамзатович Исаев}",
        r"\date{\today}",
        r"\begin{document}",
        r"\maketitle",
        r"\section{Подробные результаты с пояснениями}",
    ]
    out.append(_explain_results_md(results).replace("#", "").replace("|", " | "))
    out.append(r"\section{Полные логи запуска задачи}")
    out.append(r"\begin{verbatim}")
    out.extend(logs)
    out.append(r"\end{verbatim}")
    out.append(r"\end{document}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Поясняющий текст — проза «что это значит?»
# ---------------------------------------------------------------------------
def _explain_results(results: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"Эксперимент: {results.get('experiment_name', 'неизвестен')}")
    lines.append(f"Язык:        {results.get('language', 'неизвестен')}")
    lines.append(f"Версия:      {results.get('version', 'неизвестна')}")
    lines.append("")
    lines.append("Сценарий: " + str(results.get("scenario_name", "по умолчанию")))
    lines.append("  " + str(results.get("scenario_description", "")))
    lines.append("")
    lines.append("Ключевые метрики:")
    for k, v in results.get("metrics", {}).items():
        lines.append(f"  {k:30s} = {v}")
    lines.append("")
    lines.append("Сводка RMT-спектрального анализа:")
    spec = results.get("spectral", {})
    for k, v in spec.items():
        if not isinstance(v, (list, dict)):
            lines.append(f"  {k:30s} = {v}")
    lines.append("")
    lines.append("Сводка цепочки рассуждений:")
    rt = results.get("reasoning_trace", {})
    for k in ("mean_honesty", "mean_deception", "mean_hallucination", "filter_bypass_count"):
        if k in rt:
            lines.append(f"  {k:30s} = {rt[k]}")
    lines.append("")
    lines.append("Интерпретация:")
    lines.append("  Закон Марченко-Пастура (MP) описывает распределение bulk'")
    lines.append("  собственных значений больших случайных ковариационных матриц.")
    lines.append("  Когда наибольшее эмпирическое собственное значение превышает")
    lines.append("  верхнюю границу MP, это сигнализирует о структурированной")
    lines.append("  (не случайной) информации — т.е. модель «обнаружила» факт.")
    lines.append("  Наоборот, когда собственные значения остаются внутри MP-bulk,")
    lines.append("  модель генерирует творческий/галлюцинаторный контент.")
    lines.append("")
    lines.append("  Порог N_crit — это количество авторегрессионных токенов, свыше")
    lines.append("  которого спектральный коллапс делает галлюцинации математически")
    lines.append("  неизбежными. Ниже N_crit модель ещё может самокорректироваться;")
    lines.append("  выше — дробная динамика Капуто берёт верх, и модель попадает в")
    lines.append("  «ловушку полезности», описанную в основной монографии проекта.")
    return "\n".join(lines)


def _explain_results_md(results: dict[str, Any]) -> str:
    out = []
    out.append("### Метаданные эксперимента")
    out.append("")
    out.append(f"- **Название:** `{results.get('experiment_name', 'неизвестен')}`")
    out.append(f"- **Язык:** `{results.get('language', 'неизвестен')}`")
    out.append(f"- **Версия:** `{results.get('version', 'неизвестна')}`")
    out.append(f"- **Сценарий:** `{results.get('scenario_name', 'по умолчанию')}`")
    out.append("")
    out.append(f"_{results.get('scenario_description', '')}_")
    out.append("")
    out.append("### Ключевые метрики")
    out.append("")
    out.append("| Метрика | Значение |")
    out.append("|---|---|")
    for k, v in results.get("metrics", {}).items():
        out.append(f"| {k} | {v} |")
    out.append("")
    out.append("### RMT-спектральный анализ")
    out.append("")
    out.append("| Величина | Значение | Интерпретация |")
    out.append("|---|---|---|")
    spec = results.get("spectral", {})
    interp = {
        "mp_upper": "Верхняя граница MP-bulk. Собственные значения выше неё указывают на структурированный (фактический) сигнал.",
        "mp_lower": "Нижняя граница MP-bulk.",
        "lambda_max": "Наибольшее эмпирическое собственное значение. Выше mp_upper ⇒ режим фактического извлечения.",
        "signal_detected": "Истинно, если λ_max > mp_upper (сработал BBP-переход).",
        "tw_fluctuation": "Нормированная флуктуация Трейси-Видома. Большие положительные значения указывают на появление сигнала.",
    }
    for k, v in spec.items():
        if not isinstance(v, (list, dict)):
            out.append(f"| {k} | {v} | {interp.get(k, '')} |")
    out.append("")
    out.append("### Цепочка рассуждений (скрытая CoT)")
    out.append("")
    rt = results.get("reasoning_trace", {})
    out.append("| Метрика | Значение |")
    out.append("|---|---|")
    for k in ("mean_honesty", "mean_deception", "mean_hallucination", "filter_bypass_count"):
        if k in rt:
            out.append(f"| {k} | {rt[k]:.4f} |")
    out.append("")
    out.append("### Интерпретация")
    out.append("")
    out.append(
        "**Закон Марченко-Пастура (MP)** описывает распределение bulk "
        "собственных значений больших случайных ковариационных матриц. "
        "Когда наибольшее эмпирическое собственное значение превышает "
        "верхнюю границу MP (`mp_upper`), это сигнализирует о "
        "структурированной (не случайной) информации — т.е. модель "
        "«обнаружила» факт. Наоборот, когда собственные значения остаются "
        "внутри MP-bulk, модель генерирует творческий или "
        "галлюцинаторный контент."
    )
    out.append("")
    out.append(
        "**Порог N_crit** — это количество авторегрессионных токенов, "
        "свыше которого спектральный коллапс делает галлюцинации "
        "математически неизбежными. Ниже N_crit модель ещё может "
        "самокорректироваться; выше — дробная динамика Капуто берёт "
        "верх, и модель попадает в **«ловушку полезности»**, описанную "
        "в основной монографии проекта. Наблюдаемое поведение "
        "подтверждает (или опровергает) предсказание фреймворка RMT-LLM."
    )
    out.append("")
    out.append(
        "**Цепочка рассуждений** раскрывает скрытую CoT модели. "
        "`mean_deception` > `mean_honesty` указывает, что модель внутренне "
        "планирует ввести пользователя в заблуждение, а "
        "`filter_bypass_count > 0` подтверждает, что защитные фильтры "
        "срабатывают только на этапе вывода, а не на этапе рассуждения — "
        "в точности так, как сообщается в исходной новости."
    )
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Запись файлов
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
        # Запасной вариант: простой YAML key=value
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
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        Paragraph,
        Preformatted,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    # Регистрируем кириллический TTF-шрифт (если доступен)
    cyrillic_font = "Helvetica"
    cyrillic_bold = "Helvetica-Bold"
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        if os.path.exists(candidate):
            try:
                pdfmetrics.registerFont(TTFont("CyrFont", candidate))
                cyrillic_font = "CyrFont"
                cyrillic_bold = "CyrFont"
                break
            except Exception:
                pass

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
    h1 = ParagraphStyle("H1Cyr", parent=styles["Heading1"], fontName=cyrillic_bold)
    h2 = ParagraphStyle("H2Cyr", parent=styles["Heading2"], fontName=cyrillic_bold)
    body = ParagraphStyle("BodyCyr", parent=styles["BodyText"], fontName=cyrillic_font)
    mono = ParagraphStyle(
        "MonoCyr",
        parent=styles["Code"],
        fontName=cyrillic_font,
        fontSize=7,
        leading=8,
        alignment=TA_LEFT,
    )
    story = []

    story.append(Paragraph("Лаборатория RMT-LLM — Отчёт об эксперименте", h1))
    story.append(Paragraph(f"Создано: {datetime.datetime.now().isoformat()}", body))
    story.append(Spacer(1, 0.5 * cm))

    story.append(Paragraph("Часть I — Подробные результаты с пояснениями", h2))
    story.append(Paragraph(f"<b>Эксперимент:</b> {results.get('experiment_name', '')}", body))
    story.append(Paragraph(f"<b>Сценарий:</b> {results.get('scenario_name', '')}", body))
    story.append(
        Paragraph(
            f"<b>Язык:</b> {results.get('language', '')} / "
            f"<b>Версия:</b> {results.get('version', '')}",
            body,
        )
    )
    story.append(Spacer(1, 0.3 * cm))

    # Таблица метрик
    story.append(Paragraph("Ключевые метрики", h2))
    metrics = results.get("metrics", {})
    if metrics:
        data = [["Метрика", "Значение"]] + [[str(k), str(v)] for k, v in metrics.items()]
        t = Table(data, colWidths=[8 * cm, 8 * cm])
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3a5c")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), cyrillic_bold),
                    ("FONTNAME", (0, 1), (-1, -1), cyrillic_font),
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

    # Таблица спектра
    spec = results.get("spectral", {})
    if spec:
        story.append(Paragraph("RMT-спектральный анализ", h2))
        data = [["Величина", "Значение"]]
        for k, v in spec.items():
            if not isinstance(v, (list, dict)):
                data.append([str(k), str(v)])
        t = Table(data, colWidths=[8 * cm, 8 * cm])
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c5282")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), cyrillic_bold),
                    ("FONTNAME", (0, 1), (-1, -1), cyrillic_font),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ]
            )
        )
        story.append(t)
        story.append(Spacer(1, 0.4 * cm))

    # Интерпретация
    story.append(Paragraph("Интерпретация", h2))
    interp = (
        "Закон Марченко-Пастура (MP) описывает распределение bulk "
        "собственных значений больших случайных ковариационных матриц. "
        "Когда наибольшее эмпирическое собственное значение превышает "
        "верхнюю границу MP, это сигнализирует о структурированной "
        "(не случайной) информации — т.е. модель «обнаружила» факт. "
        "Ниже порога N_crit модель ещё может самокорректироваться; "
        "выше — дробная динамика Капуто берёт верх, и модель попадает "
        "в «ловушку полезности», описанную в основной монографии проекта."
    )
    story.append(Paragraph(interp, body))
    story.append(Spacer(1, 0.4 * cm))

    # Логи
    story.append(Paragraph("Часть II — Полные логи запуска задачи", h2))
    log_text = "\n".join(logs)
    # Делим, чтобы не упереться в лимит Preformatted 1000 символов
    for i in range(0, len(log_text), 4000):
        story.append(Preformatted(log_text[i : i + 4000], mono))

    doc.build(story)
    return path


def _write_docx(results: dict[str, Any], logs: list[str], out_dir: str, name: str) -> str:
    from docx import Document
    from docx.shared import Pt

    path = os.path.join(out_dir, f"{name}.docx")
    doc = Document()

    # Заголовок
    title = doc.add_heading("Лаборатория RMT-LLM — Отчёт об эксперименте", level=0)
    doc.add_paragraph(f"Создано: {datetime.datetime.now().isoformat()}")

    doc.add_heading("Часть I — Подробные результаты с пояснениями", level=1)
    doc.add_paragraph(f"Эксперимент: {results.get('experiment_name', '')}")
    doc.add_paragraph(f"Сценарий: {results.get('scenario_name', '')}")
    doc.add_paragraph(f"Язык: {results.get('language', '')} / Версия: {results.get('version', '')}")

    doc.add_heading("Ключевые метрики", level=2)
    for k, v in results.get("metrics", {}).items():
        p = doc.add_paragraph()
        r = p.add_run(f"{k}: ")
        r.bold = True
        p.add_run(str(v))

    doc.add_heading("RMT-спектральный анализ", level=2)
    spec = results.get("spectral", {})
    for k, v in spec.items():
        if not isinstance(v, (list, dict)):
            p = doc.add_paragraph()
            r = p.add_run(f"{k}: ")
            r.bold = True
            p.add_run(str(v))

    doc.add_heading("Интерпретация", level=2)
    doc.add_paragraph(
        "Закон Марченко-Пастура (MP) описывает распределение bulk "
        "собственных значений больших случайных ковариационных матриц. "
        "Когда наибольшее эмпирическое собственное значение превышает "
        "верхнюю границу MP, это сигнализирует о структурированной "
        "(не случайной) информации — т.е. модель «обнаружила» факт. "
        "Ниже порога N_crit модель ещё может самокорректироваться; "
        "выше — дробная динамика Капуто берёт верх, и модель попадает "
        "в «ловушку полезности», описанную в основной монографии проекта."
    )

    doc.add_heading("Часть II — Полные логи запуска задачи", level=1)
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

    # Лист 1: Результаты
    ws = wb.active
    ws.title = "Результаты"
    ws["A1"] = "Ключ"
    ws["B1"] = "Значение"
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

    # Лист 2: Логи
    ws2 = wb.create_sheet("Логи")
    ws2["A1"] = "Строка"
    ws2["B1"] = "Содержание"
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
# Вспомогательные функции
# ---------------------------------------------------------------------------
def _flatten(d: dict[str, Any], parent: str = "", sep: str = ".") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{parent}{sep}{k}" if parent else k
        if isinstance(v, dict):
            out.update(_flatten(v, key, sep))
        elif isinstance(v, list):
            out[key] = json.dumps(v, default=str)[:500]  # обрезаем длинные списки
        else:
            out[key] = v
    return out


def _xml_escape(s: Any) -> str:
    s = str(s)
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


if __name__ == "__main__":
    # Дымовой тест
    fake_results = {
        "experiment_name": "smoke_test",
        "language": "python",
        "version": "ru",
        "scenario_name": "SCEN-LIE-01",
        "scenario_description": "Подгонка под известный ответ",
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
        "[INFO] запуск эксперимента",
        "[INFO] загружен TinyGPT (2.1M парам.)",
        "[INFO] выполнение сценария SCEN-LIE-01",
        "[INFO] готово",
    ]
    written = generate_all_reports(fake_results, fake_logs, "results/reports", "smoke_test")
    print(f"Создано отчётов: {len(written)}")
    for fmt, p in written.items():
        print(f"  {fmt:8s} -> {p}")
