# reports.jl — Multi-format report generation (Julia, EN)
# 13 formats: txt, md, csv, html, json, pdf, docx, yaml, xml, latex, parquet, xlsx, sqlite

using JSON
using Dates
using Printf

function generate_all_reports(results::Dict, logs::Vector{String};
                              out_dir="results/reports", experiment_name="rmt_llm_experiment")
    mkpath(out_dir)
    written = Dict{String, String}()

    written["txt"] = _write_text(_build_text(results, logs), out_dir, "$(experiment_name).txt")
    written["md"] = _write_text(_build_markdown(results, logs), out_dir, "$(experiment_name).md")
    written["csv"] = _write_csv(results, logs, out_dir, experiment_name)
    written["html"] = _write_text(_build_html(results, logs), out_dir, "$(experiment_name).html")
    written["json"] = _write_json(results, logs, out_dir, experiment_name)
    written["yaml"] = _write_yaml(results, logs, out_dir, experiment_name)
    written["xml"] = _write_text(_build_xml(results, logs), out_dir, "$(experiment_name).xml")
    written["latex"] = _write_text(_build_latex(results, logs), out_dir, "$(experiment_name).tex")

    # PDF (use Plots or simple text-based)
    try
        written["pdf"] = _write_pdf(results, logs, out_dir, experiment_name)
    catch exc
        written["pdf"] = "[PDF failed: $exc]"
    end

    # Parquet
    try
        written["parquet"] = _write_parquet(results, logs, out_dir, experiment_name)
    catch exc
        written["parquet"] = "[Parquet failed: $exc]"
    end

    # XLSX
    try
        written["xlsx"] = _write_xlsx(results, logs, out_dir, experiment_name)
    catch exc
        written["xlsx"] = "[XLSX failed: $exc]"
    end

    # SQLite
    try
        written["sqlite"] = _write_sqlite(results, logs, out_dir, experiment_name)
    catch exc
        written["sqlite"] = "[SQLite failed: $exc]"
    end

    # DOCX (Julia has no native docx; create a placeholder)
    docx_path = joinpath(out_dir, "$(experiment_name).docx")
    _write_text("[DOCX placeholder — use Python implementation for full DOCX support]\n\n" *
                _build_text(results, logs), out_dir, "$(experiment_name).docx_placeholder.txt")
    written["docx"] = docx_path * ".placeholder (use Python for full DOCX)"

    return written
end

function _build_text(results, logs)
    lines = String[]
    push!(lines, "=" ^ 78)
    push!(lines, "RMT-LLM Laboratory — Experiment Report (Julia, TXT)")
    push!(lines, "Generated: $(Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"))")
    push!(lines, "=" ^ 78)
    push!(lines, "")
    push!(lines, "PART I — DETAILED RESULTS WITH EXPLANATIONS")
    push!(lines, "-" ^ 78)
    push!(lines, _explain_results(results))
    push!(lines, "")
    push!(lines, "PART II — FULL TASK LAUNCH LOGS")
    push!(lines, "-" ^ 78)
    append!(lines, logs)
    return join(lines, "\n")
end

function _explain_results(results)
    lines = String[]
    push!(lines, "Experiment: $(get(results, "experiment_name", "unknown"))")
    push!(lines, "Scenario:   $(get(results, "scenario_name", "default"))")
    push!(lines, "Language:   $(get(results, "language", "unknown"))")
    push!(lines, "")
    push!(lines, "Key metrics:")
    for (k, v) in get(results, "metrics", Dict())
        push!(lines, "  $(rpad(k, 30)) = $v")
    end
    push!(lines, "")
    push!(lines, "RMT spectral analysis:")
    spec = get(results, "spectral", Dict())
    for (k, v) in spec
        if !(v isa Vector || v isa Dict)
            push!(lines, "  $(rpad(k, 30)) = $v")
        end
    end
    push!(lines, "")
    push!(lines, "Interpretation:")
    push!(lines, "  The Marchenko-Pastur (MP) law describes the bulk distribution of")
    push!(lines, "  eigenvalues of large random covariance matrices. When the largest")
    push!(lines, "  empirical eigenvalue exceeds the MP upper bound, this signals")
    push!(lines, "  structured (non-random) information — i.e. the model has 'detected'")
    push!(lines, "  a fact. Conversely, eigenvalues inside the MP bulk indicate")
    push!(lines, "  creative or hallucinated generation.")
    push!(lines, "")
    push!(lines, "  The N_crit threshold is the autoregressive token count beyond which")
    push!(lines, "  spectral collapse makes hallucination mathematically inevitable.")
    return join(lines, "\n")
end

function _build_markdown(results, logs)
    lines = String[]
    push!(lines, "# RMT-LLM Laboratory — Experiment Report (Julia)")
    push!(lines, "")
    push!(lines, "**Generated:** $(Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"))  ")
    push!(lines, "**Experiment:** `$(get(results, "experiment_name", ""))`  ")
    push!(lines, "**Scenario:** `$(get(results, "scenario_name", ""))`")
    push!(lines, "")
    push!(lines, "## Part I — Detailed Results with Explanations")
    push!(lines, "")
    push!(lines, "### Key Metrics")
    push!(lines, "")
    push!(lines, "| Metric | Value |")
    push!(lines, "|---|---|")
    for (k, v) in get(results, "metrics", Dict())
        push!(lines, "| $k | $v |")
    end
    push!(lines, "")
    push!(lines, "### Interpretation")
    push!(lines, "")
    push!(lines, "The Marchenko-Pastur (MP) law describes the bulk distribution of eigenvalues")
    push!(lines, "of large random covariance matrices. When the largest empirical eigenvalue")
    push!(lines, "exceeds the MP upper bound, this signals structured (non-random) information.")
    push!(lines, "")
    push!(lines, "## Part II — Full Task Launch Logs")
    push!(lines, "")
    push!(lines, "```")
    append!(lines, logs)
    push!(lines, "```")
    return join(lines, "\n")
end

function _build_html(results, logs)
    md = _build_markdown(results, logs)
    html = ["<!DOCTYPE html><html><head><meta charset='utf-8'>",
            "<title>RMT-LLM Lab Report (Julia)</title>",
            "<style>body{font-family:Arial;max-width:1100px;margin:2em auto;padding:0 1em}",
            "pre{background:#1a202c;color:#e2e8f0;padding:1em;border-radius:6px}</style>",
            "</head><body>"]
    for line in split(md, "\n")
        if startswith(line, "# ")
            push!(html, "<h1>$(line[3:end])</h1>")
        elseif startswith(line, "## ")
            push!(html, "<h2>$(line[4:end])</h2>")
        elseif startswith(line, "### ")
            push!(html, "<h3>$(line[5:end])</h3>")
        elseif startswith(line, "| ")
            push!(html, "<code>$(line)</code><br>")
        elseif startswith(line, "**")
            push!(html, "<p>$(line)</p>")
        elseif !isempty(line)
            push!(html, "<p>$(line)</p>")
        end
    end
    push!(html, "</body></html>")
    return join(html, "\n")
end

function _build_xml(results, logs)
    lines = ["<?xml version='1.0' encoding='UTF-8'?>", "<rmt_llm_report>",
             "  <generated>$(Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"))</generated>",
             "  <results>"]
    for (k, v) in _flatten(results)
        push!(lines, "    <$(k)>$(string(v))</$(k)>")
    end
    push!(lines, "  </results>")
    push!(lines, "  <logs>")
    for line in logs
        push!(lines, "    <log>$(string(line))</log>")
    end
    push!(lines, "  </logs>")
    push!(lines, "</rmt_llm_report>")
    return join(lines, "\n")
end

function _build_latex(results, logs)
    lines = [raw"\documentclass[11pt]{article}",
             raw"\usepackage[utf8]{inputenc}",
             raw"\title{RMT-LLM Laboratory Report (Julia)}",
             raw"\author{Iskhak Hamzatovich Isaev}",
             raw"\date{\today}",
             raw"\begin{document}",
             raw"\maketitle",
             raw"\section{Detailed Results}",
             _explain_results(results),
             raw"\section{Full Task Launch Logs}",
             raw"\begin{verbatim}"]
    append!(lines, logs)
    push!(lines, raw"\end{verbatim}")
    push!(lines, raw"\end{document}")
    return join(lines, "\n")
end

function _flatten(d::Dict, parent="", sep=".")
    out = Dict{String, Any}()
    for (k, v) in d
        key = isempty(parent) ? string(k) : "$(parent)$(sep)$(k)"
        if v isa Dict
            merge!(out, _flatten(v, key, sep))
        elseif v isa Vector
            out[key] = JSON.json(v)[1:min(500, end)]
        else
            out[key] = v
        end
    end
    return out
end

function _write_text(content::String, out_dir::String, name::String)
    path = joinpath(out_dir, name)
    write(path, content)
    return path
end

function _write_csv(results, logs, out_dir, name)
    path = joinpath(out_dir, "$(name).csv")
    open(path, "w") do f
        println(f, "section,key,value")
        for (k, v) in _flatten(results)
            println(f, "results,$(k),$(string(v))")
        end
        for (i, line) in enumerate(logs)
            println(f, "log,line_$(@sprintf("%05d", i-1)),$(string(line))")
        end
    end
    return path
end

function _write_json(results, logs, out_dir, name)
    path = joinpath(out_dir, "$(name).json")
    open(path, "w") do f
        JSON.print(f, Dict("generated_at" => Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
                            "results" => results, "logs" => logs), 2)
    end
    return path
end

function _write_yaml(results, logs, out_dir, name)
    path = joinpath(out_dir, "$(name).yaml")
    # Simple YAML writer (no external deps)
    open(path, "w") do f
        println(f, "generated_at: $(Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"))")
        println(f, "results:")
        for (k, v) in _flatten(results)
            println(f, "  $k: $(string(v))")
        end
        println(f, "logs:")
        for line in logs
            println(f, "  - $(repr(line))")
        end
    end
    return path
end

function _write_pdf(results, logs, out_dir, name)
    # Use Plots to render text into a PDF page
    using Plots
    text_content = _build_text(results, logs)
    # Chunk to avoid plot text limit
    path = joinpath(out_dir, "$(name).pdf")
    plt = plot(1:1, 1:1, seriestype=:scatter, legend=false,
               axis=false, grid=false, framestyle=:none,
               annotations=(1, 1, text(text_content[1:min(2000, end)], 6, :left, :bottom)))
    savefig(plt, path)
    return path
end

function _write_parquet(results, logs, out_dir, name)
    using Arrow, Tables
    path = joinpath(out_dir, "$(name).parquet")
    rows = [(section="results", key=k, value=string(v)) for (k, v) in _flatten(results)]
    for (i, line) in enumerate(logs)
        push!(rows, (section="log", key="line_$(@sprintf("%05d", i-1))", value=string(line)))
    end
    Arrow.write(path, rows)
    return path
end

function _write_xlsx(results, logs, out_dir, name)
    using XLSX
    path = joinpath(out_dir, "$(name).xlsx")
    XLSX.openxlsx(path, mode="w") do xf
        sheet = xf[1]
        XLSX.rename!(sheet, "Results")
        sheet["A1"] = "Key"; sheet["B1"] = "Value"
        row = 2
        for (k, v) in _flatten(results)
            sheet["A$row"] = string(k); sheet["B$row"] = string(v)
            row += 1
        end
    end
    return path
end

function _write_sqlite(results, logs, out_dir, name)
    using SQLite
    path = joinpath(out_dir, "$(name).sqlite")
    isfile(path) && rm(path)
    db = SQLite.DB(path)
    DBInterface.execute(db, "CREATE TABLE IF NOT EXISTS results (key TEXT PRIMARY KEY, value TEXT)")
    DBInterface.execute(db, "CREATE TABLE IF NOT EXISTS logs (line_no INTEGER PRIMARY KEY, content TEXT)")
    for (k, v) in _flatten(results)
        DBInterface.execute(db, "INSERT OR REPLACE INTO results VALUES (?, ?)", (string(k), string(v)))
    end
    for (i, line) in enumerate(logs)
        DBInterface.execute(db, "INSERT OR REPLACE INTO logs VALUES (?, ?)", (i-1, string(line)))
    end
    DBInterface.execute(db, "INSERT OR REPLACE INTO results VALUES ('generated_at', ?)",
                        (Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),))
    DBInterface.close(db)
    return path
end
