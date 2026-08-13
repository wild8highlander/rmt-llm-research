/*
 * Reports.java — 13-format report generator (Java, EN)
 */

package com.rmt.llm.lab.en;

import java.util.*;
import java.io.*;
import java.nio.file.*;
import java.time.*;
import java.time.format.*;

public class Reports {

    @SuppressWarnings("unchecked")
    public static Map<String, String> generateAll(Map<String, Object> results,
                                                   List<String> logs,
                                                   String outDir, String name) throws IOException {
        Files.createDirectories(Paths.get(outDir));
        Map<String, String> written = new LinkedHashMap<>();

        String text = buildText(results, logs);
        written.put("txt", writeFile(outDir, name + ".txt", text));
        written.put("md", writeFile(outDir, name + ".md", buildMarkdown(results, logs)));
        written.put("csv", writeCsv(results, logs, outDir, name));
        written.put("html", writeFile(outDir, name + ".html", buildHtml(results, logs)));
        written.put("json", writeJson(results, logs, outDir, name));
        written.put("yaml", writeYaml(results, logs, outDir, name));
        written.put("xml", writeFile(outDir, name + ".xml", buildXml(results, logs)));
        written.put("latex", writeFile(outDir, name + ".tex", buildLatex(results, logs)));
        written.put("pdf", writePdfPlaceholder(results, logs, outDir, name));
        written.put("docx", writeDocxPlaceholder(results, logs, outDir, name));
        written.put("parquet", writeParquetPlaceholder(results, logs, outDir, name));
        written.put("xlsx", writeXlsxPlaceholder(results, logs, outDir, name));
        written.put("sqlite", writeSqlite(results, logs, outDir, name));
        return written;
    }

    static String writeFile(String dir, String name, String content) throws IOException {
        String path = Paths.get(dir, name).toString();
        Files.write(Paths.get(path), content.getBytes("UTF-8"));
        return path;
    }

    static String buildText(Map<String, Object> r, List<String> logs) {
        StringBuilder sb = new StringBuilder();
        sb.append("=".repeat(78)).append("\n");
        sb.append("RMT-LLM Laboratory — Experiment Report (Java, TXT)\n");
        sb.append("Generated: ").append(LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME)).append("\n");
        sb.append("=".repeat(78)).append("\n\n");
        sb.append("PART I — DETAILED RESULTS WITH EXPLANATIONS\n");
        sb.append("-".repeat(78)).append("\n");
        sb.append(explainResults(r));
        sb.append("\n\nPART II — FULL TASK LAUNCH LOGS\n");
        sb.append("-".repeat(78)).append("\n");
        for (String line : logs) sb.append(line).append("\n");
        return sb.toString();
    }

    @SuppressWarnings("unchecked")
    static String explainResults(Map<String, Object> r) {
        StringBuilder sb = new StringBuilder();
        sb.append("Experiment: ").append(r.getOrDefault("experiment_name", "unknown")).append("\n");
        sb.append("Scenario:   ").append(r.getOrDefault("scenario_name", "default")).append("\n");
        sb.append("Language:   ").append(r.getOrDefault("language", "unknown")).append("\n\n");
        sb.append("Key metrics:\n");
        Map<String, Object> metrics = (Map<String, Object>) r.getOrDefault("metrics", new HashMap<>());
        for (Map.Entry<String, Object> e : metrics.entrySet())
            sb.append(String.format("  %-30s = %s%n", e.getKey(), e.getValue()));
        sb.append("\nInterpretation:\n");
        sb.append("  The Marchenko-Pastur (MP) law describes the bulk distribution of\n");
        sb.append("  eigenvalues of large random covariance matrices. When the largest\n");
        sb.append("  empirical eigenvalue exceeds the MP upper bound, this signals\n");
        sb.append("  structured (non-random) information — i.e. the model has 'detected'\n");
        sb.append("  a fact. Conversely, eigenvalues inside the MP bulk indicate\n");
        sb.append("  creative or hallucinated generation.\n");
        return sb.toString();
    }

    static String buildMarkdown(Map<String, Object> r, List<String> logs) {
        StringBuilder sb = new StringBuilder();
        sb.append("# RMT-LLM Laboratory — Experiment Report (Java)\n\n");
        sb.append("**Generated:** ").append(LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME)).append("  \n");
        sb.append("**Experiment:** `").append(r.getOrDefault("experiment_name", "")).append("`  \n");
        sb.append("**Scenario:** `").append(r.getOrDefault("scenario_name", "")).append("`\n\n");
        sb.append("## Part I — Detailed Results with Explanations\n\n");
        sb.append("### Key Metrics\n\n| Metric | Value |\n|---|---|\n");
        Map<String, Object> metrics = (Map<String, Object>) r.getOrDefault("metrics", new HashMap<>());
        for (Map.Entry<String, Object> e : metrics.entrySet())
            sb.append("| ").append(e.getKey()).append(" | ").append(e.getValue()).append(" |\n");
        sb.append("\n### Interpretation\n\n");
        sb.append("The Marchenko-Pastur (MP) law describes the bulk distribution of eigenvalues of large random covariance matrices. ");
        sb.append("When the largest empirical eigenvalue exceeds the MP upper bound, this signals structured (non-random) information.\n\n");
        sb.append("## Part II — Full Task Launch Logs\n\n```\n");
        for (String line : logs) sb.append(line).append("\n");
        sb.append("```\n");
        return sb.toString();
    }

    static String buildHtml(Map<String, Object> r, List<String> logs) {
        return "<!DOCTYPE html><html><head><meta charset='utf-8'><title>RMT-LLM Lab (Java)</title>" +
               "<style>body{font-family:Arial;max-width:1100px;margin:2em auto;padding:0 1em}" +
               "pre{background:#1a202c;color:#e2e8f0;padding:1em;border-radius:6px}</style>" +
               "</head><body>" + buildMarkdown(r, logs).replace("\n", "<br>") + "</body></html>";
    }

    static String buildXml(Map<String, Object> r, List<String> logs) {
        StringBuilder sb = new StringBuilder();
        sb.append("<?xml version='1.0' encoding='UTF-8'?>\n<rmt_llm_report>\n");
        sb.append("  <generated>").append(LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME)).append("</generated>\n  <results>\n");
        for (Map.Entry<String, Object> e : flatten(r).entrySet())
            sb.append("    <").append(e.getKey()).append(">").append(e.getValue()).append("</").append(e.getKey()).append(">\n");
        sb.append("  </results>\n  <logs>\n");
        for (String line : logs)
            sb.append("    <log>").append(escapeXml(line)).append("</log>\n");
        sb.append("  </logs>\n</rmt_llm_report>\n");
        return sb.toString();
    }

    static String buildLatex(Map<String, Object> r, List<String> logs) {
        StringBuilder sb = new StringBuilder();
        sb.append("\\documentclass[11pt]{article}\n\\usepackage[utf8]{inputenc}\n");
        sb.append("\\title{RMT-LLM Laboratory Report (Java)}\n\\author{Iskhak Hamzatovich Isaev}\n");
        sb.append("\\date{\\today}\n\\begin{document}\n\\maketitle\n");
        sb.append("\\section{Detailed Results}\n");
        sb.append(explainResults(r).replace("_", "\\_"));
        sb.append("\\section{Full Task Launch Logs}\n\\begin{verbatim}\n");
        for (String line : logs) sb.append(line).append("\n");
        sb.append("\\end{verbatim}\n\\end{document}\n");
        return sb.toString();
    }

    static String writeCsv(Map<String, Object> r, List<String> logs, String outDir, String name) throws IOException {
        String path = Paths.get(outDir, name + ".csv").toString();
        try (PrintWriter pw = new PrintWriter(path, "UTF-8")) {
            pw.println("section,key,value");
            for (Map.Entry<String, Object> e : flatten(r).entrySet())
                pw.println("results," + e.getKey() + "," + e.getValue());
            for (int i = 0; i < logs.size(); i++)
                pw.println("log,line_" + String.format("%05d", i) + "," + logs.get(i));
        }
        return path;
    }

    static String writeJson(Map<String, Object> r, List<String> logs, String outDir, String name) throws IOException {
        String path = Paths.get(outDir, name + ".json").toString();
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("generated_at", LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME));
        payload.put("results", r);
        payload.put("logs", logs);
        Files.write(Paths.get(path), SimpleJson.stringify(payload, 2).getBytes("UTF-8"));
        return path;
    }

    static String writeYaml(Map<String, Object> r, List<String> logs, String outDir, String name) throws IOException {
        String path = Paths.get(outDir, name + ".yaml").toString();
        try (PrintWriter pw = new PrintWriter(path, "UTF-8")) {
            pw.println("generated_at: " + LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME));
            pw.println("results:");
            for (Map.Entry<String, Object> e : flatten(r).entrySet())
                pw.println("  " + e.getKey() + ": " + e.getValue());
            pw.println("logs:");
            for (String line : logs) pw.println("  - " + line);
        }
        return path;
    }

    static String writePdfPlaceholder(Map<String, Object> r, List<String> logs, String outDir, String name) throws IOException {
        // Java has no built-in PDF generation; write a placeholder text file
        String path = Paths.get(outDir, name + ".pdf.txt").toString();
        Files.write(Paths.get(path), ("[PDF placeholder — use Python implementation for full PDF]\n\n" + buildText(r, logs)).getBytes("UTF-8"));
        return path + " (placeholder)";
    }

    static String writeDocxPlaceholder(Map<String, Object> r, List<String> logs, String outDir, String name) throws IOException {
        String path = Paths.get(outDir, name + ".docx.txt").toString();
        Files.write(Paths.get(path), ("[DOCX placeholder — use Python implementation for full DOCX]\n\n" + buildText(r, logs)).getBytes("UTF-8"));
        return path + " (placeholder)";
    }

    static String writeParquetPlaceholder(Map<String, Object> r, List<String> logs, String outDir, String name) throws IOException {
        String path = Paths.get(outDir, name + ".parquet.csv").toString();
        return writeCsv(r, logs, outDir, name + ".parquet") + " (CSV fallback)";
    }

    static String writeXlsxPlaceholder(Map<String, Object> r, List<String> logs, String outDir, String name) throws IOException {
        String path = Paths.get(outDir, name + ".xlsx.csv").toString();
        return writeCsv(r, logs, outDir, name + ".xlsx") + " (CSV fallback)";
    }

    static String writeSqlite(Map<String, Object> r, List<String> logs, String outDir, String name) throws IOException {
        // Pure Java SQLite is not built-in; create a TSV file as fallback
        String path = Paths.get(outDir, name + ".sqlite.tsv").toString();
        try (PrintWriter pw = new PrintWriter(path, "UTF-8")) {
            pw.println("section\tkey\tvalue");
            for (Map.Entry<String, Object> e : flatten(r).entrySet())
                pw.println("results\t" + e.getKey() + "\t" + e.getValue());
            for (int i = 0; i < logs.size(); i++)
                pw.println("log\tline_" + i + "\t" + logs.get(i));
        }
        return path + " (TSV fallback — use Python for real SQLite)";
    }

    @SuppressWarnings("unchecked")
    static Map<String, Object> flatten(Map<String, Object> d) {
        return flatten(d, "");
    }

    @SuppressWarnings("unchecked")
    static Map<String, Object> flatten(Map<String, Object> d, String parent) {
        Map<String, Object> out = new LinkedHashMap<>();
        for (Map.Entry<String, Object> e : d.entrySet()) {
            String key = parent.isEmpty() ? e.getKey() : parent + "." + e.getKey();
            Object v = e.getValue();
            if (v instanceof Map) out.putAll(flatten((Map<String, Object>) v, key));
            else if (v instanceof List) out.put(key, SimpleJson.stringify(v, 0).substring(0, Math.min(500, SimpleJson.stringify(v, 0).length())));
            else out.put(key, v);
        }
        return out;
    }

    static String escapeXml(String s) {
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                .replace("\"", "&quot;");
    }
}
