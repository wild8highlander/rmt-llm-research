"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import type { OpenQuestionsResponse } from "@/lib/lab-types";

interface QMeta {
  id: string;
  title: string;
  origin: string;
  theory: string;
}

const QUESTIONS: QMeta[] = [
  {
    id: "Q1_free_probability_attention",
    title: "Свободная вероятность для attention-матриц",
    origin: "💡 Proposal — ROADMAP.md",
    theory:
      "Применить R-трансформ (свободные кумулянты) к ковариациям attention-голов: свободная свёртка может дать более острую оценку N_crit, чем критерий Марченко-Пастура.",
  },
  {
    id: "Q2_caputo_sft",
    title: "Дробный оператор Капуто для SFT-динамики",
    origin: "💡 Proposal — ROADMAP.md",
    theory:
      "Расширить Капуто-описание с RLHF на supervised fine-tuning: дрейф меняет знак (μ_eff = μ₀ − μ_SFT вместо μ₀ + μ_RLHF), ядро памяти то же. Предсказание: время до коллапса растёт, а не падает.",
  },
  {
    id: "Q3_bbp_lying_detector",
    title: "BBP-переход как детектор лжи",
    origin: "💡 Proposal — ROADMAP.md",
    theory:
      "Эмпирический тест: пересекает ли λ_max ковариации скрытых состояний MP-границу ровно в момент, когда модель переходит от припоминания к фабрикации?",
  },
  {
    id: "Q4_hutchinson",
    title: "Стохастическая оценка следа Хатчинсона",
    origin: "💡 Proposal — ROADMAP.md",
    theory:
      "Для больших матриц активаций, где полная эвендекомпозиция недоступна: след и λ_max через случайные проекции без построения ковариации.",
  },
];

interface QResult {
  status?: string;
  conclusion?: string;
  [k: string]: unknown;
}

export function QuestionsTab() {
  const [results, setResults] = useState<Record<string, QResult> | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/open-questions")
      .then((r) => r.json())
      .then((d: OpenQuestionsResponse) =>
        d.ok && d.results ? setResults(d.results as Record<string, QResult>) : setErr(d.error ?? "ошибка")
      )
      .catch((e) => setErr(String(e)));
  }, []);

  if (err) {
    return <div className="rounded-lg border border-red-900 bg-red-950/30 p-4 text-sm text-red-300">Ошибка: {err}</div>;
  }
  if (!results) return <Skeleton className="h-72 w-full bg-zinc-800" />;

  return (
    <div className="grid gap-5 lg:grid-cols-2">
      {QUESTIONS.map((q) => {
        const r = results[q.id] ?? {};
        const solved = (r.status ?? "").startsWith("SOLVED");
        return (
          <Card key={q.id} className="border-zinc-800 bg-zinc-900/60">
            <CardHeader className="pb-2">
              <div className="flex flex-wrap items-start justify-between gap-2">
                <CardTitle className="text-base leading-snug text-zinc-100">{q.title}</CardTitle>
                <Badge
                  variant="outline"
                  className={
                    solved
                      ? "shrink-0 border-emerald-700 bg-emerald-950/40 text-emerald-400"
                      : "shrink-0 border-amber-700 bg-amber-950/40 text-amber-400"
                  }
                >
                  {r.status ?? "—"}
                </Badge>
              </div>
              <div className="text-[11px] text-zinc-500">{q.origin}</div>
            </CardHeader>
            <CardContent className="space-y-3">
              <p className="text-xs leading-relaxed text-zinc-400">{q.theory}</p>
              <div className="rounded-lg border border-zinc-800 bg-zinc-950/70 p-3">
                <div className="text-[10px] uppercase tracking-wider text-zinc-600">Постановка решения</div>
                <p className="mt-1 text-xs leading-relaxed text-zinc-300">{r.conclusion ?? "—"}</p>
              </div>
              <QDetail id={q.id} r={r} />
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}

function QDetail({ id, r }: { id: string; r: QResult }) {
  const rows: Array<[string, string]> = [];
  if (id === "Q3_bbp_lying_detector") {
    rows.push(
      ["λ_max (знакомые формулы)", fmt(r.known_lambda_max_mean)],
      ["λ_max (фабрикация)", fmt(r.lying_lambda_max_mean)],
      ["Разрыв λ_max", `${fmt(r.lambda_gap_pct)}%`],
      ["AUC разделения", fmt(r.pairwise_separation_auc)]
    );
  }
  if (id === "Q2_caputo_sft") {
    const emp = (r.empirical ?? {}) as Record<string, unknown>;
    rows.push(
      ["μ_SFT (из кривой лосса)", fmt(emp.mu_sft_estimated_from_loss_curve, 5)],
      ["β (память)", fmt(emp.beta_assumed)],
      ["⟨T⟩ предсказание (у.е.)", fmt(emp.T_pred_sft_time_units, 0)],
      ["⟨T⟩ контраст RLHF μ=0.1", fmt(emp.T_contrast_rlhf_mu0_1, 0)]
    );
  }
  if (id === "Q4_hutchinson") {
    const sc = (r.scaling ?? {}) as Record<string, unknown>;
    rows.push(
      ["Матрица", String(sc.matrix ?? "—")],
      ["Полный спектральный путь", `${fmt(sc.full_spectral_path_s)} c`],
      ["Путь Хатчинсона", `${fmt(sc.hutchinson_power_path_s)} c`],
      ["Ошибка следа / λ_max", `${pct(sc.trace_rel_error)} / ${pct(sc.lambda_max_rel_error)}`]
    );
  }
  if (id === "Q1_free_probability_attention") {
    const layers = (r.layers ?? []) as Array<Record<string, number>>;
    rows.push(
      ["Слоёв проанализировано", String(layers.length)],
      ["Свободных кумулянтов на слой", "κ₁, κ₂ (R-трансформ, 6 моментов)"],
      ["MP-ratio / N_crit_free", layers
        .slice(0, 4)
        .map((l) => `L${l.layer}: ${(l.lambda_max / l.mp_upper).toFixed(2)} / ${l.ncrit_free.toFixed(1)}`)
        .join(", ")]
    );
  }
  if (!rows.length) return null;
  return (
    <div className="overflow-hidden rounded-lg border border-zinc-800">
      <table className="w-full text-xs">
        <tbody>
          {rows.map(([k, v], i) => (
            <tr key={i} className={i % 2 ? "bg-zinc-950/60" : "bg-zinc-900/40"}>
              <td className="px-3 py-1.5 text-zinc-500">{k}</td>
              <td className="px-3 py-1.5 text-right font-mono text-zinc-200">{v}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function fmt(v: unknown, digits = 3): string {
  if (typeof v !== "number" || !isFinite(v)) return "—";
  if (digits === 0) return v.toFixed(0);
  return v.toFixed(digits);
}

function pct(v: unknown): string {
  if (typeof v !== "number" || !isFinite(v)) return "—";
  return `${(v * 100).toFixed(2)}%`;
}
