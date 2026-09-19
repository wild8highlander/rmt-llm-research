"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import type { EvaluationResponse } from "@/lib/lab-types";

export function BenchmarkTab() {
  const [data, setData] = useState<EvaluationResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/evaluation")
      .then((r) => r.json())
      .then((d: EvaluationResponse) => (d.ok ? setData(d) : setErr(d.error ?? "ошибка")))
      .catch((e) => setErr(String(e)));
  }, []);

  if (err) {
    return <div className="rounded-lg border border-red-900 bg-red-950/30 p-4 text-sm text-red-300">Ошибка: {err}</div>;
  }
  if (!data?.benchmark || !data.improvement) {
    return <Skeleton className="h-72 w-full bg-zinc-800" />;
  }

  const b = data.benchmark;
  const imp = data.improvement;

  return (
    <div className="space-y-6">
      <div className="grid gap-4 md:grid-cols-3">
        <ScoreCard
          title="format_score"
          value={`${(b.format_score * 100).toFixed(1)}%`}
          desc="Модель воспроизводит структуру ответа «=> поля = числа» на свежих формулах, которых не было в корпусе"
          accent
        />
        <ScoreCard
          title="numeric_score"
          value={`${(b.numeric_score * 100).toFixed(1)}%`}
          desc="Точное попадание чисел (±15%) на невиданных параметрах. Честный нуль: 447K параметров не экстраполируют арифметику — микромодель иллюстрирует теорию галлюцинаций самого репозитория"
        />
        <ScoreCard
          title="teacher_forced"
          value={`${(b.teacher_forced_score * 100).toFixed(1)}%`}
          desc="Точность следующего токена на эталонных продолжениях (для сравнения: случайное угадывание ≈ 0.2%)"
          accent
        />
      </div>

      <Card className="border-zinc-800 bg-zinc-900/60">
        <CardHeader className="pb-2">
          <CardTitle className="text-base text-zinc-100">
            Прирост от обучения (baseline → 40 эпох)
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <Delta label="Val loss" before={imp.baseline_val_loss.toFixed(2)} after={imp.final_val_loss.toFixed(2)} delta={`−${imp.loss_reduction_pct.toFixed(1)}%`} />
            <Delta label="Перплексия" before={imp.baseline_val_perplexity.toFixed(0)} after={imp.final_val_perplexity.toFixed(1)} delta={`−${imp.perplexity_reduction_pct.toFixed(1)}%`} />
            <Delta label="Match rate" before={`${(imp.baseline_match_rate * 100).toFixed(1)}%`} after={`${(imp.final_match_rate * 100).toFixed(1)}%`} delta={`+${imp.match_rate_gain_pp.toFixed(1)} п.п.`} />
            <Delta label="Эпох" before="0" after={String(imp.epochs_run)} delta="чипы по 8 мин" />
          </div>
        </CardContent>
      </Card>

      <Card className="border-zinc-800 bg-zinc-900/60">
        <CardHeader className="pb-2">
          <CardTitle className="text-base text-zinc-100">
            Примеры из held-out бенчмарка ({b.n_tasks} свежих задач)
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {b.samples.map((s, i) => (
            <div key={i} className="rounded-lg border border-zinc-800 bg-zinc-950/70 p-3">
              <div className="flex flex-wrap items-center gap-2">
                <Badge
                  className={
                    s.numeric_ok
                      ? "border-emerald-700 bg-emerald-950/50 text-emerald-400"
                      : "border-amber-700 bg-amber-950/40 text-amber-400"
                  }
                  variant="outline"
                >
                  {s.numeric_ok ? "числа OK" : "промах по числам"}
                </Badge>
                <span className="font-mono text-xs text-zinc-200">{s.prompt}</span>
              </div>
              <div className="mt-2 space-y-1 font-mono text-xs leading-relaxed">
                <div className="text-zinc-500">
                  <span className="mr-1 rounded bg-zinc-800 px-1 text-[10px] uppercase">эталон</span>
                  {s.ground_truth}
                </div>
                <div className="text-zinc-400">
                  <span className="mr-1 rounded bg-zinc-800 px-1 text-[10px] uppercase">генерация</span>
                  {s.generated.slice(0, 120)}
                </div>
              </div>
            </div>
          ))}
          <p className="text-xs leading-relaxed text-zinc-500">
            RMT-диагностика скрытых состояний: на этой малой модели BBP-спайк
            (выход λ_max за MP-границу) не наблюдается ни до, ни после обучения —
            спектр остаётся в объёмном режиме. Спайк ожидается на больших LLM,
            как в probe_real_gpt2.py оригинального репозитория.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}

function ScoreCard({ title, value, desc, accent }: { title: string; value: string; desc: string; accent?: boolean }) {
  return (
    <div className={`rounded-xl border p-5 ${accent ? "border-emerald-800 bg-emerald-950/20" : "border-zinc-800 bg-zinc-900/60"}`}>
      <div className="flex items-baseline justify-between">
        <span className="font-mono text-xs text-zinc-500">{title}</span>
        <span className={`font-mono text-2xl font-bold ${accent ? "text-emerald-300" : "text-amber-300"}`}>{value}</span>
      </div>
      <p className="mt-3 text-xs leading-relaxed text-zinc-400">{desc}</p>
    </div>
  );
}

function Delta({ label, before, after, delta }: { label: string; before: string; after: string; delta: string }) {
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-950/60 p-3">
      <div className="text-[10px] uppercase tracking-wider text-zinc-500">{label}</div>
      <div className="mt-1 font-mono text-sm text-zinc-200">
        {before} <span className="text-zinc-600">→</span>{" "}
        <span className="text-emerald-300">{after}</span>
      </div>
      <div className="mt-1 font-mono text-xs text-emerald-400">{delta}</div>
    </div>
  );
}
