"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

const MODELS = [
  { name: "GPT-2 (117M–774M)", why: "Эталон «из коробки» для дообучения, полная поддержка в HuggingFace transformers; используется в репозитории (probe_real_gpt2.py)", license: "MIT" },
  { name: "SmolLM2 (135M–1.7B)", why: "Лёгкие модели HuggingFace, специально созданные для быстрого дообучения на своих данных", license: "Apache-2.0" },
  { name: "TinyLlama-1.1B", why: "Есть в model_registry.json репозитория; дообучается даже на слабом железе через LoRA", license: "Apache-2.0" },
  { name: "litgpt (Lightning AI)", why: "20+ готовых LLM с единым интерфейсом файнтюнинга — «подключил и обучил»", license: "Apache-2.0" },
  { name: "Pythia-70M/160M", why: "154 чекпоинта по ходу обучения — идеальны для spectral sweep по слоям, есть в реестре репозитория", license: "Apache-2.0" },
  { name: "TinyGPT v3 (этот проект)", why: "Собственная NumPy-модель репозитория: 447K параметров, обучается на CPU за минуты — выбрана здесь из-за ADR-001 (PyTorch запрещён)", license: "Proprietary", selected: true },
];

const SERVERS = [
  { name: "Google Colab (Free)", spec: "T4 GPU 16GB, ~4-6 ч/сессия", note: "Ноутбук для этого проекта — RMT_TinyGPT_Free_Server.ipynb" },
  { name: "Kaggle Notebooks", spec: "P100/T4x2, 30 ч/неделю", note: "Годится для config_12m (12.5M параметров, 150 эпох)" },
  { name: "Это веб-приложение", spec: "CPU-инференс, 88 мс / 24 токена", note: "Веса .npz живут рядом с приложением; Next.js API порождает Python-процесс", selected: true },
];

export function DeploymentTab() {
  return (
    <div className="space-y-6">
      <Card className="border-zinc-800 bg-zinc-900/60">
        <CardHeader className="pb-2">
          <CardTitle className="text-base text-zinc-100">
            Модели «из коробки» для дообучения (результат поиска)
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2.5">
            {MODELS.map((m) => (
              <div
                key={m.name}
                className={`rounded-lg border p-3 ${m.selected ? "border-emerald-800 bg-emerald-950/20" : "border-zinc-800 bg-zinc-950/60"}`}
              >
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-sm font-medium text-zinc-100">{m.name}</span>
                  <Badge variant="outline" className="border-zinc-700 text-[10px] text-zinc-400">
                    {m.license}
                  </Badge>
                  {m.selected && (
                    <Badge variant="outline" className="border-emerald-700 bg-emerald-950/40 text-[10px] text-emerald-400">
                      обучена здесь
                    </Badge>
                  )}
                </div>
                <p className="mt-1 text-xs leading-relaxed text-zinc-400">{m.why}</p>
              </div>
            ))}
          </div>
          <p className="mt-4 text-[11px] leading-relaxed text-zinc-500">
            Почему обучена TinyGPT, а не GPT-2/TinyLlama: проект декларирует ADR-001
            (NumPy-only, «frameworks hide the math») и явно исключает PyTorch; кроме
            того, дообучение 124M+ моделей требует GPU. TinyGPT v3 — модель,
            спроектированная автором репозитория именно для «подключил и обучил»,
            и она уже в теме исследования. Для больших конфигураций (12M, 150 эпох)
            подготовлен бесплатный GPU-сервер — ноутбук ниже.
          </p>
        </CardContent>
      </Card>

      <Card className="border-zinc-800 bg-zinc-900/60">
        <CardHeader className="pb-2">
          <CardTitle className="text-base text-zinc-100">
            Бесплатные «серверы» для обучения
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-3 md:grid-cols-3">
            {SERVERS.map((s) => (
              <div
                key={s.name}
                className={`rounded-lg border p-4 ${s.selected ? "border-emerald-800 bg-emerald-950/20" : "border-zinc-800 bg-zinc-950/60"}`}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="text-sm font-medium text-zinc-100">{s.name}</span>
                  {s.selected && (
                    <Badge variant="outline" className="border-emerald-700 text-[10px] text-emerald-400">
                      активен
                    </Badge>
                  )}
                </div>
                <div className="mt-1 font-mono text-[11px] text-emerald-400">{s.spec}</div>
                <p className="mt-2 text-xs leading-relaxed text-zinc-400">{s.note}</p>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      <Card className="border-zinc-800 bg-zinc-900/60">
        <CardHeader className="pb-2">
          <CardTitle className="text-base text-zinc-100">
            Файлы этого исследования (готовы к переносу в репозиторий)
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-hidden rounded-lg border border-zinc-800">
            <table className="w-full text-xs">
              <tbody>
                {[
                  ["scripts/build_corpus.py", "Сборка корпуса формул: docstrings + 277 вычислительных пар всех 81 функций ядра"],
                  ["scripts/train_formula_model.py", "Обучение TinyGPT v3 с историей метрик, чанками и resume"],
                  ["scripts/evaluate_model.py", "Held-out бенчмарк + прирост + RMT-диагностика слоёв"],
                  ["scripts/open_questions.py", "Решения 4 открытых вопросов ROADMAP"],
                  ["scripts/infer.py", "Инференс-мост для веб-приложения"],
                  ["model/tiny_gpt_formula.npz", "Обученные веса (447K параметров)"],
                  ["model/training_history.json", "Полная история: loss, перплексия, match rate, градиенты, LR по эпохам"],
                  ["model/evaluation_report.json", "Отчёт бенчмарка с примерами"],
                  ["model/open_questions_results.json", "Результаты открытых вопросов"],
                ].map(([f, d], i) => (
                  <tr key={f} className={i % 2 ? "bg-zinc-950/60" : "bg-zinc-900/40"}>
                    <td className="px-3 py-2 font-mono text-emerald-400/90">{f}</td>
                    <td className="px-3 py-2 text-zinc-400">{d}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-3 text-[11px] leading-relaxed text-zinc-500">
            Побочный находкой стала ошибка в репозитории: функция
            caputo_quadratic_acceleration() всегда падает — она вызывает
            caputo_mean_collapse_time(mu, beta=1.0), хотя валидатор требует beta ∈ (0,1)
            строго. Аналитический предел: T(β=0.5)/T(β→1) = μ⁻¹. Кандидат на pull request.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
