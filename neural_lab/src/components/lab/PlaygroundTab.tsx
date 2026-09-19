"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Loader2, Play, RotateCcw } from "lucide-react";
import type { InferResponse } from "@/lib/lab-types";

const PRESETS = [
  { label: "Границы MP-закона", prompt: "mp_bounds(q=0.5, sigma2=1.0)" },
  { label: "BBP-переход", prompt: "bbp_lambda_max(theta=1.2, q=0.5, sigma2=1.0)" },
  { label: "Время коллапса Капуто", prompt: "caputo_mean_collapse_time(mu_eff=0.1, beta=0.5, c=1.0)" },
  { label: "Среднее Трейси-Уидома", prompt: "tracy_widom_mean()" },
  { label: "Правило N_crit", prompt: "caputo_n_crit(theta_b=7.07, gamma_1=14.134725)" },
  { label: "Ландауэр", prompt: "landauer_cost(n_bits=100, T=300)" },
  { label: "Свободная вероятность", prompt: "mp_r_transform(z=0.5+0.1i, q=0.6)" },
  { label: "Китинг-Снайт", prompt: "ks_n_crit_correction(n=256)" },
];

export function PlaygroundTab() {
  const [prompt, setPrompt] = useState(PRESETS[0].prompt);
  const [temperature, setTemperature] = useState(0.3);
  const [maxTokens, setMaxTokens] = useState(48);
  const [seed, setSeed] = useState(42);
  const [result, setResult] = useState<InferResponse | null>(null);
  const [loading, setLoading] = useState(false);

  async function run() {
    setLoading(true);
    setResult(null);
    try {
      const res = await fetch("/api/infer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          max_new_tokens: maxTokens,
          temperature,
          seed,
        }),
      });
      const data = (await res.json()) as InferResponse;
      setResult(data);
    } catch {
      setResult({ ok: false, error: "Сеть недоступна" });
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="grid gap-6 lg:grid-cols-5">
      <Card className="lg:col-span-2 border-zinc-800 bg-zinc-900/60">
        <CardHeader>
          <CardTitle className="text-lg text-zinc-100">
            Управление генерацией
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-5">
          <div>
            <div className="mb-2 text-xs font-medium uppercase tracking-wide text-zinc-500">
              Промпт (формула из репозитория)
            </div>
            <Input
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              className="border-zinc-700 bg-zinc-950 font-mono text-sm text-emerald-300"
              placeholder="имя_формулы(параметры)"
            />
            <div className="mt-3 flex flex-wrap gap-1.5">
              {PRESETS.map((p) => (
                <button
                  key={p.prompt}
                  onClick={() => setPrompt(p.prompt)}
                  className="rounded-md border border-zinc-700 bg-zinc-800/70 px-2 py-1 text-[11px] text-zinc-300 transition-colors hover:border-emerald-600 hover:text-emerald-300"
                >
                  {p.label}
                </button>
              ))}
            </div>
          </div>

          <div>
            <div className="mb-2 flex items-center justify-between text-xs">
              <span className="font-medium uppercase tracking-wide text-zinc-500">
                Температура
              </span>
              <span className="font-mono text-emerald-400">{temperature.toFixed(2)}</span>
            </div>
            <Slider
              value={[temperature]}
              onValueChange={(v) => setTemperature(v[0])}
              min={0}
              max={1.5}
              step={0.05}
              className="py-2"
            />
            <p className="mt-1 text-[11px] text-zinc-500">
              0 — жадная генерация (детерминированная), выше — креативнее и рискованнее
            </p>
          </div>

          <div>
            <div className="mb-2 flex items-center justify-between text-xs">
              <span className="font-medium uppercase tracking-wide text-zinc-500">
                Токенов сгенерировать
              </span>
              <span className="font-mono text-emerald-400">{maxTokens}</span>
            </div>
            <Slider
              value={[maxTokens]}
              onValueChange={(v) => setMaxTokens(v[0])}
              min={8}
              max={80}
              step={4}
              className="py-2"
            />
          </div>

          <div>
            <div className="mb-2 text-xs font-medium uppercase tracking-wide text-zinc-500">
              Seed
            </div>
            <div className="flex items-center gap-2">
              <Input
                type="number"
                value={seed}
                onChange={(e) => setSeed(Number(e.target.value) || 0)}
                className="w-28 border-zinc-700 bg-zinc-950 font-mono text-sm text-zinc-200"
              />
              <Button
                variant="outline"
                size="sm"
                onClick={() => setSeed(Math.floor(Math.random() * 10000))}
                className="border-zinc-700 text-zinc-300 hover:text-emerald-300"
              >
                <RotateCcw className="h-3.5 w-3.5" /> Случайный
              </Button>
            </div>
          </div>

          <Button
            onClick={run}
            disabled={loading}
            className="w-full bg-emerald-600 text-white hover:bg-emerald-500"
          >
            {loading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Нейросеть думает...
              </>
            ) : (
              <>
                <Play className="mr-2 h-4 w-4" /> Сгенерировать продолжение
              </>
            )}
          </Button>
        </CardContent>
      </Card>

      <div className="space-y-4 lg:col-span-3">
        <Card className="border-zinc-800 bg-zinc-900/60">
          <CardHeader className="pb-2">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <CardTitle className="text-lg text-zinc-100">Ответ модели</CardTitle>
              {result?.ok && result.model_info && (
                <Badge
                  variant="outline"
                  className="border-emerald-700 bg-emerald-950/40 text-emerald-400"
                >
                  {result.model_info.name} · {(result.model_info.params / 1000).toFixed(0)}K параметров
                </Badge>
              )}
            </div>
          </CardHeader>
          <CardContent>
            {!result && !loading && (
              <div className="flex h-48 items-center justify-center rounded-lg border border-dashed border-zinc-700 text-sm text-zinc-500">
                Нажмите «Сгенерировать продолжение» — модель допишет формулу
              </div>
            )}
            {loading && (
              <div className="flex h-48 items-center justify-center rounded-lg border border-dashed border-zinc-700 text-sm text-zinc-500">
                <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Инференс через NumPy-ядро TinyGPT...
              </div>
            )}
            {result && !result.ok && (
              <div className="rounded-lg border border-red-900 bg-red-950/30 p-4 text-sm text-red-300">
                Ошибка инференса: {result.error}
              </div>
            )}
            {result?.ok && (
              <div className="space-y-4">
                <div className="rounded-lg border border-zinc-700 bg-zinc-950 p-4 font-mono text-sm leading-relaxed">
                  <span className="text-amber-300">{result.prompt}</span>
                  <span className="text-emerald-300">{result.generated}</span>
                </div>
                <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                  <Stat label="Токенов" value={String(result.tokens_generated)} />
                  <Stat label="Инференс" value={`${result.gen_ms} мс`} />
                  <Stat
                    label="Перплексия ответа"
                    value={
                      result.generated_perplexity
                        ? result.generated_perplexity.toFixed(1)
                        : "—"
                    }
                  />
                  <Stat label="Загрузка весов" value={`${result.load_ms} мс`} />
                </div>
                <p className="text-xs leading-relaxed text-zinc-500">
                  Модель обучена на 277 точных вычислительных парах формул RMT. На
                  знакомых промптах она воспроизводит структуру и значения; на
                  невиданных параметрах честно «галлюцинирует» — ровно тот феномен,
                  который исследует этот репозиторий. Низкая перплексия = модель
                  уверена; сравните уверенность на знакомых и новых формулах.
                </p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-950/60 p-3">
      <div className="text-[10px] uppercase tracking-wider text-zinc-500">{label}</div>
      <div className="mt-1 font-mono text-sm text-emerald-400">{value}</div>
    </div>
  );
}
