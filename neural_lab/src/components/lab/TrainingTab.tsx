"use client";

import { useEffect, useState } from "react";
import {
  Line, LineChart, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import type { MetricsResponse } from "@/lib/lab-types";

export function TrainingTab() {
  const [data, setData] = useState<MetricsResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/metrics")
      .then((r) => r.json())
      .then((d: MetricsResponse) => (d.ok ? setData(d) : setErr(d.error ?? "ошибка")))
      .catch((e) => setErr(String(e)));
  }, []);

  if (err) {
    return <div className="rounded-lg border border-red-900 bg-red-950/30 p-4 text-sm text-red-300">Ошибка: {err}</div>;
  }
  if (!data?.summary || !data.curves) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-24 w-full bg-zinc-800" />
        <Skeleton className="h-72 w-full bg-zinc-800" />
      </div>
    );
  }

  const s = data.summary;
  const lossReduction = (1 - s.final_val_loss / s.baseline_val_loss) * 100;
  const pplBase = Math.exp(s.baseline_val_loss);
  const pplGain = pplBase / s.final_val_perplexity;

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 lg:grid-cols-6">
        <MetricCard label="Параметров" value={`${(s.model_params / 1000).toFixed(0)}K`} sub={`${s.config.layers} слоя · hidden ${s.config.hidden}`} />
        <MetricCard label="Эпох обучения" value={String(s.epochs)} sub={`${(s.train_tokens / 1000).toFixed(1)}K токенов корпуса`} />
        <MetricCard label="Val loss" value={`${s.baseline_val_loss.toFixed(2)} → ${s.final_val_loss.toFixed(2)}`} sub={`−${lossReduction.toFixed(1)}%`} accent />
        <MetricCard label="Перплексия" value={`${pplBase.toFixed(0)} → ${s.final_val_perplexity.toFixed(1)}`} sub={`×${pplGain.toFixed(0)} лучше`} accent />
        <MetricCard label="Match rate" value={`${(s.baseline_val_match * 100).toFixed(1)}% → ${(s.final_val_match * 100).toFixed(1)}%`} sub={`+${((s.final_val_match - s.baseline_val_match) * 100).toFixed(1)} п.п.`} accent />
        <MetricCard label="Токенизатор" value={`BPE-${s.bpe_vocab}`} sub={`${s.bpe_merges} слияний · seq ${s.seq_len}`} />
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <ChartCard title="Кросс-энтропия (nats/токен)">
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={data.curves}>
              <CartesianGrid stroke="#27272a" strokeDasharray="3 3" />
              <XAxis dataKey="epoch" stroke="#71717a" fontSize={11} tickLine={false} />
              <YAxis stroke="#71717a" fontSize={11} tickLine={false} domain={[0, 7]} />
              <Tooltip contentStyle={tooltipStyle} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Line type="monotone" dataKey="train_loss" name="train" stroke="#f59e0b" dot={false} strokeWidth={2} />
              <Line type="monotone" dataKey="val_loss" name="val" stroke="#34d399" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Перплексия на валидации (ниже = лучше)">
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={data.curves}>
              <CartesianGrid stroke="#27272a" strokeDasharray="3 3" />
              <XAxis dataKey="epoch" stroke="#71717a" fontSize={11} tickLine={false} />
              <YAxis stroke="#71717a" fontSize={11} tickLine={false} domain={[0, 120]} />
              <Tooltip contentStyle={tooltipStyle} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Line type="monotone" dataKey="val_ppl" name="val perplexity" stroke="#34d399" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Точность следующего токена, % (match rate)">
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={data.curves}>
              <CartesianGrid stroke="#27272a" strokeDasharray="3 3" />
              <XAxis dataKey="epoch" stroke="#71717a" fontSize={11} tickLine={false} />
              <YAxis stroke="#71717a" fontSize={11} tickLine={false} domain={[0, 85]} unit="%" />
              <Tooltip contentStyle={tooltipStyle} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Line type="monotone" dataKey="train_match" name="train" stroke="#f59e0b" dot={false} strokeWidth={2} />
              <Line type="monotone" dataKey="val_match" name="val" stroke="#34d399" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Диагностика: норма градиента и learning rate">
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={data.curves}>
              <CartesianGrid stroke="#27272a" strokeDasharray="3 3" />
              <XAxis dataKey="epoch" stroke="#71717a" fontSize={11} tickLine={false} />
              <YAxis yAxisId="l" stroke="#71717a" fontSize={11} tickLine={false} domain={[0, 4]} />
              <YAxis yAxisId="r" orientation="right" stroke="#71717a" fontSize={11} tickLine={false} domain={[0, 0.0014]} tickFormatter={(v: number) => v.toFixed(2)} />
              <Tooltip contentStyle={tooltipStyle} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Line yAxisId="l" type="monotone" dataKey="grad_norm" name="grad norm" stroke="#a78bfa" dot={false} strokeWidth={2} />
              <Line yAxisId="r" type="monotone" dataKey="lr" name="learning rate" stroke="#38bdf8" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>

      <Card className="border-zinc-800 bg-zinc-900/60">
        <CardHeader className="pb-2">
          <CardTitle className="text-base text-zinc-100">Рецепт обучения</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-wrap gap-2">
          {[
            "RoPE (позиции)", `GQA ${s.config.heads}/${s.config.kv_heads} (Q/KV головы)`,
            "Weight tying", `Label smoothing ${s.config.label_smoothing}`,
            `Dropout ${s.config.dropout}`, "AdamW + cosine LR (1.2e-3 → 6e-5)",
            "Grad clip 1.0", "Batch 8×2 (grad accum)", "Pure NumPy autodiff (ADR-001)",
          ].map((t) => (
            <Badge key={t} variant="outline" className="border-zinc-700 bg-zinc-950 text-zinc-300">
              {t}
            </Badge>
          ))}
        </CardContent>
      </Card>
    </div>
  );
}

const tooltipStyle = {
  backgroundColor: "#18181b",
  border: "1px solid #3f3f46",
  borderRadius: 8,
  fontSize: 12,
  color: "#e4e4e7",
};

function MetricCard({ label, value, sub, accent }: { label: string; value: string; sub?: string; accent?: boolean }) {
  return (
    <div className={`rounded-xl border p-4 ${accent ? "border-emerald-800 bg-emerald-950/20" : "border-zinc-800 bg-zinc-900/60"}`}>
      <div className="text-[10px] uppercase tracking-wider text-zinc-500">{label}</div>
      <div className={`mt-1.5 font-mono text-sm font-semibold ${accent ? "text-emerald-300" : "text-zinc-100"}`}>{value}</div>
      {sub && <div className="mt-1 text-[11px] text-zinc-500">{sub}</div>}
    </div>
  );
}

function ChartCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <Card className="border-zinc-800 bg-zinc-900/60">
      <CardHeader className="pb-0">
        <CardTitle className="text-sm font-medium text-zinc-300">{title}</CardTitle>
      </CardHeader>
      <CardContent className="pt-2">{children}</CardContent>
    </Card>
  );
}
