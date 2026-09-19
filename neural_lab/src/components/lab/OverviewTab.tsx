"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Brain, Sigma, TrendingUp, FlaskConical, Server, ExternalLink,
} from "lucide-react";

export function OverviewTab({ onGoTab }: { onGoTab: (t: string) => void }) {
  return (
    <div className="space-y-6">
      <Card className="border-zinc-800 bg-gradient-to-br from-zinc-900/80 to-zinc-900/40">
        <CardHeader className="pb-3">
          <div className="flex items-center gap-2 text-emerald-400">
            <Brain className="h-5 w-5" />
            <CardTitle className="text-lg text-zinc-100">
              Что это за приложение
            </CardTitle>
          </div>
        </CardHeader>
        <CardContent className="space-y-4 text-sm leading-relaxed text-zinc-300">
          <p>
            Здесь <span className="text-emerald-300">живёт нейросеть TinyGPT v3 (Formula-Edition)</span> —
            трансформер на чистом NumPy из репозитория{" "}
            <a
              href="https://github.com/wild8highlander/rmt-llm-research"
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-0.5 text-emerald-400 underline decoration-emerald-800 underline-offset-2 hover:decoration-emerald-400"
            >
              wild8highlander/rmt-llm-research <ExternalLink className="h-3 w-3" />
            </a>
            , дообученная на корпусе из <b>277 точных вычислительных пар</b> формул
            теории случайных матриц (RMT): закон Марченко-Пастура, BBP-переход,
            распределение Трейси-Уидома, динамика Капуто, принцип Ландауэра и другие.
          </p>
          <p>
            Модель «из коробки» (RoPE, GQA, AdamW, косинусное расписание — всё уже
            внутри репозитория, ADR-001: NumPy-only) обучена с нуля за 40 эпох на CPU.
            Все метрики честные: история обучения, held-out бенчмарк и RMT-диагностика
            скрытых состояний считаются реальным кодом, а не выдумкой.
          </p>
          <div className="flex flex-wrap gap-2 pt-1">
            {["447K параметров", "BPE-512 токенизатор", "277 вычислительных пар", "40 эпох", "held-out бенчмарк", "4 открытых вопроса"].map((t) => (
              <Badge key={t} variant="outline" className="border-emerald-800 bg-emerald-950/30 text-emerald-300">
                {t}
              </Badge>
            ))}
          </div>
        </CardContent>
      </Card>

      <div className="grid gap-4 md:grid-cols-3">
        <ActionCard
          icon={<FlaskConical className="h-5 w-5 text-emerald-400" />}
          title="Нейросеть"
          text="Живой playground: выберите формулу RMT — модель допишет вычисление. Температура, seed, перплексия ответа."
          cta="Открыть playground"
          tab="playground"
          onGoTab={onGoTab}
        />
        <ActionCard
          icon={<TrendingUp className="h-5 w-5 text-emerald-400" />}
          title="Обучение и бенчмарк"
          text="Кривые loss/перплексии/match-rate по эпохам, прирост против baseline и честный held-out бенчмарк формул."
          cta="Смотреть метрики"
          tab="training"
          onGoTab={onGoTab}
        />
        <ActionCard
          icon={<Sigma className="h-5 w-5 text-emerald-400" />}
          title="Открытые вопросы"
          text="4 proposals из ROADMAP: free probability для attention, Капуто-SFT, BBP-детектор лжи, оценка Хатчинсона — с решениями."
          cta="Изучить решения"
          tab="questions"
          onGoTab={onGoTab}
        />
      </div>

      <Card className="border-zinc-800 bg-zinc-900/60">
        <CardHeader className="pb-2">
          <CardTitle className="text-base text-zinc-100">Как устроен конвейер</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-3 md:grid-cols-5">
            {[
              ["1. Корпус", "build_corpus.py извлекает docstrings + генерирует 277 пар «параметры → результат» вызовами настоящих функций rmt_llm"],
              ["2. Токенизация", "BPE-512 (256 слияний) из tiny_gpt_trainer.py: 29.0K train / 2.5K val токенов"],
              ["3. Обучение", "TinyGPT v3 (RoPE+GQA+weight tying), 447K параметров, AdamW + cosine LR, 40 эпох на CPU"],
              ["4. Оценка", "held-out бенчмарк свежих формул, прирост vs baseline, RMT-спектр слоёв"],
              ["5. Деплой", "веса .npz живут здесь; Next.js API порождает Python-инференс (88 мс / 24 токена)"],
            ].map(([t, d]) => (
              <div key={t} className="rounded-lg border border-zinc-800 bg-zinc-950/60 p-3">
                <div className="text-xs font-semibold text-emerald-400">{t}</div>
                <p className="mt-1.5 text-[11px] leading-relaxed text-zinc-400">{d}</p>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      <Card className="border-zinc-800 bg-zinc-900/60">
        <CardHeader className="pb-2">
          <div className="flex items-center gap-2">
            <Server className="h-4 w-4 text-emerald-400" />
            <CardTitle className="text-base text-zinc-100">
              Теория в двух словах (из монографии репозитория)
            </CardTitle>
          </div>
        </CardHeader>
        <CardContent className="text-xs leading-relaxed text-zinc-400">
          Монография <i>LLM_Analysis_Merged.pdf</i> связывает машинную арифметику и
          галлюцинации: ошибка округления IEEE 754 через жорданову клетку размера k
          усиливается как δλ ~ ε^(1/k); каузальная маска делает ложный токен
          необратимым (принцип Ландауэра: стирание из KV-кэша стоит k_B·T·ln2 на бит);
          RLHF добавляет дрейф μ_RLHF, и время до коллапса падает как
          ⟨T_crit⟩ ∝ μ_eff^(−1/β) — при β≈0.5 это квадратичное сжатие честной цепочки
          рассуждений («Ловушка Полезности»). Обученная здесь модель воспроизводит
          эти формулы и их числовое поведение.
        </CardContent>
      </Card>
    </div>
  );
}

function ActionCard({
  icon, title, text, cta, tab, onGoTab,
}: {
  icon: React.ReactNode;
  title: string;
  text: string;
  cta: string;
  tab: string;
  onGoTab: (t: string) => void;
}) {
  return (
    <div className="flex flex-col rounded-xl border border-zinc-800 bg-zinc-900/60 p-5">
      <div className="flex items-center gap-2">
        {icon}
        <h3 className="font-semibold text-zinc-100">{title}</h3>
      </div>
      <p className="mt-2 flex-1 text-xs leading-relaxed text-zinc-400">{text}</p>
      <button
        onClick={() => onGoTab(tab)}
        className="mt-4 w-full rounded-lg border border-emerald-800 bg-emerald-950/30 py-2 text-xs font-medium text-emerald-300 transition-colors hover:bg-emerald-900/40"
      >
        {cta}
      </button>
    </div>
  );
}
