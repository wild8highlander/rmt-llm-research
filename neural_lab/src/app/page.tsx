"use client";

import { useState } from "react";
import { OverviewTab } from "@/components/lab/OverviewTab";
import { PlaygroundTab } from "@/components/lab/PlaygroundTab";
import { TrainingTab } from "@/components/lab/TrainingTab";
import { BenchmarkTab } from "@/components/lab/BenchmarkTab";
import { QuestionsTab } from "@/components/lab/QuestionsTab";
import { DeploymentTab } from "@/components/lab/DeploymentTab";
import { Activity } from "lucide-react";

const TABS = [
  { id: "overview", label: "Обзор" },
  { id: "playground", label: "Нейросеть" },
  { id: "training", label: "Обучение" },
  { id: "benchmark", label: "Бенчмарк" },
  { id: "questions", label: "Открытые вопросы" },
  { id: "deploy", label: "Модель и сервер" },
];

export default function Home() {
  const [tab, setTab] = useState("overview");

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-200 flex flex-col">
      <header className="sticky top-0 z-20 border-b border-zinc-800 bg-zinc-950/90 backdrop-blur">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-3 px-4 py-3 sm:px-6">
          <div className="flex items-center gap-2.5">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg border border-emerald-800 bg-emerald-950/50">
              <Activity className="h-4.5 w-4.5 text-emerald-400" />
            </div>
            <div>
              <h1 className="text-sm font-semibold leading-tight text-zinc-50">
                RMT-LLM Neural Lab
              </h1>
              <p className="text-[11px] leading-tight text-zinc-500">
                TinyGPT v3 Formula-Edition · живая нейросеть репозитория rmt-llm-research
              </p>
            </div>
          </div>
          <nav className="flex flex-wrap gap-1" aria-label="Вкладки">
            {TABS.map((t) => (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                aria-current={tab === t.id ? "page" : undefined}
                className={`rounded-lg px-3 py-1.5 text-xs font-medium transition-colors ${
                  tab === t.id
                    ? "bg-emerald-950/60 text-emerald-300 ring-1 ring-emerald-800"
                    : "text-zinc-400 hover:bg-zinc-900 hover:text-zinc-200"
                }`}
              >
                {t.label}
              </button>
            ))}
          </nav>
        </div>
      </header>

      <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-6 sm:px-6">
        {tab === "overview" && <OverviewTab onGoTab={setTab} />}
        {tab === "playground" && <PlaygroundTab />}
        {tab === "training" && <TrainingTab />}
        {tab === "benchmark" && <BenchmarkTab />}
        {tab === "questions" && <QuestionsTab />}
        {tab === "deploy" && <DeploymentTab />}
      </main>

      <footer className="mt-auto border-t border-zinc-800 bg-zinc-950">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-2 px-4 py-4 text-[11px] text-zinc-500 sm:px-6">
          <span>
            Обучение и инференс — чистый NumPy (ADR-001). Модель: 447K параметров,
            RoPE + GQA, BPE-512, 40 эпох на CPU.
          </span>
          <a
            href="https://github.com/wild8highlander/rmt-llm-research"
            target="_blank"
            rel="noreferrer"
            className="text-emerald-500 hover:text-emerald-400"
          >
            github.com/wild8highlander/rmt-llm-research
          </a>
        </div>
      </footer>
    </div>
  );
}
