# RMT-LLM Neural Lab

Веб-приложение (Next.js + TypeScript), где «живёт» обученная модель
из `research/tinygpt_formula/`: playground вызывает `scripts/infer.py`
через API, остальные вкладки показывают артефакты обучения.

## Вкладки

| Вкладка | Что показывает |
|---|---|
| Overview | сводка проекта: модель, корпус, цель ROADMAP |
| Playground | живой инференс: 8 пресетов формул, температура, seed, число токенов |
| Training | кривые лосса/перплексии/match_rate по реальной истории обучения |
| Benchmark | held-out счёты (format/numeric/teacher-forced) + примеры ответов |
| Questions | 4 открытых вопроса ROADMAP и их эмпирические решения |
| Deployment | модели для дообучения, бесплатные серверы, найденные баги репозитория |

## Запуск

```bash
cd neural_lab
npm install        # или bun install
npm run dev        # http://localhost:3000
```

Требования: Node.js 20+; для вкладки Playground — Python 3.10+ и NumPy
в системе (инференс вызывается как subprocess: `research/tinygpt_formula/scripts/infer.py`).
API-эндпоинты: `/api/infer`, `/api/metrics`, `/api/evaluation`,
`/api/open-questions`, `/api` (health — проверяет наличие файлов модели).
