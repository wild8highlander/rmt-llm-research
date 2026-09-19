# RMT-LLM Neural Lab

A web application (Next.js + TypeScript) where the trained model from
`research/tinygpt_formula/` "lives": the playground calls `scripts/infer.py`
through the API, while the other tabs display the training artifacts.

## Tabs

| Tab | What it shows |
|---|---|
| Overview | project summary: model, corpus, ROADMAP goal |
| Playground | live inference: 8 formula presets, temperature, seed, token count |
| Training | loss / perplexity / match_rate curves from the real training history |
| Benchmark | held-out scores (format / numeric / teacher-forced) + answer samples |
| Questions | the 4 ROADMAP open questions and their empirical answers |
| Deployment | models for fine-tuning, free compute servers, repository bugs found |

## Getting started

```bash
cd neural_lab
npm install        # or bun install
npm run dev        # http://localhost:3000
```

Requirements: Node.js 20+; for the Playground tab, Python 3.10+ and NumPy
must be available on the system (inference runs as a subprocess:
`research/tinygpt_formula/scripts/infer.py`).
API endpoints: `/api/infer`, `/api/metrics`, `/api/evaluation`,
`/api/open-questions`, `/api` (health — checks for the model files).
