<p align="center">
  <a href="../README.md"><img src="../assets/branding/aurora-icon.svg" width="72" alt="RMT × LLM Research"></a>
</p>

# 🌐 neural_lab — Web App

<p align="center">
  <em>A Next.js playground where the trained formula model lives</em><br><br>
  <img src="../assets/branding/aurora-divider.svg" width="420" alt=""><br><br>
  <a href="../README.md"><img src="https://img.shields.io/badge/folder-neural--lab-00ffa3" alt="repository root"></a>
  <img src="https://img.shields.io/badge/next.js-000000" alt="Next.js"><img src="https://img.shields.io/badge/typescript-3178C6" alt="TypeScript">
</p>

A web application (Next.js + TypeScript) where the trained model from
[`research/tinygpt_formula/`](research/tinygpt_formula/README.md)
"lives": the playground calls `scripts/infer.py` through the API, while
the other tabs display the training artifacts.

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

---

<p align="center">
  <sub>
    <a href="../README.md">← root</a> · <a href="README.md">top</a> · <a href="../README.md">↑ parent</a> · <a href="../README.md">🏠 repository root</a> · <a href="research/tinygpt_formula/README.md">the model →</a>
  </sub>
</p>
