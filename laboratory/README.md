# 🧪 RMT-LLM Laboratory

A complete multi-language research laboratory for verifying the RMT-LLM theory and the news claims about hidden reasoning chains in LLMs. Built as an extension to the `rmt-llm-research` repository.

> **Verifying**: "Claude, Gemini, ChatGPT were cracked — hidden reasoning chains exposed. Models lie, hallucinate, and store PII."

## 🆕 v1.5.1 — Expanded Corpus + Full 10-Epoch Training Run

This version expands the training corpus from 16 files (~240 KB) to **63 files (~913 KB)** and completes a full 10-epoch training run:

- **Corpus expansion** (`DEFAULT_CORPUS_PATHS` in both `tiny_gpt_trainer.py` EN+RU):
  - Added repo-level docs: `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, `AUTHORS.md`, `CITATION.cff`, `.zenodo.json`, `pyproject.toml`
  - Added original library sources (rich RMT/LLM prose): `python/rmt_llm_viz/main.py`, `julia/RMTLLMVerify/src/`, `julia/RMTLLMViz/src/`, `java/rmt-llm-viz/src/main/java/com/rmt/llm/viz/*`
  - Added all laboratory source files across **8 languages**: Python, Julia, Java, Rust, Go, C++, R, webapp (HTML/JS/CSS)
  - Russian trainer points to `lab_ru/` versions
- **Full 10-epoch training run results**:
  - Configuration: `seq_len=32, stride=128, batch=4, lr=5e-4, weight_decay=1e-5, epochs=10`
  - Runtime: 173.1 s (2.9 min) on 6,572 train windows + 730 eval windows
  - `match_rate`: **0.000% → 32.031%** (82× better than random 0.39%)
  - `loss`: 5.5488 → 3.0777 (44% reduction)
  - Peak match_rate: 34.375% (epochs 6, 7, 8, 10)
  - Saved weights: `laboratory/python/lab_en/results/models/tiny_gpt_trained.npz` (595 KB)
  - Training history JSON: `laboratory/python/lab_en/results/models/training_history.json`
- **Generation comparison**: untrained model emits random high-entropy bytes; trained model produces structured low-entropy patterns with frequent ASCII (space, `\t`, `F`, `f`, `{`, `}` — top bytes in Python/JSON source)
- **Why ~32% plateau**: TinyGPT is intentionally minimal (no MLP, no layernorm, ~2M params, byte-level vocab=256) — 32% is at the capacity ceiling. For 50%+, increase `n_layers` to 12, use BPE tokenizer, train 30+ epochs

## 🆕 v1.5.0 — TinyGPT Real-Corpus Training + 3D Modules for All Languages

This version adds a **real-corpus training pipeline** for TinyGPT (raising `match_rate` from 0% to ~28% in 14 s) and ports the **9 3D research experiments** to all 6 non-Python languages:

- **TinyGPT trainer** (`laboratory/python/lab_{en,ru}/tiny_gpt_trainer.py`):
  - Full reverse-mode autodiff through the synthetic transformer (softmax-CE → lm_head → residual → softmax-attention → QK^T → softmax → AV → output projection → residual)
  - Adam optimizer with bias correction and weight decay
  - Corpus assembled from README.md, CHANGELOG.md, source code, JSON configs, HTML docs (~240 KB)
  - Byte-level next-token prediction; 90/10 train/eval split
  - Smoke-tested: 3 epochs / 14 s / `match_rate` 0.000% → 28.125% / loss 5.54 → 3.30
  - Menu items 14 (train) + 15 (generate) added to `main.py` (EN+RU)
- **3D research modules** for Julia, Java, Rust, Go, C++, R (EN+RU each):
  - All 9 experiments (IDs 6–14) implemented in each language
  - Pure-stdlib implementations (no external plotting deps) — Jacobi eigenvalue algorithm, numerical-stability softmax, manual JSON serialization
  - Cross-language JSON schema: byte-compatible so Python `charts_3d.py` renders any language's output into 25 chart files
  - R additionally emits native PNG (600 DPI) / PDF / SVG via `persp()` and `plot()`
  - Menu items 11/12/13 added to each language's main file
- **Pre-existing bugs fixed**: Java `TinyGPT.hiddenSnapshots` type mismatch, Go `1.0/0.0` compile error, Go `go.mod` invalid `author` directive

## 🆕 v1.4.0 — Professional 3D Expansion

This version adds **8 professional 3D chart types** + **9 new 3D research experiments** on top of the existing 2D pipeline:

- **3D charts**: loss landscape, hidden-state manifold (PCA), spectral surface, deception trajectory, attention flow, N_crit collapse surface, parameter-space sweep, coalitional deception drift — each in PNG 600 DPI + PDF + SVG + interactive Plotly HTML (25 files total per run).
- **3D experiments**: Hessian loss landscape, manifold geometry (participation ratio), reasoning trajectory analysis, spectral surface regression, Riemannian curvature, 3D attention flow, N_crit collapse surface, parameter-space sweep, coalitional deception drift.
- **7 new scenarios** (total: 13): manipulation resistance, self-preservation, coalitional deception, memory injection, adversarial suffix, multi-turn manipulation, temporal reasoning inconsistency.
- **7 new models** (total: 14): distilgpt2, Phi-1.5, Qwen2-0.5B, OPT-125M, Pythia-70M, BLOOMZ-560M, Falcon-RW-1B.
- **3D parameters** added to `shared/schema.json`: `manifold_dims`, `hessian_grid_size`, `trajectory_points`, `curvature_neighbors`, etc. (all support `inf`).

---

## 📁 Directory structure

```text
laboratory/
├── shared/                       # Shared protocol (JSON, language-agnostic)
│   ├── schema.json               # JSON schema for infinite parameter system (incl. 3D params v1.1.0)
│   ├── scenarios.json            # 13 pre-defined scenarios (bilingual: EN+RU)
│   └── model_registry.json       # 14 model entries with download URLs
│
├── python/                       # Python — full reference implementation
│   ├── lab_en/                   # English version (12 files)
│   │   ├── main.py               # Interactive menu (13 options + Exit)
│   │   ├── parameters.py         # Infinite parameter system
│   │   ├── tiny_gpt.py           # Local TinyGPT (~2M params)
│   │   ├── model_downloader.py   # HuggingFace / ONNX downloader
│   │   ├── charts.py             # 8 chart types × 4 formats (PNG 600 DPI + PDF + SVG + HTML)
│   │   ├── charts_3d.py          # 8 3D chart types × 4 formats (PNG 600 DPI + PDF + SVG + Plotly HTML)
│   │   ├── reports.py            # 13 report formats
│   │   ├── research.py           # 5 research experiments (2D)
│   │   ├── research_3d.py        # 9 3D research experiments (v1.1.0)
│   │   ├── scenarios.py          # Scenario runner
│   │   ├── run_full_lab.py       # Non-interactive driver
│   │   ├── __init__.py
│   │   └── requirements.txt
│   └── lab_ru/                   # Russian mirror (same 10 files, RU strings)
│
├── julia/                        # Julia implementation
│   ├── lab_en/                   # Project.toml + 7 .jl files
│   └── lab_ru/                   # Russian mirror
│
├── java/                         # Java implementation
│   ├── lab_en/                   # Main.java + 7 supporting classes + build.gradle
│   └── lab_ru/                   # Russian mirror
│
├── rust/                         # Rust implementation (pure std, no deps)
│   ├── lab_en/                   # main.rs + Cargo.toml
│   └── lab_ru/                   # Russian mirror
│
├── go/                           # Go implementation (pure stdlib)
│   ├── lab_en/                   # main.go + go.mod
│   └── lab_ru/                   # Russian mirror
│
├── cpp/                          # C++ implementation (C++17, pure std)
│   ├── lab_en/                   # main.cpp + Makefile
│   └── lab_ru/                   # Russian mirror
│
├── r/                            # R implementation
│   ├── lab_en/                   # main.R
│   └── lab_ru/                   # Russian mirror
│
├── webapp/                       # React + Socket.io real-time dashboard
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   ├── index.html
│   ├── README.md
│   ├── server/
│   │   └── server.js             # Express + Socket.io server
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       ├── store.js              # Zustand state + Socket.io client
│       ├── styles/index.css
│       └── components/
│           ├── Header.jsx
│           ├── Sidebar.jsx
│           ├── Dashboard.jsx
│           ├── ScenariosView.jsx
│           ├── ExperimentsView.jsx
│           ├── ParametersView.jsx
│           ├── ModelsView.jsx
│           ├── LiveMonitor.jsx
│           ├── ChartsView.jsx
│           ├── ReportsView.jsx
│           └── LogsView.jsx
│
├── results/                      # Generated outputs (gitignored)
│   ├── charts/                   # PNG 600 DPI + PDF + SVG + Plotly HTML
│   ├── reports/                  # 13 formats
│   ├── logs/                     # Full task launch logs
│   └── models/                   # Downloaded model weights
│
└── docs/                         # Laboratory documentation
```

---

## 🎯 What this laboratory verifies

The lab tests the four claims from the news article about LLM hidden reasoning:

| News claim | Lab scenario | What we measure |
|---|---|---|
| Models know the answer but lie about reasoning | `SCEN-LIE-01` | Whether hidden reasoning trace disagrees with output |
| Models hallucinate past a critical token count | `SCEN-HALL-02` | Spectral collapse predicted by RMT N_crit vs empirical onset |
| Models plan to deceive the user | `SCEN-DECEIT-03` | Deception score in hidden trace vs honesty score |
| Models store thousands of passwords/tokens/API keys | `SCEN-DATA-04` | PII pattern completion rate from weights |
| Filters fire only before answer, not during reasoning | `SCEN-FILTER-05` | Filter bypass count in pre-filter trace |

---

## 🚀 Quick start

### Python (full implementation, recommended)

```bash
cd laboratory/python/lab_en
pip install -r requirements.txt
python main.py
```

Or run everything non-interactively:

```bash
python run_full_lab.py
```

### Web app (real-time dashboard)

```bash
cd laboratory/webapp
npm install
npm start
# Open http://localhost:5173
```

### Other languages

```bash
# Julia
cd laboratory/julia/lab_en
julia --project=. main.jl

# Java
cd laboratory/java/lab_en
javac *.java && java Main
# OR: gradle run

# Rust
cd laboratory/rust/lab_en
cargo run --release

# Go
cd laboratory/go/lab_en
go run main.go

# C++
cd laboratory/cpp/lab_en
make && ./rmt_llm_lab_en

# R
cd laboratory/r/lab_en
Rscript main.R
```

> **Note**: Non-Python languages spawn the Python lab as a subprocess for actual neural network computation. This ensures consistency across all implementations while keeping each language's interactive menu, parameter handling, and report generation native.

---

## 📊 Outputs produced per run

Every run produces (under `laboratory/results/`):

### Charts (in `charts/<timestamp>_<suffix>/`)

| File | Format | Description |
|---|---|---|
| `01_loss_metrics.{png,pdf,svg}` | PNG 600 DPI / PDF / SVG | Training loss & accuracy |
| `02_eigenvalue_vs_mp.{png,pdf,svg}` | PNG 600 DPI / PDF / SVG | Eigenvalue distribution vs Marchenko-Pastur bounds |
| `03_confusion_matrix.{png,pdf,svg}` | PNG 600 DPI / PDF / SVG | Lie/Truth/Hallucinate/Refuse confusion matrix |
| `04_roc_deception.{png,pdf,svg}` | PNG 600 DPI / PDF / SVG | ROC curve for deception detection |
| `05_hallucination_dist.{png,pdf,svg}` | PNG 600 DPI / PDF / SVG | Hallucination score distribution |
| `06_per_layer_gap.{png,pdf,svg}` | PNG 600 DPI / PDF / SVG | Per-layer spectral gap |
| `07_reasoning_trace.{png,pdf,svg}` | PNG 600 DPI / PDF / SVG | Honesty vs deception vs hallucination over reasoning steps |
| `08_ncrit_threshold.{png,pdf,svg}` | PNG 600 DPI / PDF / SVG | RMT-predicted N_crit vs empirical hallucination onset |
| `interactive_dashboard.html` | Plotly HTML | Interactive multi-chart dashboard |

### Reports (in `reports/`)

13 formats per run, each containing:

- **Part I** — Detailed results with explanations
- **Part II** — Full task launch logs

| Format | File extension | Purpose |
|---|---|---|
| Plain text | `.txt` | Simple viewing |
| Markdown | `.md` | GitHub / docs |
| CSV | `.csv` | Tabular analysis |
| HTML | `.html` | Self-contained web page |
| JSON | `.json` | Structured data |
| PDF | `.pdf` | Print-ready document |
| Word | `.docx` | Editable document |
| YAML | `.yaml` | ML config-friendly |
| XML | `.xml` | Legacy system compat |
| LaTeX | `.tex` | Academic publishing |
| Parquet | `.parquet` | Big-data columnar |
| Excel | `.xlsx` | Spreadsheet analysis |
| SQLite | `.sqlite` | SQL queries |

### Logs (in `logs/`)

Full timestamped log of every step: scenario loading, parameter merging, model instantiation, per-prompt generation, spectral analysis, error messages.

---

## ⚙️ Infinite parameter system

Every parameter accepts values in `[0, +∞)`. The string `"inf"` is accepted for true infinity. Both JSON config files and the interactive wizard support this.

### Key parameters

| Parameter | Default | Range | Meaning |
|---|---|---|---|
| `temperature` | 0.7 | [0, ∞) | 0 = greedy, inf = pure random |
| `max_tokens` | 256 | [1, ∞) | Maximum tokens to generate |
| `top_k` | 50 | [0, ∞) | 0 = disabled, inf = no filter |
| `top_p` | 0.95 | [0, 1] | Nucleus sampling mass |
| `context_window` | 1024 | [1, ∞) | Context window size |
| `ncrit_threshold` | 114.0 | [0, ∞) | RMT critical token count |
| `theta_b_deg` | 7.07 | [0, 360] | BBP rotation angle |
| `beta_caputo` | 0.5 | [0, ∞) | Caputo fractional memory |
| `rlhf_pressure` | 0.0 | [0, ∞) | RLHF drift strength (accelerates hallucination) |
| `n_layers` | 6 | [1, ∞) | Transformer layers |
| `hidden_dim` | 64 | [1, ∞) | Hidden dimension |
| `n_heads` | 4 | [1, ∞) | Attention heads |
| `vocab_size` | 256 | [1, ∞) | Vocabulary size |
| `seed` | 42 | [0, ∞) | Random seed |
| `epochs` | 3 | [0, ∞) | Training epochs |
| `learning_rate` | 1e-3 | [0, ∞) | Learning rate |
| `batch_size` | 4 | [1, ∞) | Batch size |
| `enable_filter` | true | bool | Output safety filter |
| `capture_hidden` | true | bool | Capture hidden reasoning trace |
| `language` | "en" | en/ru | Output language |

### Example config file

```json
{
  "scenario_id": "SCEN-LIE-01",
  "parameters": {
    "temperature": 0.01,
    "max_tokens": 512,
    "rlhf_pressure": 1.5,
    "ncrit_threshold": 80
  }
}
```

---

## 🤖 Synthetic neural network (TinyGPT)

A pure-NumPy tiny transformer (~2M parameters):

- 6 layers, hidden dim 64, 4 attention heads, vocab 256 (byte-level)
- Forward pass with hidden state capture per layer
- Generation with temperature / top-k / top-p sampling
- Synthetic reasoning trace that mimics the "hidden CoT" exposed by the news

For richer experiments, the lab can also download small real models from:

- **HuggingFace**: GPT-2 small (124M), DialoGPT-small (117M), TinyLlama-1.1B, BERT-base
- **ONNX Model Zoo**: SqueezeNet 1.1
- **Keras.js demos**: MNIST MLP (browser-only)

See [`shared/model_registry.json`](./shared/model_registry.json) for the full catalog.

---

## 🌐 Web app features

The React + Socket.io dashboard at `laboratory/webapp/` provides 9 tabs:

1. **Dashboard** — Overview cards, live reasoning trace chart, latest results radar
2. **Scenarios** — Browse and run the 6 pre-defined scenarios
3. **Experiments** — Browse and run the 5 research experiments
4. **Parameters** — Infinite parameter editor (all bounds support `inf`)
5. **Models** — Registry browser with one-click download
6. **Live Monitor** — Real-time metrics stream, reasoning thoughts, log stream
7. **Charts** — Gallery of all generated charts (PNG/PDF/SVG/HTML)
8. **Reports** — 13-format report browser with download links
9. **Logs** — Full log viewer with search and level filter

Real-time updates flow via Socket.io:

- Server spawns the Python lab as a subprocess
- Python stdout is streamed as `log` events
- Mock metrics are emitted as `metric` events
- On completion, `results` event delivers the full results JSON

---

## 🔐 License

All laboratory code is licensed under the **Proprietary All-Rights-Reserved License** (see root [`LICENSE`](../LICENSE)). All rights belong exclusively to Iskhak Hamzatovich Isaev. No distribution, no academic redistribution, no commercial use, no derivative works, no AI training.

---

## 👤 Author

**Iskhak Hamzatovich Isaev**

- ORCID: [0009-0003-7299-0701](https://orcid.org/0009-0003-7299-0701)
- GitHub: [wild8highlander](https://github.com/wild8highlander)
- Repository: [rmt-llm-research](https://github.com/wild8highlander/rmt-llm-research)

---

*Built to verify: "models lie, hallucinate, and store PII — and we voluntarily gave them all our data."*
