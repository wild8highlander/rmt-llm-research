<p align="center">
  <img src="../../../assets/logos/sc_exp.banner.svg" width="760" alt="science/experiments — animated logo: rocket with flickering flame">
</p>

# 🚀 science/experiments — Runners

<p align="center">
  <em>CLI entry points that turn a frozen protocol into fixed results</em><br><br>
  <a href="../README.md"><img src="https://img.shields.io/badge/module-science--experiments-fb7185" alt="experiments"></a>
  <img src="https://img.shields.io/badge/CLI-fbbf24" alt="CLI"><img src="https://img.shields.io/badge/resumable-fbbf24" alt="resumable">
</p>

The only place where the science core *runs*. An experiment here is not
a notebook: it is a command-line runner that executes a pre-registered
protocol end-to-end — instrument the model, extract features, fit
predictors, evaluate against baselines — and writes everything into an
output folder with a passport, machine-readable results, and a generated
report. The runner never reads a previous result folder as input for new
claims; every run is self-contained and reproducible from the passport.

## Contents

| File | Purpose |
|---|---|
| `run_h1.py` | the H1 experiment: full pipeline `data → instrumentation → features → predictors → evaluation → REPORT`, producing `passport.json`, `report.json`, JSONL traces, and a generated `REPORT.md` |
| `__init__.py` | package surface |

## Running

```bash
# full H1 run (~3 min CPU after the model is trained)
RMT_LLM_BASE=research/tinygpt_formula \
  python3 neural_lab/science/experiments/run_h1.py

# smoke run (~1 min, greedy generation) — pipeline integrity check
SMOKE=1 python3 neural_lab/science/experiments/run_h1.py --out /tmp/h1_smoke

# long runs can stop and resume (observation checkpoints)
RMT_H1_OBS_RESUME=1 python3 neural_lab/science/experiments/run_h1.py ...
```

## Output contract

Every run lands in its own `--out` folder (never overwriting a
committed result): `passport.json` (sha256 of weights/corpus/dataset,
seeds, environment), `tasks.json`, `observations.jsonl`,
`labels.jsonl`, `predictions.jsonl`, `report.json`, and `REPORT.md`
generated from the data, not from hands. Fixed results live in
[`results/`](../results/README.md); the frozen protocol they obey is
[`../../EXPERIMENT_PROTOCOL.md`](../../EXPERIMENT_PROTOCOL.md).

---

<p align="center">
  <img src="../../../assets/logos/sc_exp.avatar.svg" width="56" alt="science/experiments animated avatar"><br>
  <sub>
    <a href="../README.md">← science</a> · <a href="../../README.md">← neural_lab</a> · <a href="../results/README.md">results →</a>
  </sub>
</p>
