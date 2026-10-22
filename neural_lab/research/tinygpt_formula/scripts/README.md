<p align="center">
  <img src="../../../assets/logos/scripts.banner.svg" width="760" alt="scripts — animated logo: two counter-rotating gears with an orbiting signal dot">
</p>

# ⚙️ scripts — Pipeline

<p align="center">
  <em>Corpus → training → evaluation → inference → open questions</em><br><br>
  <a href="../README.md"><img src="https://img.shields.io/badge/folder-research-a78bfa" alt="tinygpt_formula/"></a>
  <img src="https://img.shields.io/badge/5%20stages-00ffa3" alt="5 stages"><img src="https://img.shields.io/badge/cpu----friendly-22d3ee" alt="CPU friendly">
</p>

The five-stage pipeline of the Formula-Edition experiment. Run the
stages in order and the full study reproduces in roughly 20 minutes on a
CPU — every stage is deterministic and writes its artifacts next door in
[`model/`](../model/README.md). Each script also refuses to guess: if an
upstream artifact is missing, it says so and exits instead of silently
rebuilding with different content, which is what keeps old reports
interpretable months later.

| Script | Stage |
|---|---|
| `build_corpus.py` | 1 — build the formula + control corpora into [`data/`](../data/README.md) |
| `train_formula_model.py` | 2 — train TinyGPT (NumPy, BPE, fixed seeds) |
| `evaluate_model.py` | 3 — held-out evaluation: format / numeric / teacher-forced |
| `infer.py` | 4 — interactive inference; also backs the [`neural_lab/`](../../../README.md) playground |
| `open_questions.py` | 5 — empirical answers to the ROADMAP open questions |

## Stage details

**1 — `build_corpus.py`.** Walks the `src/rmt_llm/` core and the monograph
source list, emits `formula_corpus_train.txt` / `formula_corpus_val.txt`
plus the control twin with corrupted numerics (seed 42). Safe to re-run:
the output is byte-identical for the same repository state, and any
divergence is a signal that the core changed — which should be visible,
not hidden.

**2 — `train_formula_model.py`.** Trains TinyGPT v3 with AdamW + cosine
schedule and writes weights, tokenizer, and per-epoch history to
`model/`. Interrupted runs resume from the last completed epoch;
`TRAIN_MAX_MINUTES=15` makes it easy to train in coffee-break chunks on
a laptop or a phone.

**3 — `evaluate_model.py`.** Scores the trained model on 37 held-out
tasks the corpus never contained, splitting the result into format /
numeric / teacher-forced so that "fluent but wrong" is visible as such.
Writes `evaluation_report.json`, which both this README's parent table
and the web app's Benchmark tab read.

**4 — `infer.py`.** JSON-in / JSON-out inference used by humans, tests,
and the web app alike:

```bash
python3 research/tinygpt_formula/scripts/infer.py \
  '{"prompt":"mp_bounds(q=0.5, sigma2=1.0)","max_new_tokens":48,"temperature":0.0,"seed":42}'
```

**5 — `open_questions.py`.** Runs the four empirical tests (free
cumulants, Caputo-Langevin transfer, BBP lie-detector probe, Hutchinson
trace) and writes `model/open_questions_results.json` with per-question
status and raw numbers.

## Common failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `weights not found` in stage 3/4 | training never completed | run stage 2 (or restore `model/*.npz`) |
| corpus hash mismatch in a report | core changed since the artifacts were built | re-run stages 1–3; the old report stays valid for the old hash |
| training crawls on Termux | BLAS without acceleration | use `pkg install python-numpy` (OpenBLAS build), see `../scaling/TERMUX_PROOT_UBUNTU.md` |

Cloud variant: the notebooks in [`colab/`](../colab/README.md) run the
same stages on free GPU servers.

---

<p align="center">
  <img src="../../../assets/logos/scripts.avatar.svg" width="56" alt="scripts animated avatar"><br>
  <sub>
    <a href="../README.md">← tinygpt_formula</a> · <a href="README.md">top</a> · <a href="../README.md">↑ parent</a> · <a href="../../README.md">🏠 repository root</a> · <a href="../colab/README.md">cloud notebooks →</a>
  </sub>
</p>
