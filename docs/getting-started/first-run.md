# First Run

This page walks you through your **first complete TinyGPT training run** —
from `git clone` to a model that generates text.

Estimated time: **~20 minutes** on a modern laptop (no GPU needed).

---

## Step 1: Install

```bash
git clone https://github.com/wild8highlander/rmt-llm-research.git
cd rmt-llm-research
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Verify:

```bash
pytest laboratory/python/lab_en/tests/ -v --tb=short | tail -5
```

You should see `101 passed`.

---

## Step 2: Open the lab menu

```bash
cd laboratory/python/lab_en
python main.py
```

You'll see a 15-item menu:

```
=== RMT-LLM Research Laboratory ===
1.  Run scenario SCEN-LIE-01 ...
2.  Run scenario SCEN-HALL-02 ...
...
14. Train TinyGPT model
15. Generate text from trained TinyGPT
...
```

---

## Step 3: Train (menu item 14)

Choose `14`. The trainer will:

1. **Fit BPE tokenizer** (~2 seconds, 256 merges on the corpus)
2. **Build sliding-window dataset** (`seq_len=32`, `stride=64`)
3. **Run 30 epochs** of Adam with cosine LR + 3-epoch warmup
4. **Save** weights + BPE + training history

You'll see per-epoch progress:

```
[Epoch 01/30] loss=6.2251  grad_norm=2.41  lr=1.64e-04  mr=0.000%
[Epoch 02/30] loss=5.4192  grad_norm=2.18  lr=2.31e-04  mr=0.000%
...
[Epoch 15/30] loss=0.1236  grad_norm=0.12  lr=4.31e-04  mr=4.839%
...
[Epoch 30/30] loss=0.0085  grad_norm=0.01  lr=5.00e-05  mr=6.452%

Training complete. Loss reduced 730x. Match rate: 0% -> 6.5%
Weights saved to: results/models/tiny_gpt_trained.npz
BPE saved to:     results/models/tiny_gpt_bpe.json
```

Total wall-clock time: ~17 minutes.

### Understanding the metrics

| Metric | What it means | Good values |
|--------|---------------|-------------|
| `loss` | Cross-entropy on the batch | Decreasing; < 0.1 = memorized |
| `grad_norm` | L2 norm of the gradient | Stable; not exploding/vanishing |
| `lr` | Current learning rate (cosine schedule) | Rises (warmup) then falls |
| `mr` | Match rate (greedy next-token accuracy) | Increasing; 6.5% = 33× random |

### If training crashes

The most common cause is **insufficient RAM**. Try reducing `n_layers` in
`tiny_gpt_trainer.py:TrainConfig` from 12 to 6, or `hidden_dim` from 128 to 64.

The trainer also supports **chunked training** with checkpoint recovery —
see `scripts/train_v2_chunk.py` for an example that runs 5 epochs per
invocation and resumes from a checkpoint.

---

## Step 4: Generate text (menu item 15)

Choose `15` from the menu. You'll be prompted for:

| Field | Suggested value | Notes |
|-------|-----------------|-------|
| Weights path | `results/models/tiny_gpt_trained.npz` | From step 3 |
| BPE path | `results/models/tiny_gpt_bpe.json` | From step 3 |
| Prompt | `def train(` | Any string the model has seen |
| Temperature | `0.5` | 0 = greedy, 1 = max randomness |
| Max new tokens | `50` | How many tokens to generate |

### Sample output (temperature 0.5)

```
Prompt: 'def train('
Output: 'def train(er, erFloatilamthpathvvth) ) catendpath:endpathnell'
```

The model produces **real Python tokens** (`def`, `train`, `Float`, `path`,
`end`, `return`) but the sentence-level structure is incoherent — this is
expected for a 2.5M-param model trained on 2MB of source code.

### Try different prompts

| Prompt | Why |
|--------|-----|
| `The RMT-LLM` | Tests prefix continuation |
| `import numpy` | Tests import-statement structure |
| `class Tiny` | Tests class-definition structure |
| `# ` | Tests comment generation |
| Russian: `class Tiny` (in `lab_ru/`) | Tests the RU mirror |

### Try different temperatures

- `0.0` — greedy decoding (often stuck in repetition loops)
- `0.5` — focused but varied (recommended starting point)
- `1.0` — full randomness (most varied output)
- `1.5` — too random (mostly gibberish)

---

## Step 5: Inspect the training history

```python
import json
from pathlib import Path

hist_path = Path("results/models/training_history.json")
history = json.loads(hist_path.read_text())

for epoch in history["epochs"][-5:]:
    print(
        f"Epoch {epoch['epoch']:2d}  "
        f"loss={epoch['loss']:.4f}  "
        f"grad_norm={epoch['grad_norm']:.4f}  "
        f"lr={epoch['lr']:.2e}  "
        f"mr={epoch['match_rate']:.3%}"
    )
```

Expected output (last 5 epochs):

```
Epoch 26  loss=0.0128  grad_norm=0.0150  lr=5.00e-05  mr=6.452%
Epoch 27  loss=0.0099  grad_norm=0.0120  lr=5.00e-05  mr=6.452%
Epoch 28  loss=0.0087  grad_norm=0.0100  lr=5.00e-05  mr=6.452%
Epoch 29  loss=0.0085  grad_norm=0.0090  lr=5.00e-05  mr=6.452%
Epoch 30  loss=0.0085  grad_norm=0.0090  lr=5.00e-05  mr=6.452%
```

---

## Step 6: Run the cross-implementation tests

```bash
cd ../../../..    # back to repo root
pytest src/rmt_llm/tests/test_cross.py -v
```

This verifies that the Python, Julia, and JSON-schema outputs agree.

---

## Step 7: Run a 3D visualization

```bash
cd python/rmt_llm_viz
python main.py --viz 1      # Marchenko-Pastur 3D density
```

A matplotlib window should appear with the MP density surface over the
$(\lambda, q)$ plane.

---

## What's next?

- [Tutorial: Quickstart Notebook](../tutorials/01-quickstart.md) — full
  walkthrough in Jupyter
- [API Reference](../api/index.md) — every public function
- [Architecture](../architecture/index.md) — how the pieces fit together
- [FAQ](faq.md) — common questions
