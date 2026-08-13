# Tutorial 2: Training TinyGPT

This tutorial walks you through training a 2.5M-parameter transformer from
scratch using pure NumPy. By the end you'll have:

- A trained model that produces syntactically-plausible Python code
- A plot of the loss / match-rate curves
- An intuition for why 30 epochs is enough (and why 100 isn't dramatically better)

The full runnable version is at
[`notebooks/02_training_tinygpt.ipynb`](https://github.com/wild8highlander/rmt-llm-research/blob/main/notebooks/02_training_tinygpt.ipynb).

---

## 1. Set up the trainer

```python
import sys
sys.path.insert(0, '../laboratory/python/lab_en')

import numpy as np
from tiny_gpt import TinyGPT, TinyGPTConfig
from tiny_gpt_trainer import BPETokenizer, TrainConfig, train

# Default config — 12 layers, hidden=128, vocab=512, BPE
config = TinyGPTConfig(
    vocab_size=512,
    hidden_dim=128,
    n_layers=12,
    n_heads=4,
    max_seq_len=32,
    mlp_ratio=4,
    use_mlp=True,
    use_layernorm=True,
    activation="gelu",
)
print(f"Total parameters: {TinyGPT(config).n_params:,}")
# Total parameters: 2,543,104
```

---

## 2. Build the BPE tokenizer

```python
# Load a small corpus (the project's own source code works well)
corpus_path = "../laboratory/python/lab_en/tiny_gpt.py"
with open(corpus_path) as f:
    corpus = f.read()

# Also include the trainer source
with open("../laboratory/python/lab_en/tiny_gpt_trainer.py") as f:
    corpus += "\n" + f.read()

print(f"Corpus size: {len(corpus):,} chars")

# Fit BPE with 256 merges (vocab=512 = 256 bytes + 256 merges)
tokenizer = BPETokenizer(vocab_size=512)
tokenizer.fit(corpus, n_merges=256)
print(f"BPE merges: {tokenizer.n_merges}")

# Verify roundtrip
sample = "def train(model):"
ids = tokenizer.encode(sample)
decoded = tokenizer.decode(ids)
assert decoded == sample, f"Roundtrip failed: {sample!r} -> {decoded!r}"
print(f"Roundtrip OK: {sample!r} -> {len(ids)} tokens -> {decoded!r}")
```

---

## 3. Configure training

```python
train_config = TrainConfig(
    epochs=30,
    batch_size=1,
    learning_rate=5e-4,
    weight_decay=1e-5,
    warmup_epochs=3,
    lr_schedule="cosine",
    min_lr_ratio=0.1,
    use_mlp=True,
    use_layernorm=True,
    seq_len=32,
    stride=64,
    eval_split=0.1,
    seed=42,
)
print(f"Training {config.n_layers} layers × {config.hidden_dim} hidden "
      f"for {train_config.epochs} epochs")
print(f"Optimizer: Adam (β1=0.9, β2=0.999, wd={train_config.weight_decay})")
print(f"LR schedule: cosine with {train_config.warmup_epochs}-epoch warmup")
```

---

## 4. Train!

```python
# This takes ~17 minutes on a CPU. For a quick demo, set epochs=3.
history = train(
    config=config,
    train_config=train_config,
    tokenizer=tokenizer,
    corpus=corpus,
    output_dir="results/notebook_run",
    verbose=True,
)
```

You'll see per-epoch progress:

```
[Epoch 01/30] loss=6.23  grad=2.41  lr=1.6e-04  mr=0.00%
[Epoch 02/30] loss=5.42  grad=2.18  lr=2.3e-04  mr=0.00%
[Epoch 05/30] loss=3.67  grad=1.62  lr=3.7e-04  mr=3.23%
[Epoch 10/30] loss=0.71  grad=0.41  lr=4.9e-04  mr=4.84%
[Epoch 15/30] loss=0.12  grad=0.12  lr=4.3e-04  mr=4.84%
[Epoch 20/30] loss=0.03  grad=0.04  lr=2.5e-04  mr=4.84%
[Epoch 25/30] loss=0.01  grad=0.02  lr=1.0e-04  mr=4.84%
[Epoch 30/30] loss=0.01  grad=0.01  lr=5.0e-05  mr=6.45%
```

---

## 5. Plot the training curves

```python
import matplotlib.pyplot as plt

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

epochs = [h["epoch"] for h in history["epochs"]]
losses = [h["loss"] for h in history["epochs"]]
mrs    = [h["match_rate"] * 100 for h in history["epochs"]]
lrs    = [h["lr"] for h in history["epochs"]]

ax1.semilogy(epochs, losses, 'b-o', ms=3)
ax1.set_ylabel('Loss (log scale)')
ax1.set_title('TinyGPT training — 30 epochs')
ax1.grid(True, alpha=0.3)

ax2.plot(epochs, mrs, 'g-o', ms=3)
ax2.set_ylabel('Match rate (%)')
ax2.set_ylim(0, max(mrs) * 1.2)
ax2.grid(True, alpha=0.3)

ax3.plot(epochs, lrs, 'r-o', ms=3)
ax3.set_ylabel('Learning rate')
ax3.set_xlabel('Epoch')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
plt.show()
```

You should see:

1. **Loss** — exponential decay, 730× reduction (6.23 → 0.0085)
2. **Match rate** — rises in steps (0% → 3% → 5% → 6.5%)
3. **LR** — cosine schedule: 3-epoch warmup → peak at 5e-4 → decay to 5e-5

---

## 6. Inspect the weights

```python
import numpy as np

weights = np.load("results/notebook_run/tiny_gpt_trained.npz")
print("Weight keys (first 10):")
for k in list(weights.keys())[:10]:
    print(f"  {k}: {weights[k].shape}")

# Check weight statistics
for k in ["layers.0.attn_wq", "layers.0.W_fc1", "layers.0.ln1_gamma"]:
    if k in weights:
        w = weights[k]
        print(f"{k}: mean={w.mean():.4f}, std={w.std():.4f}, "
              f"min={w.min():.4f}, max={w.max():.4f}")
```

---

## 7. Generate text

```python
from tiny_gpt_trainer import load_trained_model, generate_sample

model, tok = load_trained_model(
    "results/notebook_run/tiny_gpt_trained.npz",
    "results/notebook_run/tiny_gpt_bpe.json",
)

prompts = ["def train(", "import numpy", "class Tiny", "# "]
for prompt in prompts:
    ids = tok.encode(prompt)
    out = generate_sample(model, ids, max_new_tokens=30, temperature=0.5)
    print(f"\nPrompt: {prompt!r}")
    print(f"Output: {tok.decode(out)!r}")
```

---

## Understanding the results

### Why does loss decay exponentially?

The corpus is small (~2MB), so the model is essentially **memorizing** it.
After 30 epochs, the loss is 0.0085 — well into the memorization regime.
This is fine for a pedagogical model; the goal is not generalization but
transparency.

### Why is match_rate only 6.5%?

Three reasons (see [FAQ](../getting-started/faq.md#why-is-match_rate-only-65)):

1. Small model (2.5M params)
2. Small corpus (2MB vs. GPT-2's 40GB)
3. Small context (32 tokens vs. 2048+)

6.5% on a 512-token BPE vocab is 33× the random baseline (1/512 ≈ 0.2%).

### Why does the LR schedule use cosine decay?

Cosine LR + warmup is the modern default (GPT-3, LLaMA, etc.):

- **Warmup** (epochs 1-3) prevents early instability when gradients are large
- **Cosine decay** smoothly reduces LR to encourage convergence
- **min_lr_ratio=0.1** keeps LR at 10% of peak (not zero) to avoid getting stuck

### What would improve match_rate?

| Change | Expected lift | Cost |
|--------|---------------|------|
| 10× larger corpus | +5-10% | 10× training time + data |
| 4× larger model (10M params) | +5-15% | 4× training time + memory |
| 4× longer context (128 tokens) | +3-8% | 4× memory per batch |
| 100 epochs (was 30) | +1-2% | 3× training time |

Diminishing returns kick in fast. The current setup is a good
pedagogical sweet spot.

---

## What's next?

- [Tutorial 3: Generation & Sampling](03-generation-and-sampling.md)
- [API: trainer](../api/trainer.md)
- [Architecture: Request Flow](../architecture/request-flow.md)
