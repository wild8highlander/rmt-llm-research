# Tutorial 3: Generation & Sampling

This tutorial covers loading trained TinyGPT weights and generating text with
different sampling strategies.

The full runnable version is at
[`notebooks/03_generation_and_sampling.ipynb`](https://github.com/wild8highlander/rmt-llm-research/blob/main/notebooks/03_generation_and_sampling.ipynb).

---

## 1. Load the trained model

```python
import sys
sys.path.insert(0, '../laboratory/python/lab_en')

from tiny_gpt_trainer import load_trained_model

# Use the pre-trained weights checked into the repo
WEIGHTS = "../laboratory/python/lab_en/results/models/tiny_gpt_trained.npz"
BPE     = "../laboratory/python/lab_en/results/models/tiny_gpt_bpe.json"

model, tokenizer = load_trained_model(WEIGHTS, BPE)
print(f"Loaded: {model.n_layers} layers, {model.n_params:,} params")
print(f"BPE: {tokenizer.n_merges} merges, vocab={tokenizer.vocab_size}")
```

---

## 2. Greedy decoding (temperature=0)

Greedy decoding picks the highest-probability token at every step. It's
deterministic but tends to get stuck in repetition loops.

```python
from tiny_gpt_trainer import generate_sample

prompt = "def train("
ids = tokenizer.encode(prompt)
out = generate_sample(model, ids, max_new_tokens=50, temperature=0.0)
print(tokenizer.decode(out))
```

Typical output:

```text
def train(arararars.llllllll...)
```

The model has memorized that `ar` is a frequent bigram and gets stuck.
This is expected for an overfit small model.

---

## 3. Stochastic sampling (temperature=0.5)

```python
for _ in range(3):
    out = generate_sample(model, ids, max_new_tokens=50, temperature=0.5, seed=None)
    print(tokenizer.decode(out))
    print("---")
```

You'll see different output each time, with **real Python tokens**
(`Float`, `String`, `return`, `import`, `def`) mixed with random tokens.

---

## 4. Temperature sweep

```python
import matplotlib.pyplot as plt

temperatures = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5]
samples = {}

for temp in temperatures:
    np.random.seed(42)  # reproducible per-temperature
    out = generate_sample(model, ids, max_new_tokens=80, temperature=temp)
    samples[temp] = tokenizer.decode(out)

for temp, text in samples.items():
    print(f"\n=== T={temp} ===")
    print(text[:200])
```

Expected pattern:

| Temperature | Behavior |
|-------------|----------|
| 0.1 | Almost greedy — repetition loops |
| 0.3 | Mild variation — mostly repetition |
| 0.5 | Focused but varied (recommended) |
| 0.7 | More random — real tokens + noise |
| 1.0 | Full randomness — most varied |
| 1.5 | Too random — mostly noise |

---

## 5. Top-k and top-p sampling (advanced)

The current `generate_sample` does **full-vocabulary sampling** (every token
has nonzero probability). For production-grade sampling, you'd implement:

- **Top-k**: keep only the top-k highest-probability tokens
- **Top-p (nucleus)**: keep the smallest set of tokens whose cumulative
  probability ≥ p

Example implementation sketch:

```python
def top_k_sample(logits, k=10, temperature=1.0):
    logits = logits / temperature
    top_k_idx = np.argpartition(logits, -k)[-k:]
    top_k_logits = logits[top_k_idx]
    probs = np.exp(top_k_logits - top_k_logits.max())
    probs /= probs.sum()
    choice = np.random.choice(len(top_k_idx), p=probs)
    return top_k_idx[choice]
```

This is left as an exercise — see [`tiny_gpt_trainer.py`](https://github.com/wild8highlander/rmt-llm-research/blob/main/laboratory/python/lab_en/tiny_gpt_trainer.py)
for the current implementation.

---

## 6. Batch generation

```python
prompts = [
    "def train(",
    "import numpy",
    "class Tiny",
    "The RMT-LLM",
    "# A comment",
]

for prompt in prompts:
    ids = tokenizer.encode(prompt)
    out = generate_sample(model, ids, max_new_tokens=40, temperature=0.7)
    text = tokenizer.decode(out)
    print(f"\nPrompt: {prompt!r}")
    print(f"  Out:  {text!r}")
```

---

## 7. Measuring generation speed

```python
import time

n_tokens = 100
n_runs = 5

times = []
for _ in range(n_runs):
    t0 = time.perf_counter()
    out = generate_sample(model, ids, max_new_tokens=n_tokens, temperature=0.5)
    times.append(time.perf_counter() - t0)

mean_time = sum(times) / n_runs
tokens_per_sec = n_tokens / mean_time
print(f"Mean time: {mean_time:.3f}s for {n_tokens} tokens")
print(f"Throughput: {tokens_per_sec:.1f} tokens/sec")
```

On a typical CPU you should see **~50-100 tokens/sec** for the 12-layer
hidden-128 model.

---

## 8. Inspecting the attention pattern

```python
# Forward pass with cache
import numpy as np
x = np.array(tokenizer.encode("def train("))
logits, cache = model.forward_with_cache(x)

# cache contains per-layer attention weights
layer_0_attn = cache["layers"][0]["attention_weights"]  # (n_heads, T, T)
print(f"Attention shape: {layer_0_attn.shape}")  # (4, 10, 10)

# Plot head 0
plt.imshow(layer_0_attn[0], cmap='viridis')
plt.colorbar(label='Attention weight')
plt.xlabel('Key position')
plt.ylabel('Query position')
plt.title('Layer 0, Head 0 attention')
plt.show()
```

You should see a **lower-triangular** pattern (causal masking) with
diagonal dominance — the model mostly attends to itself and the immediately
preceding tokens.

---

## 9. Comparing to the untrained model

```python
from tiny_gpt import TinyGPT, TinyGPTConfig

# Build an untrained model with the same architecture
config = TinyGPTConfig(
    vocab_size=512, hidden_dim=128, n_layers=12, n_heads=4, max_seq_len=32
)
untrained = TinyGPT(config)
untrained.init_weights(seed=0)

# Compare outputs
prompt = "def train("
ids = tokenizer.encode(prompt)

trained_out = generate_sample(model, ids, max_new_tokens=30, temperature=0.5)
untrained_out = generate_sample(untrained, ids, max_new_tokens=30, temperature=0.5)

print("Trained:  ", tokenizer.decode(trained_out))
print("Untrained:", tokenizer.decode(untrained_out))
```

The trained model produces real Python tokens (`Float`, `return`, `import`);
the untrained model produces pure noise.

---

## What's next?

- [Tutorial 4: Multilingual Labs](04-multilingual-labs.md)
- [API: TinyGPT](../api/tinygpt.md)
- [Architecture: Request Flow](../architecture/request-flow.md)
