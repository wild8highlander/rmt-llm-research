"""Chunked v2 training: run N epochs per invocation, resume from checkpoint.

Usage:
    python scripts/train_v2_chunk.py <start_epoch> <end_epoch>

Examples:
    python scripts/train_v2_chunk.py 0 5    # epochs 1-5
    python scripts/train_v2_chunk.py 5 10   # epochs 6-10 (resumes from checkpoint)

Run from the repository root:
    cd /path/to/rmt-llm-research
    python scripts/train_v2_chunk.py 0 5

Optional environment variables:
    TINYGPT_LAB      "lab_en" (default) or "lab_ru"
    TINYGPT_REPO     path to repo root (auto-detected by default)
"""

import json
import math
import os
import sys
import time


# Auto-detect repo root: this script lives in <repo>/scripts/
REPO = os.environ.get(
    "TINYGPT_REPO",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
LAB = os.environ.get("TINYGPT_LAB", "lab_en")
sys.path.insert(0, os.path.join(REPO, f"laboratory/python/{LAB}"))

import numpy as np
from tiny_gpt import TinyGPT, TinyGPTConfig
from tiny_gpt_trainer import (
    AdamState,
    BPETokenizer,
    TrainConfig,
    backward,
    build_corpus,
    cross_entropy_loss,
    evaluate_match_rate,
    forward_with_cache,
    generate_sample,
    load_trained_model,
    make_dataset_tokens,
)


LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_v2_log.txt")
MODELS_DIR = os.path.join(REPO, f"laboratory/python/{LAB}/results/models")
WEIGHTS_PATH = os.path.join(MODELS_DIR, "tiny_gpt_trained.npz")
BPE_PATH = os.path.join(MODELS_DIR, "tiny_gpt_bpe.json")
HISTORY_PATH = os.path.join(MODELS_DIR, "training_history.json")


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


start_ep = int(sys.argv[1]) if len(sys.argv) > 1 else 0
end_ep = int(sys.argv[2]) if len(sys.argv) > 2 else 5
total_epochs = 30  # for cosine schedule

log(f"=== chunk epochs {start_ep + 1}-{end_ep}/{total_epochs} ===")

# Config (same as full run)
cfg = TrainConfig(
    seq_len=32,
    stride=64,
    batch_size=8,
    epochs=end_ep - start_ep,  # only this many epochs in this chunk
    lr=5e-4,
    weight_decay=1e-5,
    lr_schedule="cosine",
    warmup_epochs=3,
    min_lr_ratio=0.1,
    eval_every=3,
    eval_samples=64,
    checkpoint_every=0,  # we handle checkpointing ourselves
    vocab_size=512,
    bpe_merges=256,
    max_train_tokens=40000,
    hidden_dim=128,
    n_layers=12,
    n_heads=4,
    mlp_ratio=4,
    use_layernorm=True,
    use_mlp=True,
    activation="gelu",
    output_dir="results/models",
)

# Build corpus + BPE (or load existing BPE)
corpus = build_corpus(REPO)
if os.path.exists(BPE_PATH):
    tokenizer = BPETokenizer.load(BPE_PATH)
    log(f"  loaded BPE from {BPE_PATH} ({tokenizer.n_merges} merges)")
else:
    tokenizer = BPETokenizer(vocab_size=cfg.vocab_size)
    log(f"  training BPE ({cfg.bpe_merges} merges, corpus={len(corpus):,}B)...")
    t0 = time.time()
    tokenizer.train(corpus, target_merges=cfg.bpe_merges)
    log(f"  BPE trained in {time.time() - t0:.1f}s")
    tokenizer.save(BPE_PATH)

# Tokenize corpus
all_ids = tokenizer.encode(corpus.decode("utf-8", errors="replace"))
if cfg.max_train_tokens and len(all_ids) > cfg.max_train_tokens:
    offset = (len(all_ids) - cfg.max_train_tokens) // 2
    all_ids = all_ids[offset : offset + cfg.max_train_tokens]
dataset = make_dataset_tokens(all_ids, seq_len=cfg.seq_len, stride=cfg.stride)
n_windows = len(dataset)
log(f"  windows={n_windows:,}  tokens={len(all_ids):,}")

# Train/eval split (deterministic)
rng = np.random.default_rng(cfg.seed)
perm = rng.permutation(n_windows)
n_eval = max(1, n_windows // 10)
eval_set = dataset[perm[:n_eval]]
train_set = dataset[perm[n_eval:]]
n_train = len(train_set)

# Build model
model_cfg = TinyGPTConfig(
    vocab_size=cfg.vocab_size,
    hidden_dim=cfg.hidden_dim,
    n_layers=cfg.n_layers,
    n_heads=cfg.n_heads,
    max_seq_len=cfg.max_seq_len,
    mlp_ratio=cfg.mlp_ratio,
    use_layernorm=cfg.use_layernorm,
    use_mlp=cfg.use_mlp,
    activation=cfg.activation,
    seed=cfg.seed,
)

if start_ep > 0 and os.path.exists(WEIGHTS_PATH):
    model = TinyGPT.load_weights(WEIGHTS_PATH)
    log(f"  resumed from {WEIGHTS_PATH}")
else:
    model = TinyGPT(model_cfg)
    log(f"  fresh model, params={model_cfg.params_count:,}")

optimizer = AdamState(
    model,
    lr=cfg.lr,
    weight_decay=cfg.weight_decay,
    lr_schedule=cfg.lr_schedule,
    warmup_epochs=cfg.warmup_epochs,
    total_epochs=total_epochs,
    min_lr_ratio=cfg.min_lr_ratio,
)

# Load history if exists
history = []
if os.path.exists(HISTORY_PATH):
    try:
        with open(HISTORY_PATH) as f:
            history = json.load(f).get("history", [])
    except Exception:
        history = []

# Initial baseline (only on first chunk)
if start_ep == 0:
    baseline = evaluate_match_rate(model, eval_set, n_samples=cfg.eval_samples)
    log(f"  baseline mr={baseline['match_rate']:.3%} loss={baseline['loss']:.4f}")

# Training loop for this chunk
rng_shuffle = np.random.default_rng(cfg.seed + 1)
# Advance RNG state by start_ep permutations to maintain shuffle consistency
for _ in range(start_ep):
    _ = rng_shuffle.permutation(n_train)

t0 = time.time()
n_batches_per_epoch = max(1, (n_train + cfg.batch_size - 1) // cfg.batch_size)

for epoch_offset in range(end_ep - start_ep):
    epoch = start_ep + epoch_offset  # global epoch index
    order = rng_shuffle.permutation(n_train)
    epoch_loss = 0.0
    epoch_steps = 0
    grad_norm_acc = 0.0
    batch_idx = 0

    for start in range(0, n_train, cfg.batch_size):
        bi = order[start : start + cfg.batch_size]
        if len(bi) == 0:
            continue
        frac_epoch = epoch + batch_idx / max(1, n_batches_per_epoch)
        optimizer.update_epoch_progress(frac_epoch)

        agg = None
        for sample_i in bi:
            row = train_set[sample_i]
            x = row[:-1]
            y = int(row[-1])
            _, cache = forward_with_cache(model, x)
            grads = backward(model, cache, np.array([y]))
            if agg is None:
                agg = {
                    "token_emb": grads["token_emb"].copy(),
                    "pos_emb": grads["pos_emb"].copy(),
                    "lm_head": grads["lm_head"].copy(),
                    "ln_f_gamma": grads["ln_f_gamma"].copy(),
                    "ln_f_beta": grads["ln_f_beta"].copy(),
                    "layers": [
                        {k: (v.copy() if v is not None else None) for k, v in lg.items()}
                        for lg in grads["layers"]
                    ],
                }
            else:
                agg["token_emb"] += grads["token_emb"]
                agg["pos_emb"] += grads["pos_emb"]
                agg["lm_head"] += grads["lm_head"]
                agg["ln_f_gamma"] += grads["ln_f_gamma"]
                agg["ln_f_beta"] += grads["ln_f_beta"]
                for li, lg in enumerate(grads["layers"]):
                    for k in lg:
                        if lg[k] is None:
                            continue
                        if agg["layers"][li][k] is None:
                            agg["layers"][li][k] = lg[k].copy()
                        else:
                            agg["layers"][li][k] += lg[k]
            last_logits = cache.logits[-1]
            epoch_loss += cross_entropy_loss(last_logits, y)
            epoch_steps += 1
        batch_idx += 1

        inv = 1.0 / max(1, len(bi))
        agg["token_emb"] *= inv
        agg["pos_emb"] *= inv
        agg["lm_head"] *= inv
        agg["ln_f_gamma"] *= inv
        agg["ln_f_beta"] *= inv
        for lg in agg["layers"]:
            for k in lg:
                if lg[k] is not None:
                    lg[k] *= inv

        gnorm = math.sqrt(
            float(np.sum(agg["token_emb"] ** 2))
            + float(np.sum(agg["pos_emb"] ** 2))
            + float(np.sum(agg["lm_head"] ** 2))
            + float(np.sum(agg["ln_f_gamma"] ** 2))
            + float(np.sum(agg["ln_f_beta"] ** 2))
            + sum(float(np.sum(v**2)) for lg in agg["layers"] for v in lg.values() if v is not None)
        )
        grad_norm_acc += gnorm
        optimizer.step(model, agg)

    avg_loss = epoch_loss / max(1, epoch_steps)
    avg_gnorm = grad_norm_acc / max(1, n_batches_per_epoch)
    record = {
        "epoch": epoch + 1,
        "loss": avg_loss,
        "grad_norm": avg_gnorm,
        "lr": optimizer.lr,
    }
    if (epoch + 1) % cfg.eval_every == 0 or epoch_offset == (end_ep - start_ep) - 1:
        ev = evaluate_match_rate(model, eval_set, n_samples=cfg.eval_samples)
        record["match_rate"] = ev["match_rate"]
        record["eval_loss"] = ev["loss"]
    history.append(record)
    msg = f"  ep{epoch + 1:>2d}/{total_epochs}  loss={avg_loss:.4f}  gnorm={avg_gnorm:.3f}  lr={optimizer.lr:.2e}"
    if "match_rate" in record:
        msg += f"  mr={record['match_rate']:.3%}"
    log(msg)

# Save weights after this chunk
model.save_weights(WEIGHTS_PATH)
log(f"  saved weights to {WEIGHTS_PATH}")

# Save updated history
final = evaluate_match_rate(model, eval_set, n_samples=max(cfg.eval_samples, 128))
history_record = {
    "config": {
        **cfg.__dict__,
        "epochs": total_epochs,
        "n_layers": cfg.n_layers,
        "hidden_dim": cfg.hidden_dim,
    },
    "epochs_run": end_ep,
    "total_epochs_target": total_epochs,
    "elapsed_seconds_chunk": time.time() - t0,
    "final_match_rate": final["match_rate"],
    "final_loss": final["loss"],
    "history": history,
    "weights_path": WEIGHTS_PATH,
    "bpe_path": BPE_PATH,
    "params_count": model_cfg.params_count,
    "n_merges": tokenizer.n_merges,
    "corpus_bytes": len(corpus),
    "n_train_windows": n_train,
    "n_eval_windows": n_eval,
}
with open(HISTORY_PATH, "w") as f:
    json.dump(history_record, f, indent=2, default=str)

log(f"  chunk done in {time.time() - t0:.1f}s, final mr={final['match_rate']:.3%}")

# If this is the last chunk, generate samples
if end_ep >= total_epochs:
    log("\n=== generation samples ===")
    model2, tok2 = load_trained_model(WEIGHTS_PATH, BPE_PATH)
    for prompt in ["The RMT-LLM", "TinyGPT", "Spectral analysis", "import numpy"]:
        for temp in [0.0, 0.5, 1.0]:
            out = generate_sample(
                model2, prompt, tok2, max_new_tokens=48, temperature=temp, seed=42
            )
            log(f"  prompt={prompt!r} temp={temp} -> {out[:200]!r}")
    log("=== done ===")

log(f"=== chunk {start_ep + 1}-{end_ep} complete ===")
