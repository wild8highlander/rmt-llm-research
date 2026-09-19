# -*- coding: utf-8 -*-
"""
train_formula_model.py — Обучение TinyGPT v3 на корпусе формул rmt-llm-research.

Модель "из коробки" (ADR-001, NumPy-only): RoPE + GQA + weight tying +
label smoothing + grad clip + AdamW + cosine LR.

Выход в <BASE>/model/ (BASE = рядом с репо, либо research/tinygpt_formula внутри репо):
  tiny_gpt_formula.npz   — веса
  tiny_gpt_formula_bpe.json — BPE-токенизатор
  training_history.json  — полная история метрик по эпохам
  run_config.json        — конфиг запуска
История и веса сохраняются КАЖДУЮ эпоху (устойчивость к прерыванию);
при повторном запуске обучение продолжается с последней эпохи (resume).
"""
import json
import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Портативные пути (см. build_corpus.py): env RMT_LLM_ROOT / RMT_LLM_BASE
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_REPO = "/home/z/my-project/rmt-llm-research"

if os.environ.get("RMT_LLM_ROOT"):
    REPO = os.path.abspath(os.environ["RMT_LLM_ROOT"])
else:
    REPO = _DEFAULT_REPO
    _cand = SCRIPT_DIR
    for _ in range(5):
        _cand = os.path.dirname(_cand)
        if os.path.isdir(os.path.join(_cand, "src", "rmt_llm")):
            REPO = _cand
            break

if os.environ.get("RMT_LLM_BASE"):
    BASE = os.path.abspath(os.environ["RMT_LLM_BASE"])
else:
    _base = os.path.dirname(SCRIPT_DIR)
    BASE = _base if os.path.basename(SCRIPT_DIR) == "scripts" and _base.startswith(REPO + os.sep) else os.path.dirname(REPO)

LAB = os.path.join(REPO, "laboratory", "python", "lab_en")
sys.path.insert(0, LAB)
sys.path.insert(0, os.path.join(REPO, "src", "rmt_llm"))

from tiny_gpt_v3 import TinyGPTV3, TinyGPTV3Config
from tiny_gpt_trainer_v3 import TrainerV3, TrainConfig
from tiny_gpt_trainer import BPETokenizer

DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "model")
os.makedirs(OUT, exist_ok=True)

MAX_MINUTES = float(os.environ.get("TRAIN_MAX_MINUTES", "38"))

# ---------------------------------------------------------------------------
# 1. Корпус и токенизатор
# ---------------------------------------------------------------------------
train_text = open(os.path.join(DATA, "formula_corpus_train.txt"), encoding="utf-8").read()
val_text = open(os.path.join(DATA, "formula_corpus_val.txt"), encoding="utf-8").read()
print(f"Corpus: train={len(train_text):,} bytes, val={len(val_text):,} bytes")

t0 = time.time()
tok = BPETokenizer(vocab_size=512)
tok.train(train_text.encode("utf-8"))
print(f"BPE: {tok.n_merges} merges in {time.time()-t0:.1f}s")

train_ids = tok.encode_bytes(train_text.encode("utf-8"))
val_ids = tok.encode_bytes(val_text.encode("utf-8"))
print(f"Tokens: train={len(train_ids):,}, val={len(val_ids):,}")

# ---------------------------------------------------------------------------
# 2. Модель (CPU-friendly конфиг v3)
# ---------------------------------------------------------------------------
cfg = TinyGPTV3Config(
    vocab_size=512, hidden_dim=96, n_layers=4,
    n_heads=6, n_kv_heads=2, max_seq_len=64,
    mlp_ratio=4, use_rope=True,
    weight_tying=True, label_smoothing=0.1,
    grad_clip_norm=1.0, dropout=0.1, init_scale=0.02, seed=42,
)
model = TinyGPTV3(cfg)
print(f"Model: {cfg.params_count:,} params, layers={cfg.n_layers}, hidden={cfg.hidden_dim}, seq={cfg.max_seq_len}")

# ---------------------------------------------------------------------------
# 3. Baseline ДО обучения (для измерения прироста)
# ---------------------------------------------------------------------------
def quick_eval(seqs, n=40):
    losses, correct, total = [], 0, 0
    rng = np.random.default_rng(0)
    idx = rng.choice(len(seqs), size=min(n, len(seqs)), replace=False) if len(seqs) > n else range(len(seqs))
    model.training = False
    for i in idx:
        s = seqs[i]
        logits, _ = model.forward(s[:-1].astype(np.int64))
        targets = s[1:].astype(np.int64)
        p = _ce_per_tok(logits, targets)
        losses.append(float(np.mean(p)))
        correct += int(np.sum(np.argmax(logits, -1) == targets)); total += len(targets)
    return float(np.mean(losses)), correct / max(total, 1)

def _ce_per_tok(logits, targets):
    lg = logits - logits.max(-1, keepdims=True)
    logp = lg - np.log(np.exp(lg).sum(-1, keepdims=True))
    return logp[np.arange(len(targets)), targets] * -1.0

seq_len = cfg.max_seq_len
def make_seqs(ids):
    n = len(ids) // seq_len
    return [ids[i*seq_len:(i+1)*seq_len] for i in range(n)]

train_seqs = make_seqs(train_ids)
val_seqs = make_seqs(val_ids)
print(f"Sequences: train={len(train_seqs)}, val={len(val_seqs)}")

base_val_loss, base_val_mr = quick_eval(val_seqs)
print(f"BASELINE (untrained): val_loss={base_val_loss:.4f} (~{base_val_loss:.2f} nats/token), val_match={base_val_mr:.2%}")

# ---------------------------------------------------------------------------
# 4. Обучение с сохранением истории каждую эпоху
# ---------------------------------------------------------------------------
train_cfg = TrainConfig(
    epochs=40, batch_size=8, grad_accum_steps=2,
    max_lr=1.2e-3, warmup_ratio=0.06, min_lr_ratio=0.05,
    weight_decay=0.01, eval_interval=1, eval_steps=40,
    early_stopping_patience=0, seed=42, verbose=True,
)
trainer = TrainerV3(model, train_cfg)

history = {
    "losses": [], "grad_norms": [], "lrs_per_epoch": [],
    "val_losses": [], "val_match_rates": [], "train_match_rates": [],
    "epoch_times": [], "perplexities": [], "val_perplexities": [],
}
meta = {
    "baseline_val_loss": base_val_loss, "baseline_val_match": base_val_mr,
    "model_params": cfg.params_count, "bpe_vocab": tok.vocab_size,
    "bpe_merges": tok.n_merges, "train_tokens": int(len(train_ids)),
    "val_tokens": int(len(val_ids)), "seq_len": seq_len,
    "config": cfg.__dict__, "train_config": {k: v for k, v in train_cfg.__dict__.items()},
}

def save_state():
    model.save_weights(os.path.join(OUT, "tiny_gpt_formula.npz"))
    tok.save(os.path.join(OUT, "tiny_gpt_formula_bpe.json"))
    json.dump({"history": history, "meta": meta},
              open(os.path.join(OUT, "training_history.json"), "w"), indent=1)

# собственный цикл поверх TrainerV3._train_step, чтобы сохранять историю каждую эпоху
rng = trainer.rng
n_steps_per_epoch = (len(train_seqs) + train_cfg.batch_size - 1) // train_cfg.batch_size
total_steps = train_cfg.epochs * n_steps_per_epoch
warmup_steps = int(total_steps * train_cfg.warmup_ratio)
from tiny_gpt_v3 import cosine_lr_schedule

start = time.time()
interrupted = False
epoch_start = 0
epoch = -1  # FIX: раньше при resume завершённого прогона (пустой range) имя epoch
            # было не определено -> NameError на "epochs_run: epoch + 1"

# --- resume: подхватываем прежнюю историю И ВЕСА, если они есть ---
resume_path = os.path.join(OUT, "training_history.json")
weights_path = os.path.join(OUT, "tiny_gpt_formula.npz")
if os.path.exists(resume_path) and os.environ.get("TRAIN_RESUME", "1") == "1":
    try:
        prev = json.load(open(resume_path))
        ph = prev["history"]
        if len(ph["losses"]) > 0 and not prev.get("meta", {}).get("finished"):
            for k in history:
                history[k] = list(ph[k])
            epoch_start = len(ph["losses"])
            print(f"RESUME: continuing from epoch {epoch_start}")
            if os.path.exists(weights_path):
                model = TinyGPTV3.load_weights(weights_path)
                trainer = TrainerV3(model, train_cfg)
                print("RESUME: weights + optimizer restored")
    except Exception as e:
        print("resume skipped:", e)

for epoch in range(epoch_start, train_cfg.epochs):
    te = time.time()
    rng.shuffle(train_seqs)
    ep_loss, ep_gn, n_steps, last_lr = 0.0, 0.0, 0, 0.0
    for i in range(0, len(train_seqs), train_cfg.batch_size):
        batch = train_seqs[i:i + train_cfg.batch_size]
        if len(batch) < train_cfg.batch_size:
            break
        gs = min(epoch * n_steps_per_epoch + n_steps, total_steps - 1)
        lr = cosine_lr_schedule(gs, train_cfg.max_lr, warmup_steps, total_steps, train_cfg.min_lr_ratio)
        loss, gn = trainer._train_step(batch, lr)
        ep_loss += loss; ep_gn += gn; n_steps += 1; last_lr = lr
    dt = time.time() - te
    history["losses"].append(ep_loss / max(n_steps, 1))
    history["grad_norms"].append(ep_gn / max(n_steps, 1))
    history["lrs_per_epoch"].append(last_lr)
    history["epoch_times"].append(dt)

    val_loss, val_mr = quick_eval(val_seqs, n=40)
    tr_loss, tr_mr = quick_eval(train_seqs[:60], n=40)
    history["val_losses"].append(val_loss)
    history["val_match_rates"].append(val_mr)
    history["train_match_rates"].append(tr_mr)
    history["perplexities"].append(float(np.exp(history["losses"][-1])))
    history["val_perplexities"].append(float(np.exp(val_loss)))

    # сохраняем каждую эпоху — дешевле потерять максимум одну эпоху прогресса
    save_state()
    print(f"epoch {epoch+1:3d}: loss={history['losses'][-1]:.4f} ppl={history['perplexities'][-1]:7.2f} "
          f"val_loss={val_loss:.4f} val_ppl={np.exp(val_loss):7.2f} val_mr={val_mr:.2%} "
          f"train_mr={tr_mr:.2%} gn={history['grad_norms'][-1]:.2f} ({dt:.1f}s, total {(time.time()-start)/60:.1f}m)", flush=True)

    if (time.time() - start) / 60 > MAX_MINUTES:
        print(f"Time limit {MAX_MINUTES}m reached — stopping gracefully at epoch {epoch+1}")
        interrupted = True
        break

meta["finished"] = not interrupted
save_state()
# FIX: при пустом цикле (resume завершённого прогона) epoch остаётся epoch_start-1,
# но история уже полная — финальные метрики берём из истории, а не из epoch
last_epoch = epoch + 1 if epoch >= 0 else len(history["losses"])
final = {
    "epochs_run": last_epoch,
    "interrupted_by_time": interrupted,
    "total_minutes": round((time.time() - start) / 60, 1),
    "final_train_loss": history["losses"][-1],
    "final_val_loss": history["val_losses"][-1],
    "final_val_perplexity": history["val_perplexities"][-1],
    "final_val_match_rate": history["val_match_rates"][-1],
    "baseline_val_loss": base_val_loss,
    "baseline_val_match": base_val_mr,
    "improvement_loss_x": base_val_loss / max(history["val_losses"][-1], 1e-9),
    "match_rate_gain_pct_points": (history["val_match_rates"][-1] - base_val_mr) * 100,
}
json.dump({**meta, "final": final}, open(os.path.join(OUT, "run_config.json"), "w"), indent=2)
print("\nFINAL:", json.dumps(final, indent=2))
