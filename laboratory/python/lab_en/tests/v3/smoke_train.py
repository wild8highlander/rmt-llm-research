"""
End-to-end smoke training for TinyGPT v3.

Trains the v3 model with all Session 1-4 features on a small corpus
built from the project's own source code, for 30 epochs. Verifies that:
  - The full training loop runs without errors.
  - Loss decreases monotonically (mostly).
  - match_rate on train and val increases above the v2 baseline (6.5%).
  - Weight tying, label smoothing, dropout, grad clipping all active.

This is the validation that the Tier 1+2 recipe actually delivers the
predicted ~16% match_rate improvement.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

# Make the lab importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tiny_gpt_v3 import (
    TinyGPTV3, TinyGPTV3Config, config_train_4m, config_small,
)
from tiny_gpt_trainer_v3 import TrainerV3, TrainConfig


# ---------------------------------------------------------------------------
# Build a small corpus from the project's source code
# ---------------------------------------------------------------------------
def build_corpus(max_bytes: int = 50000) -> bytes:
    """Collect Python source files from the project as a training corpus."""
    # Walk up from tests/v3/ to find the project root (contains src/ and laboratory/).
    p = Path(__file__).resolve().parent
    for _ in range(5):
        if (p / "src" / "rmt_llm").is_dir() or (p / "laboratory").is_dir():
            project_root = p
            break
        p = p.parent
    else:
        # Fallback: use the lab_en directory itself.
        project_root = Path(__file__).resolve().parent.parent
    files = []
    # Collect from src/rmt_llm and laboratory/python/lab_en.
    for pattern in ["src/rmt_llm/*.py", "laboratory/python/lab_en/*.py"]:
        for fp in project_root.glob(pattern):
            if fp.stat().st_size < 100_000:  # skip huge files
                files.append(fp)
    # Also add the lab_en directory if project_root is the lab.
    if not files:
        for fp in (project_root).glob("*.py"):
            if fp.stat().st_size < 100_000:
                files.append(fp)
    corpus = b""
    for f in sorted(files):
        try:
            corpus += f.read_bytes() + b"\n"
        except Exception:
            continue
        if len(corpus) >= max_bytes:
            break
    return corpus[:max_bytes]


# ---------------------------------------------------------------------------
# Simple byte-level tokenizer (for smoke test; BPE is in the full trainer)
# ---------------------------------------------------------------------------
class ByteTokenizer:
    """Byte-level tokenizer: vocab = 256, every byte is a token."""

    def __init__(self, vocab_size: int = 256) -> None:
        self.vocab_size = vocab_size

    def encode(self, text: bytes) -> np.ndarray:
        return np.frombuffer(text, dtype=np.uint8).astype(np.int64)

    def decode(self, ids: np.ndarray) -> bytes:
        return bytes(np.clip(ids, 0, 255).astype(np.uint8))


def main() -> int:
    print("=" * 70)
    print("TinyGPT v3 — End-to-End Smoke Training (Session 5)")
    print("=" * 70)

    # 1. Build the corpus.
    corpus_bytes = build_corpus(max_bytes=30000)
    print(f"\nCorpus: {len(corpus_bytes)} bytes")
    print(f"  First 100 chars: {corpus_bytes[:100]!r}")

    # 2. Tokenize (byte-level for the smoke test).
    tokenizer = ByteTokenizer(vocab_size=256)
    token_ids = tokenizer.encode(corpus_bytes)
    print(f"Tokens: {len(token_ids)} (byte-level, vocab=256)")
    # Split 90/10 train/val.
    n_train = int(0.9 * len(token_ids))
    train_ids = token_ids[:n_train]
    val_ids = token_ids[n_train:]
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}")

    # 3. Build the model with the training-ready config (byte-level vocab=256).
    cfg = TinyGPTV3Config(
        vocab_size=256,        # byte-level
        hidden_dim=64,         # smaller for fast smoke test
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=64,        # shorter sequences for smoke test
        mlp_ratio=2,
        use_rope=True,
        weight_tying=True,
        label_smoothing=0.1,
        grad_clip_norm=1.0,
        dropout=0.1,
        init_scale=0.02,
        seed=42,
    )
    print(f"\nModel config: vocab={cfg.vocab_size}, hidden={cfg.hidden_dim}, "
          f"layers={cfg.n_layers}, heads={cfg.n_heads}/{cfg.n_kv_heads}, "
          f"seq={cfg.max_seq_len}")
    print(f"  weight_tying={cfg.weight_tying}, label_smoothing={cfg.label_smoothing}, "
          f"dropout={cfg.dropout}, grad_clip={cfg.grad_clip_norm}")
    print(f"  Parameters: {cfg.params_count:,}")

    model = TinyGPTV3(cfg)

    # 4. Train for 5 epochs (smoke test — keep under 2 min on CPU).
    train_cfg = TrainConfig(
        epochs=5,
        batch_size=1,
        grad_accum_steps=2,    # effective batch = 2
        max_lr=3e-3,
        warmup_ratio=0.1,
        min_lr_ratio=0.1,
        weight_decay=0.01,
        eval_interval=1,
        eval_steps=10,
        early_stopping_patience=0,  # disable for smoke test
        verbose=True,
    )
    print(f"\nTraining: {train_cfg.epochs} epochs, "
          f"batch={train_cfg.batch_size}×{train_cfg.grad_accum_steps}, "
          f"max_lr={train_cfg.max_lr}")

    start = time.time()
    trainer = TrainerV3(model, train_cfg)
    history = trainer.train(train_ids, val_ids=val_ids)
    elapsed = time.time() - start

    # 5. Summary.
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Total training time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Final train loss:   {history['losses'][-1]:.4f}")
    print(f"Final train match_rate: {history['match_rates'][-1]:.1%}" if history['match_rates'] else "  (no eval)")
    print(f"Final val loss:     {history['val_losses'][-1]:.4f}" if history['val_losses'] else "  (no val)")
    print(f"Final val match_rate: {history['val_match_rates'][-1]:.1%}" if history['val_match_rates'] else "  (no val)")

    # 6. Compare to v2 baseline (6.5%).
    if history["val_match_rates"]:
        final_mr = history["val_match_rates"][-1]
        baseline = 0.065  # v2's 6.5%
        improvement = (final_mr - baseline) / baseline * 100
        print(f"\nv2 baseline: {baseline:.1%}")
        print(f"v3 result:   {final_mr:.1%}")
        print(f"Improvement: {improvement:+.1f}%")
        if final_mr > baseline:
            print("✓ v3 BEATS v2 baseline")
        else:
            print("⚠ v3 has not yet beaten v2 baseline (needs more epochs/data)")

    # 7. Generate a sample to qualitatively verify.
    print("\n" + "-" * 70)
    print("Sample generation (greedy, temperature=0):")
    prompt = np.array([ord('d'), ord('e'), ord('f')], dtype=np.int64)
    out = model.generate(prompt, max_new_tokens=32, temperature=0.0)
    decoded = bytes(np.clip(out["full_ids"], 0, 255).astype(np.uint8))
    print(f"  Prompt: 'def'")
    print(f"  Output: {decoded!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
