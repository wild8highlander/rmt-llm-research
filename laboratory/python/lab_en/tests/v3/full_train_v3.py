"""
full_train_v3.py — Full training pipeline for TinyGPT v3.

End-to-end training that integrates all Session 6-8 components:
  1. Build a corpus from the project tree (CorpusBuilder v3).
  2. Train a BPE tokenizer with vocab=512 (BPETokenizer v3).
  3. Encode the corpus with the BPE tokenizer.
  4. Train TinyGPT v3 (config_train_4m) for N epochs.
  5. Report final loss + match_rate + sample generation.

This is the "production" training script. For a quick smoke test
(2 layers, 5 epochs, 34 seconds), use smoke_train.py instead.

Usage::

    python3 full_train_v3.py --epochs 30 --vocab 512

Author: Iskhak Hamzatovich Isaev
ORCID:  0009-0003-7299-0701
License: Proprietary — All rights reserved.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


# Make the lab importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from bpe_tokenizer_v3 import BPETokenizerV3
from corpus_builder_v3 import CorpusBuilder, CorpusConfig
from tiny_gpt_trainer_v3 import TrainConfig, TrainerV3
from tiny_gpt_v3 import (
    TinyGPTV3,
    TinyGPTV3Config,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="TinyGPT v3 full training")
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of training epochs (default 30)"
    )
    parser.add_argument("--vocab", type=int, default=512, help="BPE vocab size (default 512)")
    parser.add_argument(
        "--corpus-bytes", type=int, default=200_000, help="Max corpus bytes (default 200K)"
    )
    parser.add_argument(
        "--hidden", type=int, default=128, help="Hidden dim (default 128, use 192 for full 4M)"
    )
    parser.add_argument(
        "--layers", type=int, default=6, help="Number of layers (default 6, use 10 for full 4M)"
    )
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length (default 128)")
    parser.add_argument("--batch", type=int, default=2, help="Effective batch size (default 2)")
    parser.add_argument(
        "--accum", type=int, default=2, help="Gradient accumulation steps (default 2)"
    )
    parser.add_argument("--lr", type=float, default=3e-3, help="Max learning rate (default 3e-3)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=str, default="", help="Path to save trained weights (.npz)")
    args = parser.parse_args()

    print("=" * 70)
    print("TinyGPT v3 — Full Training Pipeline (Session 9)")
    print("=" * 70)

    # 1. Build the corpus.
    print("\n[1/5] Building corpus...")
    t0 = time.time()
    builder = CorpusBuilder(CorpusConfig(max_bytes=args.corpus_bytes))
    corpus_bytes = builder.build()
    print(f"  Corpus: {len(corpus_bytes):,} bytes ({time.time() - t0:.1f}s)")
    print(f"  {builder.summary()}")

    # 2. Train BPE tokenizer.
    print(f"\n[2/5] Training BPE tokenizer (vocab={args.vocab})...")
    t0 = time.time()
    tokenizer = BPETokenizerV3(vocab_size=args.vocab)
    tokenizer.fit(corpus_bytes, verbose=False)
    n_merges = tokenizer.n_merges
    compression = tokenizer.compression_ratio(corpus_bytes)
    print(f"  Merges: {n_merges}, compression: {compression:.2f}× ({time.time() - t0:.1f}s)")

    # 3. Encode corpus.
    print("\n[3/5] Encoding corpus...")
    t0 = time.time()
    token_ids = tokenizer.encode_corpus(corpus_bytes)
    n_train = int(0.9 * len(token_ids))
    train_ids = token_ids[:n_train]
    val_ids = token_ids[n_train:]
    print(
        f"  Tokens: {len(token_ids):,} (train={len(train_ids):,}, "
        f"val={len(val_ids):,}) [{time.time() - t0:.1f}s]"
    )

    # 4. Build model.
    print(f"\n[4/5] Building model (hidden={args.hidden}, layers={args.layers})...")
    cfg = TinyGPTV3Config(
        vocab_size=args.vocab,
        hidden_dim=args.hidden,
        n_layers=args.layers,
        n_heads=max(2, args.hidden // 32),
        n_kv_heads=max(1, (args.hidden // 32) // 2),
        max_seq_len=args.seq_len,
        mlp_ratio=4,
        use_rope=True,
        weight_tying=True,
        label_smoothing=0.1,
        grad_clip_norm=1.0,
        dropout=0.1,
        init_scale=0.02,
        seed=args.seed,
    )
    print(
        f"  Config: vocab={cfg.vocab_size}, hidden={cfg.hidden_dim}, "
        f"layers={cfg.n_layers}, heads={cfg.n_heads}/{cfg.n_kv_heads}, "
        f"seq={cfg.max_seq_len}"
    )
    print(
        f"  Tier 1+2: weight_tying={cfg.weight_tying}, "
        f"label_smoothing={cfg.label_smoothing}, "
        f"dropout={cfg.dropout}, grad_clip={cfg.grad_clip_norm}"
    )
    print(f"  Parameters: {cfg.params_count:,}")
    model = TinyGPTV3(cfg)

    # 5. Train.
    print(f"\n[5/5] Training {args.epochs} epochs (batch={args.batch}×{args.accum})...")
    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch,
        grad_accum_steps=args.accum,
        max_lr=args.lr,
        warmup_ratio=0.1,
        min_lr_ratio=0.1,
        weight_decay=0.01,
        eval_interval=max(1, args.epochs // 5),
        eval_steps=20,
        early_stopping_patience=0,  # disable for full run
        verbose=True,
        seed=args.seed,
    )
    t0 = time.time()
    trainer = TrainerV3(model, train_cfg)
    history = trainer.train(train_ids, val_ids=val_ids)
    elapsed = time.time() - t0

    # Summary.
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Total training time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"Final train loss:    {history['losses'][-1]:.4f}")
    if history["val_losses"]:
        print(f"Final val loss:      {history['val_losses'][-1]:.4f}")
    if history["val_match_rates"]:
        print(f"Final val match_rate: {history['val_match_rates'][-1]:.1%}")
    if history["match_rates"]:
        print(f"Final train match_rate: {history['match_rates'][-1]:.1%}")

    # Sample generation.
    print("\n--- Sample generation ---")
    prompts = [b"def ", b"class ", b"import ", b"return "]
    for prompt_bytes in prompts:
        prompt_ids = np.array(tokenizer.encode(prompt_bytes), dtype=np.int64)
        if len(prompt_ids) == 0:
            continue
        out = model.generate(prompt_ids, max_new_tokens=24, temperature=0.0)
        full_ids = out["full_ids"]
        decoded = tokenizer.decode(full_ids)
        print(f"  Prompt: {prompt_bytes.decode()!r}")
        print(f"  Output: {decoded.decode('utf-8', errors='replace')!r}")
        print()

    # Save weights.
    if args.save:
        print(f"Saving weights to {args.save}...")
        model.save_weights(args.save)

    return 0


if __name__ == "__main__":
    sys.exit(main())
