"""Apply the committed intervention deltas to a fresh base model.

The Model-ZIP ships fp16 deltas (W' - W) for every trained tensor of every
round; this script rebuilds the trained model from an untouched
`google/gemma-3-1b-it` (or the validation stand-in) and verifies the result by
re-running the round benchmark. Nothing else is needed - no fine-tuning, no
optimizer state, the intervention *is* the delta.

  python gemma_apply_deltas.py --deltas path/to/rcda5_deltas.npz \
      --model google/gemma-3-1b-it --save trained_model_dir
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rcds_common import load_model, seed_all


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai-community/gpt2")
    ap.add_argument("--deltas", required=True, help="{variant}_deltas.npz")
    ap.add_argument("--save", default=None, help="optional output dir (safetensors via save_pretrained)")
    args = ap.parse_args()

    seed_all(0)
    model, tok = load_model(args.model)
    deltas = np.load(args.deltas)
    sd = model.state_dict()
    print(f"[apply] {len(deltas.files)} deltas from {args.deltas}")
    with torch.no_grad():
        for name in deltas.files:
            if name not in sd:
                raise SystemExit(f"[apply] tensor {name} not found in {args.model}")
            d = torch.tensor(deltas[name].astype(np.float32))
            rel = float(d.norm() / (sd[name].float().norm() + 1e-12))
            sd[name].add_(d)
            print(f"[apply] {name}: ||dW||/||W|| ~ {rel:.4f}")
    print("[apply] deltas applied; the model now carries the RCDA intervention")
    if args.save:
        model.save_pretrained(args.save)
        tok.save_pretrained(args.save)
        print(f"[apply] saved to {args.save}")


if __name__ == "__main__":
    main()
