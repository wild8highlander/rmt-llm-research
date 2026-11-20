#!/usr/bin/env python3
"""rcda6_ppl.py — the abilities PPL parity check (Gemma3-1B-IT RCDA-6).

Teacher-forced perplexity of the 5 control texts under the base model and
under the trained edit. Writes ppl_check.json in the same format as the
MNN-Pocket kit (base / trained arms; per-text and mean).

Usage: rcda6_ppl.py --model DIR --trained-state PATH --out ppl_check.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "Code"))
import gemma_lab as L                                    # noqa: E402


@torch.no_grad()
def ppl_per_text(model, tok, texts) -> list[float]:
    out = []
    for text in texts:
        enc = L.encode_prompt(tok, text)
        logits = model(enc).logits[0].float()
        logp = torch.log_softmax(logits, dim=-1)
        n = logits.shape[0] - 1
        nll = -logp[torch.arange(n), enc[0, 1:]].mean()
        out.append(round(float(torch.exp(nll)), 1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-1b-it")
    ap.add_argument("--trained-state", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM

    tok = L.load_tokenizer(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.eval()
    ctrl_texts = [t for _, t in L.load_control_texts()]

    res = {}
    res["base"] = {"ppl_control_mean": round(float(np.mean(
        ppl_per_text(model, tok, ctrl_texts))), 1),
        "ppl_per_text": ppl_per_text(model, tok, ctrl_texts)}

    trained = load_file(args.trained_state)
    with torch.no_grad():
        for k, w in trained.items():
            dict(model.named_parameters())[k].data.copy_(w)
    res["trained"] = {"ppl_control_mean": round(float(np.mean(
        ppl_per_text(model, tok, ctrl_texts))), 1),
        "ppl_per_text": ppl_per_text(model, tok, ctrl_texts)}

    json.dump(res, open(args.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
