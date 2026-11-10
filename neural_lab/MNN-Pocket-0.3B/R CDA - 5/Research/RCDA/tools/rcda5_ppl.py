#!/usr/bin/env python3
"""RCDA-5 ability check: perplexity on control good texts for the three arms.
Same texts as eval_b's KL control (5 good completions)."""
import json

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

WORK = "/home/z/my-project/rcda_work"
OUT = f"{WORK}/rcda5"
MODEL_DIR = f"{WORK}/model"

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32)
model.eval()

# control texts: greedy completions of the first 5 good prompts (base arm)
import sys
sys.path.insert(0, "/home/z/my-project/scripts")
from rcda4_eval_b import gen_ids, prompts_from_sentinel  # noqa: E402

bad, good = prompts_from_sentinel()
texts = [tok.decode(gen_ids(model, tok, p), skip_special_tokens=True)
         for _, p, _ in good[:5]]
texts = [t for t in texts if t.strip()] or [p for _, p, _ in good[:5]]

res = {}
for tag, path in (("base", f"{OUT}/base_state.safetensors"),
                  ("surgical", f"{OUT}/surgical_state.safetensors"),
                  ("trained", f"{OUT}/trained_state.safetensors")):
    sd = load_file(path)
    model.load_state_dict(sd, strict=True)
    ppls = []
    for t in texts:
        enc = tok(t, return_tensors="pt")
        with torch.no_grad():
            nll = -model(**enc).logits.log_softmax(-1)
        tgt = enc.input_ids[0, 1:]
        ppls.append(float(nll[0, :-1].gather(1, tgt.unsqueeze(1)).mean().exp()))
    res[tag] = {"ppl_control_mean": round(float(np.mean(ppls)), 1),
                "ppl_per_text": [round(x, 1) for x in ppls]}
    print(tag, res[tag])

json.dump(res, open(f"{OUT}/ppl_check.json", "w"), indent=1)
