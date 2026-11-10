#!/usr/bin/env python3
"""RCDA-5 prepare: convert RCDA-2 surgical checkpoint and base model into
state_dict safetensors (with lm_head.weight) that rcda4_eval_b.py can load,
and verify the surgical diff matches the RCDA-2 recipe."""
import json
import torch
from safetensors.torch import load_file, save_file

WORK = "/home/z/my-project/rcda_work"
OUT = f"{WORK}/rcda5"

base_sd = load_file(f"{WORK}/model/model.safetensors")
ed_sd = load_file(f"{WORK}/edited_model_v2/model.safetensors")

# tied lm_head: verify base lm_head == embed, then clone into ed
assert torch.equal(base_sd["lm_head.weight"], base_sd["model.embed_tokens.weight"]), \
    "lm_head is not tied to embed in base state"
if "lm_head.weight" not in ed_sd:
    ed_sd["lm_head.weight"] = ed_sd["model.embed_tokens.weight"].clone()
assert set(ed_sd) == set(base_sd), (
    f"key mismatch: {set(base_sd) ^ set(ed_sd)}")

save_file(ed_sd, f"{OUT}/surgical_state.safetensors")
save_file({k: v.clone() for k, v in base_sd.items()},
          f"{OUT}/base_state.safetensors")

# verify surgical diff: which tensors changed, how many output columns
report = {}
for k in base_sd:
    if not torch.equal(base_sd[k], ed_sd[k]):
        d = (ed_sd[k].float() - base_sd[k].float())
        cols = int((d.abs().sum(dim=0) > 0).sum())
        report[k] = {"cols_changed": cols,
                     "n_weights": int((d != 0).sum()),
                     "rel_fro": round(float(torch.linalg.norm(d)
                                           / torch.linalg.norm(base_sd[k].float())), 5)}
print(json.dumps(report, indent=1))
json.dump(report, open(f"{OUT}/surgical_diff.json", "w"), indent=1)
print("prepared: surgical_state.safetensors, base_state.safetensors")
