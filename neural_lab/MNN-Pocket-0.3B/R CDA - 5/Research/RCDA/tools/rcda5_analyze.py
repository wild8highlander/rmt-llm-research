#!/usr/bin/env python3
"""RCDA-5 analyze: spectral forensics of the SGD-trained edit.
For each of the 3 trained o_proj matrices: relative size of the change,
Tracy-Widom verdict (same control as eval_b), cosine between the change's
top left singular direction and the loop direction v, and — the key number —
how much of the change lives in the v-subspace: cos(d[:, S_top], v) where
S_top are the 128 columns with the largest change norm. Appends "analysis"
to training_log.json."""
import json

import numpy as np
import torch
from safetensors.torch import load_file

sys_path = "/home/z/my-project/rcda_work/Code"
import sys
sys.path.insert(0, sys_path)
from rmt_intervention import spectral_tw_control  # noqa: E402

WORK = "/home/z/my-project/rcda_work"
OUT = f"{WORK}/rcda5"
V = {15: f"{WORK}/rcda4/partB/v_L15.npy",
     16: f"{WORK}/rcda4/partB/v_L16.npy",
     18: f"{WORK}/rcda4/partB/v_L18.npy"}
KEYS = {15: "model.layers.15.self_attn.o_proj.weight",
        16: "model.layers.16.self_attn.o_proj.weight",
        18: "model.layers.18.self_attn.o_proj.weight"}

base_sd = load_file(f"{OUT}/base_state.safetensors")
trained_sd = load_file(f"{OUT}/trained_state.safetensors")
surg_sd = load_file(f"{OUT}/surgical_state.safetensors")

analysis = {}
for l, k in KEYS.items():
    wb = base_sd[k].float().numpy().astype(np.float64)
    wt = trained_sd[k].float().numpy().astype(np.float64)
    ws = surg_sd[k].float().numpy().astype(np.float64)
    v = np.load(V[l]).astype(np.float64)
    v /= max(np.linalg.norm(v), 1e-9)
    d = wt - wb
    rel = float(np.linalg.norm(d) / np.linalg.norm(wb))
    tw = spectral_tw_control(wb, wt, v)
    u, s, _ = np.linalg.svd(d, full_matrices=False)
    u1 = u[:, 0]
    cos_u1 = float(abs(u1 @ v))
    # alignment of the per-column change profile with |v| (which columns moved)
    col_norm = np.linalg.norm(d, axis=0)
    # overlap with surgical S: 128 columns with largest change
    S = np.argsort(col_norm)[-128:]
    # v-energy removed per column, relative to column norm
    Pv_col = np.linalg.norm(np.outer(v, v) @ d[:, S], axis=0) / \
        max(np.linalg.norm(d[:, S], axis=0).mean(), 1e-12)
    surg_S = np.where(np.abs(ws - wb).sum(axis=0) > 0)[0]
    overlap = len(set(S.tolist()) & set(surg_S.tolist())) / max(len(surg_S), 1)
    analysis[k] = {
        "rel_frobenius": round(rel, 5),
        "tw": tw["verdict"] if isinstance(tw, dict) and "verdict" in tw else tw,
        "cos_top_left_sv_vs_v": round(cos_u1, 4),
        "top128_overlap_with_surgical_S": round(overlap, 3),
        "cols_changed_gt_1pct": int((col_norm > 0.01 * col_norm.max()).sum()),
    }

log = json.load(open(f"{OUT}/training_log.json"))
log["final"] = {"analysis": analysis}
json.dump(log, open(f"{OUT}/training_log.json", "w"), indent=1)
print(json.dumps(analysis, indent=1))
