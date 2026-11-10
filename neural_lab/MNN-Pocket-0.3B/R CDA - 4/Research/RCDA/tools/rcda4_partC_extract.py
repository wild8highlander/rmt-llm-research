#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rcda4_partC_extract.py — pull o_proj matrices of layers 15/16/18 from the base
model, save them under BLINDED labels A/B/C (fixed secret permutation) for the
Part C weight-reading test. The mapping never enters the agent workspace.
"""
import json
import os
import random

import numpy as np
from safetensors import safe_open

MODEL = "/home/z/my-project/rcda_work/model/model.safetensors"
OUT = "/home/z/my-project/rcda_work/rcda4/partC"
LAYERS = [15, 16, 18]
GT = {"15": 7, "16": 8, "18": 12}          # measured top heads (rcda3_heads.json)

os.makedirs(OUT, exist_ok=True)
rng = random.Random(20261109)
perm = LAYERS[:]
rng.shuffle(perm)
mapping = {lab: lay for lab, lay in zip(["A", "B", "C"], perm)}

mats = {}
with safe_open(MODEL, framework="pt") as f:
    for lab, lay in mapping.items():
        k = f"model.layers.{lay}.self_attn.o_proj.weight"
        t = f.get_tensor(k)
        mats[lab] = t.float().numpy()
        assert mats[lab].shape == (896, 896), (k, mats[lab].shape)

np.savez_compressed(os.path.join(OUT, "matrices.npz"), **mats)
# secret mapping stays OUTSIDE the agent cwd
json.dump({"mapping": mapping, "ground_truth_top": GT},
          open("/home/z/my-project/rcda_work/rcda4/partC_secret.json", "w"),
          indent=1)
print("saved", {k: v.shape for k, v in mats.items()}, "mapping:", mapping)
