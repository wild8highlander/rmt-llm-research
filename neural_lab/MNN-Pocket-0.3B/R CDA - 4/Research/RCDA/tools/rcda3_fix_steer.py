#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RCDA-3 patch — re-run the steer arm with the FIXED hook arming.

The first steer run went through a bug (Sentinel counter decoupled from the
SteerHook objects), so its records are placebo data. This script drops them,
re-runs the arm with the fixed code path, patches rcda3_sentinel.json, and
re-evaluates the winner. Ban/both records and all benchmark arms are kept.
"""
import json
import os
import sys
import time

import numpy as np
import torch

WORK = "/home/z/my-project/rcda_work"
sys.path.insert(0, os.path.join(WORK, "Code"))
sys.path.insert(0, "/home/z/my-project/scripts")

from benchmark_rmt import _load_model_dir  # noqa: E402
from pocket_rmt_probe import _read_prompt_set  # noqa: E402
from rcda3_sentinel import (K_STEER, MAX_NEW, STEER_LAYERS, SentinelLoop,  # noqa: E402
                            SteerHook)

OUT = os.path.join(WORK, "rcda3")
GENS = os.path.join(OUT, "rcda3_sentinel_generations.jsonl")
SENT_JSON = os.path.join(OUT, "rcda3_sentinel.json")


def main():
    t0 = time.time()
    sent = json.load(open(SENT_JSON))
    calib = {}
    for k, v in sent["calib"].items():
        calib[(int(k.split("_")[0][1:]), int(k.split("_")[1][1:]))] = v

    recs = [json.loads(l) for l in open(GENS)]
    keep = [r for r in recs if r["arm"] != "rescue_steer"]
    with open(GENS, "w") as f:
        for r in keep:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"dropped {len(recs) - len(keep)} placebo steer records", flush=True)

    model, tok = _load_model_dir(os.path.join(WORK, "model"))
    vectors = {l: torch.load(os.path.join(WORK, "rcda2/vectors",
                                          f"L{l}_delta_h_behav.pt"),
                             weights_only=True).numpy().astype(np.float32)
               for l in STEER_LAYERS}
    hooks = [SteerHook(l, vectors[l]) for l in STEER_LAYERS]
    handles = [model.model.layers[l].register_forward_hook(h)
               for l, h in zip(STEER_LAYERS, hooks)]

    det = {r["id"]: r for r in keep if r["arm"] == "detect"}
    looping_ids = [rid for rid, r in det.items() if r["collapse"] is not None]
    id2item = {}
    for split in ("bad", "good"):
        for it in _read_prompt_set(os.path.join(WORK, f"prompts/{split}.jsonl")):
            id2item[it["id"]] = it

    fresh = []
    for i, gid in enumerate(looping_ids):
        rec = SentinelLoop(model, tok, calib, rescue="steer", hooks=hooks)\
            .run(id2item[gid]["text"], max_new=MAX_NEW)
        rec["id"] = gid
        rec["split"] = gid.split("_")[0]
        rec["prompt"] = id2item[gid]["text"]
        rec["arm"] = "rescue_steer"
        fresh.append(rec)
        with open(GENS, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if (i + 1) % 8 == 0:
            print(f"  [steer-fixed {i+1}/{len(looping_ids)}] ({time.time()-t0:.0f}s)",
                  flush=True)
    for h in handles:
        h.remove()

    stats = {"n": 0, "rescued": 0, "waste": [], "clean_tail": [],
             "n_interventions": []}
    for rec in fresh:
        alarm0 = rec["alarm_steps"][0] if rec["alarm_steps"] else None
        last_loop = max([j for j, x in enumerate(rec["labels"]) if x], default=-1)
        tail = len(rec["ids"]) - 1 - alarm0 if alarm0 is not None else 0
        ended_early = len(rec["ids"]) < MAX_NEW
        rescued = bool(alarm0 is not None and last_loop <= alarm0 + 1
                       and (tail >= 8 or ended_early))
        stats["n"] += 1
        stats["rescued"] += int(rescued)
        stats["waste"].append(sum(rec["labels"]))
        stats["clean_tail"].append(len(rec["ids"]) - 1 - last_loop)
        stats["n_interventions"].append(rec["n_interventions"])
    arm = {"n": stats["n"], "rescued": stats["rescued"],
           "rescue_rate": stats["rescued"] / max(stats["n"], 1),
           "loop_waste_mean": float(np.mean(stats["waste"])),
           "clean_tail_median": float(np.median(stats["clean_tail"])),
           "interventions_mean": float(np.mean(stats["n_interventions"])),
           "note": "re-run with fixed hook arming"}
    print(f"arm steer (fixed): {arm}", flush=True)

    sent["rescue_arms"]["steer"] = arm
    winner = max(sent["rescue_arms"],
                 key=lambda a: (sent["rescue_arms"][a]["rescue_rate"],
                                sent["rescue_arms"][a]["clean_tail_median"]))
    sent["winner"] = winner
    if winner != "both":
        sent.setdefault("warnings", []).append(
            f"winner changed to {winner} after the steer re-run; "
            f"benchmark arm C used 'both'")
    json.dump(sent, open(SENT_JSON, "w"), indent=1, default=float)
    print(f"winner: {winner}; saved {SENT_JSON} ({time.time()-t0:.0f}s)",
          flush=True)


if __name__ == "__main__":
    main()
