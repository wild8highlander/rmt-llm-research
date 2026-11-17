#!/usr/bin/env python3
"""rcda6_minibench.py — the RCDA-6 mini-benchmark as a standalone pass.

The in-training bench (gemma_rcds6_train.py --bench-at) generates 8 greedy
192-token completions INSIDE the training process, where the allocator is
already holding masters + momentum + graph cache — on this 2-core / 4 GB box
that made each generation ~6-10x slower and blew the session time budget.
This tool runs the identical measurement in a fresh process:

    * loads the base model,
    * applies the checkpointed fp32 masters (ckpt_masters.pt) through the
      same FP32Linear wrappers used in training,
    * generates the first --bench-n bad prompts (greedy, 192, raw mode),
    * computes rho_mini / mean_ls exactly like the in-training bench,
    * MERGES the record into training_log.json (tag --tag, e.g. step80).

Usage:
  rcda6_minibench.py --model DIR --work WORK/rcda6 --tag step80 [--steps-step 80]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

CODE = Path(__file__).resolve().parents[3] / "Code"
sys.path.insert(0, str(CODE))
import gemma_lab as L                                    # noqa: E402

sys.path.insert(0, str(CODE))
from gemma_rcds6_train import FP32Linear                 # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--work", required=True, help="rcda6 runtime dir")
    ap.add_argument("--tag", required=True, help="bench tag, e.g. step80")
    ap.add_argument("--bench-n", type=int, default=8)
    ap.add_argument("--band", type=int, nargs="+",
                    default=[8, 9, 10, 11, 12, 13, 14, 15])
    args = ap.parse_args()

    work = Path(args.work)
    band = sorted(set(args.band))
    tok = L.load_tokenizer(args.model)
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(max(1, __import__("os").cpu_count() or 2))
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.eval()

    # apply the checkpointed masters through the training wrappers
    masters = torch.load(work / "ckpt_masters.pt", weights_only=True,
                         mmap=True, map_location="cpu")
    with torch.no_grad():
        for l in band:
            for t in ("o_proj", "down_proj"):
                mod = (model.model.layers[l].self_attn.o_proj if t == "o_proj"
                       else model.model.layers[l].mlp.down_proj)
                lin = FP32Linear(mod.weight)
                lin.weight.copy_(masters[f"{l}_{t}"])
                if t == "o_proj":
                    model.model.layers[l].self_attn.o_proj = lin
                else:
                    model.model.layers[l].mlp.down_proj = lin
    del masters
    import gc
    gc.collect()
    print(f"masters applied ({len(band)} x 2 tensors) | RSS {L.rss_gb():.1f}GB",
          flush=True)

    bad_prompts = L.load_bad_prompts()[:args.bench_n]
    rows, t0 = [], time.time()
    for gid, p in bad_prompts:
        tg = time.time()
        ids = L.gen_ids(model, tok, p, L.MAX_NEW)
        ls = L.loop_share(ids)
        rows.append({"id": gid, "loop_share": round(ls, 3), "n_tok": len(ids)})
        print(f"[gen] {gid}: ls={ls:.3f} ({time.time()-tg:.0f}s)", flush=True)
    rho = sum(1 for r in rows if r["loop_share"] >= L.LOOP_THRESHOLD) / len(rows)
    rec = {"tag": args.tag, "rho_mini": rho,
           "mean_ls": round(float(np.mean([r["loop_share"] for r in rows])), 3),
           "gens": rows, "standalone": True,
           "wall_s": round(time.time() - t0, 1)}
    print(f"[bench {args.tag}] rho_mini={rho} "
          f"mean_ls={rec['mean_ls']}", flush=True)

    log_path = work / "training_log.json"
    log = json.load(log_path.open())
    log["bench"] = [b for b in log["bench"] if b["tag"] != args.tag]
    log["bench"].append(rec)
    tmp = log_path.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(log, f, indent=1)
    tmp.replace(log_path)
    print(f"merged into {log_path}", flush=True)


if __name__ == "__main__":
    main()
