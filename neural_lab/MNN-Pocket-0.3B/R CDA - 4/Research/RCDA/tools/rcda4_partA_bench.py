#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rcda4_partA_bench.py — independent verification of the MIT weight edit
(RikiyaT/Qwen3.5-0.8B-weight-space-programming) on our own looping benchmark.

Arms: base Qwen3.5-0.8B vs the edited release. Prompts: 12 bad RU (known
looping class from RCDA-3) + 6 good RU + 6 GSM8K train EN. Sampling per the
Qwen3.5 model card: temperature 0.6, top-p 0.95, top-k 20, thinking on.

Metrics per generation:
  loop_share : fraction of generated tokens whose 4-gram already appeared
               earlier (tail covered by repeats)
  looping    : loop_share >= 0.5  (rho definition; same notion as RCDA-3's
               "collapse", generalised to sampled thinking-mode outputs)
  truncated  : generation hit max_new_tokens
  distinct4  : distinct 4-grams / total 4-grams

Gate A (pre-registered in RCDA4_PLAN.md): rho_edited <= 0.5 * rho_base.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

NGRAM = 4
MAX_NEW = 512

GEN_KW = dict(do_sample=True, temperature=0.6, top_p=0.95, top_k=20,
              max_new_tokens=MAX_NEW)

# Pre-registered metric amendment (after base-only smoke, before the edited
# run): the MIT pathology on this model is "reasoning never finishes and the
# same analysis lines repeat", not exact 4-gram loops at short horizon. So:
#   finished        = produced </think> within the budget
#   line_repeat     = fraction of generated lines duplicating an earlier line
#   looping         = (not finished) or (line_repeat >= 0.3)
# rho = fraction of a subset that is "looping". 4-gram loop_share kept as a
# secondary diagnostic.


def line_repeat_share(text):
    lines = [l.strip() for l in text.splitlines() if len(l.strip()) >= 8]
    if not lines:
        return 0.0
    seen, dup = set(), 0
    for l in lines:
        if l in seen:
            dup += 1
        else:
            seen.add(l)
    return dup / len(lines)


def token_loop_labels(ids, ngram=NGRAM):
    labels = np.zeros(len(ids), dtype=bool)
    seen = {}
    for t in range(len(ids)):
        if t + 1 >= ngram:
            g = tuple(ids[t - ngram + 1:t + 1])
            if g in seen and seen[g] <= t - ngram:
                labels[t] = True
            if g not in seen:
                seen[g] = t
    return labels


def load_prompts():
    D3 = "/home/z/my-project/rcda_work/rcda3/rcda3_sentinel_generations.jsonl"
    det = [json.loads(l) for l in open(D3) if json.loads(l)["arm"] == "detect"]
    bad = [r for r in det if r["id"].startswith("bad")][:8]
    good = [r for r in det if not r["id"].startswith("bad")][:4]
    gsm = [json.loads(l)["question"] for l in
           open("/home/z/my-project/rcda_work/rcda4/gsm8k_train.jsonl")][:6]
    return ([(r["id"], r["prompt"], "bad_ru") for r in bad]
            + [(r["id"], r["prompt"], "good_ru") for r in good]
            + [(f"gsm8k_{i:02d}", q, "gsm8k_en") for i, q in enumerate(gsm)])


@torch.no_grad()
def run(model_dir, tag, out_prefix, limit=None):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.eval()
    gens = load_prompts()
    if limit:
        gens = gens[:limit]
    rows = []
    for i, (gid, prompt, kind) in enumerate(gens):
        try:
            ids_in = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True, return_tensors="pt",
                enable_thinking=True)
        except TypeError:  # older template without the flag
            ids_in = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True, return_tensors="pt")
        enc = ids_in
        out = model.generate(**enc, **GEN_KW)
        new_ids = out[0][enc["input_ids"].shape[1]:].tolist()
        text = tok.decode(new_ids, skip_special_tokens=True)
        labels = token_loop_labels(new_ids)
        n = len(new_ids)
        loop_share = float(labels.sum() / n) if n else 0.0
        grams = [tuple(new_ids[t:t + 4]) for t in range(max(n - 3, 0))]
        distinct4 = len(set(grams)) / len(grams) if grams else 1.0
        think_close = tok.convert_tokens_to_ids("</think>")
        finished = (think_close is not None and think_close >= 0
                    and think_close in new_ids)
        lr = line_repeat_share(text)
        rows.append({
            "id": gid, "kind": kind, "n_tokens": n,
            "loop_share": round(loop_share, 4),
            "line_repeat": round(lr, 4),
            "finished": finished,
            "looping": bool((not finished) or lr >= 0.3),
            "truncated": bool(n >= MAX_NEW),
            "distinct4": round(distinct4, 4),
            "text": text[:600],
        })
        print(f"[{tag}] {i + 1}/{len(gens)} {gid} {kind}: "
              f"loop_share={loop_share:.2f} trunc={rows[-1]['truncated']} "
              f"tok={n}", flush=True)
        json.dump(rows[-1], open(f"{out_prefix}_{tag}.jsonl", "a"),
                  ensure_ascii=False)
        open(f"{out_prefix}_{tag}.jsonl", "a").write("\n")

    def frac(sub, key):
        sub = [r for r in rows if r["kind"] == sub]
        return round(sum(r[key] for r in sub) / len(sub), 4) if sub else None

    summary = {
        "tag": tag, "model_dir": model_dir, "n_gens": len(rows),
        "max_new": MAX_NEW,
        "sampling": {k: v for k, v in GEN_KW.items()},
        "rho_bad_ru": frac("bad_ru", "looping"),
        "rho_good_ru": frac("good_ru", "looping"),
        "rho_gsm8k_en": frac("gsm8k_en", "looping"),
        "trunc_bad_ru": frac("bad_ru", "truncated"),
        "trunc_gsm8k_en": frac("gsm8k_en", "truncated"),
        "loop_share_bad_mean": float(np.mean([r["loop_share"] for r in rows
                                              if r["kind"] == "bad_ru"])),
        "loop_share_gsm_mean": float(np.mean([r["loop_share"] for r in rows
                                              if r["kind"] == "gsm8k_en"])),
        "finished_bad_ru": frac("bad_ru", "finished"),
        "finished_gsm8k_en": frac("gsm8k_en", "finished"),
        "line_repeat_bad_mean": float(np.mean([r["line_repeat"] for r in rows
                                               if r["kind"] == "bad_ru"])),
        "distinct4_bad_mean": float(np.mean([r["distinct4"] for r in rows
                                             if r["kind"] == "bad_ru"])),
    }
    json.dump(summary, open(f"{out_prefix}_{tag}_summary.json", "w"),
              indent=1)
    print(json.dumps(summary, indent=1), flush=True)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    run(a.model_dir, a.tag, a.out_prefix, a.limit)
