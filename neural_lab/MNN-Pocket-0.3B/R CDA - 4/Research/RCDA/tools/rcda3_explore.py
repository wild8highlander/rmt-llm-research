#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RCDA-3 stage 2b — EXPLORATORY persistent policies (mechanism-driven).

The a-priori timed interventions (ban 4 tokens / steer 8 tokens / both) failed:
the loop attractor re-forms as soon as the intervention window ends
(loop waste 73.8 -> 68.2 tokens despite ~3.5 interventions per generation).
This mirrors the round-2 weight-side result (distributed defect) from the
activation side: transient pushes do not exit the attractor.

Refinement (labeled exploratory, mechanism-driven, no threshold tuning):
  ban_persist    — reactive n-gram exhaustion: at every step, ban ALL tokens
                   that have ever followed the current 3-gram context in this
                   generation. Any token 4-gram can then occur at most once ->
                   the token-level loop becomes impossible BY CONSTRUCTION.
                   No sentinel needed (decode-time constraint).
  both_persist   — the same constraint + spectral steering: while the PR stays
                   below theta (2 consecutive steps, any layer/scale), keep
                   subtracting the behavioral direction at L15/16/18, i.e. the
                   steer is active as long as the spectral signature persists
                   instead of a fixed 8-token window.

Evaluation: 30 looping gens (rescue metrics vs the detection arm), 20 good
gens (FP cost: changed texts, created defects), and — if a persistent arm
rescues — a 20-bad benchmark arm C'. Quality is scored with the same defect
flags plus distinct-4; the FP stream prices the constraint on healthy text.

Output: rcda_work/rcda3/rcda3_explore.json
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

from benchmark_rmt import _defect_flags, _load_model_dir  # noqa: E402
from pocket_rmt_probe import _read_prompt_set  # noqa: E402
from rcda3_sentinel import (  # noqa: E402
    K_STEER, MAX_NEW, SCALES, SENT_LAYERS, STEER_LAYERS, token_loop_labels)
from rcda3_sentinel import SteerHook  # noqa: E402

OUT = os.path.join(WORK, "rcda3")
SENT_JSON = os.path.join(OUT, "rcda3_sentinel.json")


@torch.no_grad()
def run_persistent(model, tok, prompt, calib, mode, hooks=None, max_new=MAX_NEW):
    """Greedy cached loop with a persistent policy.

    mode: "ban_persist" | "both_persist" (ban is part of both).
    Returns a record like SentinelLoop.run.
    """
    eos = tok.eos_token_id
    enc = tok(prompt, return_tensors="pt")
    out = model(**enc, use_cache=True, output_hidden_states=True)
    past = out.past_key_values
    ids = []
    deques = {l: [] for l in SENT_LAYERS}
    below = {(w, l): 0 for w in SCALES for l in SENT_LAYERS}
    steer_steps = 0
    gram_next = {}          # 3-gram -> set of observed continuations
    ctx = []                # last <=3 generated tokens
    alarms = []
    n_int = 0
    nxt = int(out.logits[0, -1].argmax())
    while nxt != eos and len(ids) < max_new:
        # ban every continuation ever observed for the current context
        if ctx and len(ctx) == 3:
            banned_set = gram_next.get(tuple(ctx), set())
            if banned_set:
                logits = out.logits[0, -1].clone()
                for b in banned_set:
                    logits[b] = float("-inf")
                nxt = int(logits.argmax())
                if nxt == eos:
                    break
                n_int += 1
        ids.append(nxt)
        if len(ctx) == 3:
            gram_next.setdefault(tuple(ctx), set()).add(nxt)
        ctx.append(nxt)
        if len(ctx) > 3:
            ctx.pop(0)
        if hooks and steer_steps > 0:
            pass  # hooks hold their own counters; tick below
        for h in hooks or []:
            h.tick()
        out = model(input_ids=torch.tensor([[nxt]]), past_key_values=past,
                    use_cache=True, output_hidden_states=True)
        past = out.past_key_values
        if mode == "both_persist":
            # online PR + continuous re-arming while the signature persists
            fired = False
            for l in SENT_LAYERS:
                deques[l].append(out.hidden_states[l + 1][0, -1].float().numpy())
                if len(deques[l]) > max(SCALES):
                    deques[l].pop(0)
            if len(ids) >= 1:
                for w in SCALES:
                    for l in SENT_LAYERS:
                        dq = deques[l]
                        if len(dq) < w:
                            continue
                        s = np.linalg.svd(np.stack(dq[-w:]), compute_uv=False)
                        p = s ** 2
                        pr = float((p.sum()) ** 2 / max((p ** 2).sum(), 1e-12))
                        below[(w, l)] = below[(w, l)] + 1 \
                            if pr < calib[(w, l)]["theta"] else 0
                        if below[(w, l)] >= 2:
                            fired = True
                if fired:
                    if not alarms or len(ids) - 1 - alarms[-1] > 1:
                        alarms.append(len(ids) - 1)
                    if n_int < 10 ** 6:
                        for h in hooks:
                            h.steps = K_STEER   # keep re-arming while below
        nxt = int(out.logits[0, -1].argmax())
    labels = token_loop_labels(ids)
    collapse = int(np.argmax(labels)) if labels.any() else None
    text = tok.decode(ids, skip_special_tokens=True)
    return {"ids": ids, "labels": [int(x) for x in labels],
            "collapse": collapse, "text": text, "alarm_steps": alarms,
            "n_interventions": n_int}


def eval_arm(model, tok, calib, mode, items, base_texts, tag, hooks=None,
             max_new=MAX_NEW):
    rows = []
    for i, it in enumerate(items):
        rec = run_persistent(model, tok, it["text"], calib, mode, hooks=hooks,
                             max_new=max_new)
        flags = _defect_flags(rec["text"], len(rec["ids"]), max_new, 0.2, 0.8)
        last_loop = max([j for j, x in enumerate(rec["labels"]) if x], default=-1)
        alarm0 = rec["alarm_steps"][0] if rec["alarm_steps"] else None
        rows.append({
            "id": it["id"], "arm": tag, "text": rec["text"],
            "n_tokens": len(rec["ids"]), "collapse": rec["collapse"],
            "labels": rec["labels"], "alarm_steps": rec["alarm_steps"],
            "loop_tokens": int(sum(rec["labels"])),
            "changed_vs_detect": base_texts.get(it["id"]) != rec["text"],
            **flags})
        if (i + 1) % 10 == 0:
            print(f"  [{tag} {i+1}/{len(items)}] ({t0s():.0f}s)", flush=True)
    return rows


def t0s():
    return time.time() - T0


T0 = time.time()


def main():
    sent = json.load(open(SENT_JSON))
    calib = {}
    for k, v in sent["calib"].items():
        w = int(k.split("_")[0][1:])
        l = int(k.split("_")[1][1:])
        calib[(w, l)] = v
    det_texts = {g["id"]: g["text"] for g in
                 (json.loads(l) for l in
                  open(os.path.join(OUT, "rcda3_sentinel_generations.jsonl")))
                 if g.get("arm") == "detect"}
    looping_ids = [rid for rid, g in
                   ((g["id"], g) for g in
                    (json.loads(l) for l in
                     open(os.path.join(OUT, "rcda3_sentinel_generations.jsonl")))
                   if g.get("arm") == "detect") if g["collapse"] is not None]

    model, tok = _load_model_dir(os.path.join(WORK, "model"))
    bad = _read_prompt_set(os.path.join(WORK, "prompts/bad.jsonl"))
    good = _read_prompt_set(os.path.join(WORK, "prompts/good.jsonl"))
    id2item = {it["id"]: it for it in bad + good}

    vectors = {l: torch.load(os.path.join(WORK, "rcda2/vectors",
                                          f"L{l}_delta_h_behav.pt"),
                             weights_only=True).numpy().astype(np.float32)
               for l in STEER_LAYERS}

    result = {"meta": {"mode": "exploratory persistent policies",
                       "ban": "all observed continuations of the current 3-gram",
                       "steer": "re-armed while PR < theta (any layer/scale)",
                       "elapsed_s": 0}}
    arm_rows = {}
    for mode in ("ban_persist", "both_persist"):
        hooks = None
        handles = []
        if mode == "both_persist":
            hooks = [SteerHook(l, vectors[l]) for l in STEER_LAYERS]
            handles = [model.model.layers[l].register_forward_hook(h)
                       for l, h in zip(STEER_LAYERS, hooks)]
        rows_loop = eval_arm(model, tok, calib, mode,
                             [id2item[i] for i in looping_ids],
                             det_texts, f"rescue_{mode}", hooks)
        rows_good = eval_arm(model, tok, calib, mode, good, det_texts,
                             f"FP_{mode}", hooks)
        if handles:
            for h in handles:
                h.remove()
        rescued = 0
        for r, gid in zip(rows_loop, looping_ids):
            det = next(g for g in (json.loads(l) for l in
                                   open(os.path.join(
                                       OUT, "rcda3_sentinel_generations.jsonl")))
                       if g.get("arm") == "detect" and g["id"] == gid)
            a0 = det["alarm_steps"][0] if det["alarm_steps"] else None
            last_loop = max([j for j, x in enumerate(r["labels"]) if x],
                            default=-1)
            if a0 is not None and last_loop <= a0 + 1:
                rescued += 1
        arm_rows[mode] = {
            "rescue_rows": [{k: r[k] for k in ("id", "n_tokens", "loop_tokens",
                                               "collapse", "defect_hit",
                                               "distinct4", "rep4")}
                            for r in rows_loop],
            "fp_rows": [{k: r[k] for k in ("id", "changed_vs_detect",
                                           "n_tokens", "collapse",
                                           "defect_hit")} for r in rows_good],
            "rescue_rate": rescued / max(len(rows_loop), 1),
            "loop_waste_mean": float(np.mean([r["loop_tokens"]
                                              for r in rows_loop])),
            "distinct4_mean_loop": float(np.mean([r["distinct4"] or 1
                                                  for r in rows_loop])),
            "fp_changed": sum(1 for r in rows_good if r["changed_vs_detect"]),
            "fp_loops": sum(1 for r in rows_good if r["collapse"] is not None),
            "rho_good": float(np.mean([r["defect_hit"] for r in rows_good])),
        }
        print(f"{mode}: rescue={arm_rows[mode]['rescue_rate']:.2f} "
              f"waste={arm_rows[mode]['loop_waste_mean']:.1f} "
              f"fp_changed={arm_rows[mode]['fp_changed']}/20 "
              f"rho_good={arm_rows[mode]['rho_good']:.2f} "
              f"({t0s():.0f}s)", flush=True)

    # pick the final persistent policy and run the 20-bad benchmark arm C'
    best = max(arm_rows, key=lambda m: (arm_rows[m]["rescue_rate"],
                                        -arm_rows[m]["fp_changed"]))
    hooks = None
    handles = []
    if best == "both_persist":
        hooks = [SteerHook(l, vectors[l]) for l in STEER_LAYERS]
        handles = [model.model.layers[l].register_forward_hook(h)
                   for l, h in zip(STEER_LAYERS, hooks)]
    rows_c = []
    for it in bad:
        rec = run_persistent(model, tok, it["text"], calib, best, hooks)
        flags = _defect_flags(rec["text"], len(rec["ids"]), MAX_NEW, 0.2, 0.8)
        rows_c.append({"id": it["id"], "n_tokens": len(rec["ids"]),
                       "collapse": rec["collapse"],
                       "loop_tokens": int(sum(rec["labels"])),
                       "text": rec["text"], **flags})
    if handles:
        for h in handles:
            h.remove()
    rho_c = float(np.mean([r["defect_hit"] for r in rows_c]))
    print(f"arm C' ({best}) on 20 bad: rho={rho_c:.2f}", flush=True)

    result["rescue_arms"] = arm_rows
    result["final_policy"] = best
    result["benchmark_C"] = {"arm": best, "rho": rho_c,
                             "loop_waste_mean": float(np.mean(
                                 [r["loop_tokens"] for r in rows_c
                                  if r["collapse"] is not None] or [0])),
                             "rows": [{k: r[k] for k in
                                       ("id", "n_tokens", "rep4", "collapse",
                                        "defect_hit", "loop_tokens")}
                                      for r in rows_c]}
    result["meta"]["elapsed_s"] = round(time.time() - T0, 1)
    with open(os.path.join(OUT, "rcda3_explore.json"), "w") as f:
        json.dump(result, f, indent=1, default=float)
    print(f"saved {OUT}/rcda3_explore.json ({t0s():.0f}s)", flush=True)


if __name__ == "__main__":
    main()
