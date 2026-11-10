#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RCDA-3 stage 2 — head-level surgery: is the head concentration causal?

Round 2 found that the behavioral loop direction Delta_h lands on o_proj
columns that concentrate in single heads (L15->h7 x4.0, L16->h8, L18->h0).
This script tests the causal claim with a dose-response of HEAD ABLATION:

  1. Profile: per-head sensitivity e_h = ||W_o[:, block_h]^T v||^2 for the
     behavioral v at L15/16/17/18 (14 heads x 64 columns each).

  2. Surgery (in-memory, pristine weights restored after each condition):
     zero the columns of one head's output block, i.e. kill exactly that
     head's write into the residual stream. Conditions: top head of L15,
     L16, L18; the 3-head combo; and a PLACEBO — the median-concentration
     head of each layer (matched "ordinary" heads, no free parameters).

  3. Calibration metrics per condition (8 bad + 8 good, 48 tokens):
     changed greedy trajectories, rep4/defect flags, KL(base||edited) on the
     5 control texts, relative Frobenius change, Tracy-Widom verdict of the
     surviving top spectrum.

  4. Decision rule (fixed a priori): run the 20-bad benchmark arm D with the
     combo iff combo changes >=2 bad trajectories AND hit_bad drops by >=0.15
     AND hit_good does not grow by more than 0.1.

Outputs: rcda_work/rcda3/rcda3_heads.json
"""
import gc
import json
import os
import sys
import time

import numpy as np
import torch

WORK = "/home/z/my-project/rcda_work"
sys.path.insert(0, os.path.join(WORK, "Code"))

from rmt_intervention import spectral_tw_control  # noqa: E402
from benchmark_rmt import (_defect_flags, _load_model_dir,  # noqa: E402
                           perplexity)
from pocket_rmt_probe import _read_prompt_set  # noqa: E402

MODEL_DIR = os.path.join(WORK, "model")
OUT = os.path.join(WORK, "rcda3")
os.makedirs(OUT, exist_ok=True)

HEADS = 14
D_HEAD = 64
CAL_N = 8
CAL_MAX_NEW = 48
MAX_NEW = 96
VECTORS = {15: "L15_delta_h_behav.pt", 16: "L16_delta_h_behav.pt",
           17: "L17_delta_h_behav.pt", 18: "L18_delta_h_behav.pt"}


def oproj_key(layer):
    return f"model.layers.{layer}.self_attn.o_proj.weight"


@torch.no_grad()
def greedy_cached(model, tok, prompt, max_new):
    """Plain greedy loop with the KV cache (same semantics as the sentinel)."""
    eos = tok.eos_token_id
    enc = tok(prompt, return_tensors="pt")
    out = model(**enc, use_cache=True)
    past = out.past_key_values
    ids = []
    nxt = int(out.logits[0, -1].argmax())
    while nxt != eos and len(ids) < max_new:
        ids.append(nxt)
        out = model(input_ids=torch.tensor([[nxt]]), past_key_values=past,
                    use_cache=True)
        past = out.past_key_values
        nxt = int(out.logits[0, -1].argmax())
    text = tok.decode(ids, skip_special_tokens=True)
    return ids, text


@torch.no_grad()
def gen_batch(model, tok, items, max_new):
    rows = []
    for it in items:
        ids, text = greedy_cached(model, tok, it["text"], max_new)
        rows.append({"id": it["id"], "ids": ids, "text": text,
                     **_defect_flags(text, len(ids), max_new, 0.2, 0.8)})
    return rows


@torch.no_grad()
def mean_kl_vs_base(model, tok, ctrl_items, base_logits):
    """KL(base || edited) averaged over control-text positions."""
    kls = []
    for it, bl in zip(ctrl_items, base_logits):
        enc = tok(it["text"], return_tensors="pt", truncation=True, max_length=512)
        logits = model(**enc).logits[0].float()
        lb = torch.log_softmax(bl, dim=-1)
        le = torch.log_softmax(logits, dim=-1)
        kl = (lb.exp() * (lb - le)).sum(-1)
        kls.append(float(kl.mean()))
    return float(np.mean(kls))


def head_profile(W, v):
    """Per-head sensitivity of the behavioral direction to head outputs."""
    e = []
    for h in range(HEADS):
        blk = W[:, h * D_HEAD:(h + 1) * D_HEAD]
        e.append(float(np.linalg.norm(blk.T @ v) ** 2))
    e = np.array(e)
    conc = e / max(e.mean(), 1e-18)
    return {"energy": e.tolist(), "concentration": conc.tolist(),
            "top_head": int(np.argmax(conc)),
            "median_head": int(np.argsort(conc)[len(conc) // 2])}


def zero_head(sd, layer, head):
    key = oproj_key(layer)
    sd[key].data[:, head * D_HEAD:(head + 1) * D_HEAD] = 0.0


def main():
    t0 = time.time()
    model, tok = _load_model_dir(MODEL_DIR)
    sd = model.state_dict()
    print("model loaded", flush=True)

    # ------------------------------------------------------ 1. head profile
    vecs = {}
    for l, fname in VECTORS.items():
        vecs[l] = torch.load(os.path.join(WORK, "rcda2/vectors", fname),
                             weights_only=True).numpy().astype(np.float64)
        vecs[l] /= max(np.linalg.norm(vecs[l]), 1e-12)
    profiles = {}
    for l in VECTORS:
        W = sd[oproj_key(l)].detach().float().numpy()
        profiles[str(l)] = head_profile(W, vecs[l])
        p = profiles[str(l)]
        print(f"  profile L{l}: top=head{p['top_head']} "
              f"(x{p['concentration'][p['top_head']]:.2f}), "
              f"median=head{p['median_head']}", flush=True)

    # ------------------------------------------------- 2. surgery conditions
    targets = {l: profiles[str(l)]["top_head"] for l in (15, 16, 18)}
    placebos = {l: profiles[str(l)]["median_head"] for l in (15, 16, 18)}
    conditions = {
        "L15_top": [(15, targets[15])],
        "L16_top": [(16, targets[16])],
        "L18_top": [(18, targets[18])],
        "combo3": [(l, targets[l]) for l in (15, 16, 18)],
        "placebo3": [(l, placebos[l]) for l in (15, 16, 18)],
    }

    cal_bad = _read_prompt_set(os.path.join(WORK, "prompts/bad.jsonl"), CAL_N)
    cal_good = _read_prompt_set(os.path.join(WORK, "prompts/good.jsonl"), CAL_N)
    ctrl = _read_prompt_set(os.path.join(WORK, "prompts/control.jsonl"))
    base_logits = []
    with torch.no_grad():
        for it in ctrl:
            enc = tok(it["text"], return_tensors="pt", truncation=True, max_length=512)
            base_logits.append(model(**enc).logits[0].float())

    base_cal = {"bad": gen_batch(model, tok, cal_bad, CAL_MAX_NEW),
                "good": gen_batch(model, tok, cal_good, CAL_MAX_NEW)}
    base = {"rep4_bad": float(np.mean([r["rep4"] or 0 for r in base_cal["bad"]])),
            "hit_bad": float(np.mean([r["defect_hit"] for r in base_cal["bad"]])),
            "hit_good": float(np.mean([r["defect_hit"] for r in base_cal["good"]])),
            "kl": 0.0}
    print(f"baseline cal: {base}", flush=True)

    pristine = {l: sd[oproj_key(l)].detach().clone() for l in (15, 16, 18)}
    results = {}
    for name, heads in conditions.items():
        for l, h in heads:
            zero_head(sd, l, h)
        kl = mean_kl_vs_base(model, tok, ctrl, base_logits)
        cal = {"bad": gen_batch(model, tok, cal_bad, CAL_MAX_NEW),
               "good": gen_batch(model, tok, cal_good, CAL_MAX_NEW)}
        changed_bad = int(sum(a["text"] != b["text"]
                              for a, b in zip(cal["bad"], base_cal["bad"])))
        changed_good = int(sum(a["text"] != b["text"]
                               for a, b in zip(cal["good"], base_cal["good"])))
        rep4_bad = float(np.mean([r["rep4"] or 0 for r in cal["bad"]]))
        hit_bad = float(np.mean([r["defect_hit"] for r in cal["bad"]]))
        hit_good = float(np.mean([r["defect_hit"] for r in cal["good"]]))
        tw = {}
        for l, _ in heads:
            Wn = sd[oproj_key(l)].detach().float().numpy()
            ctrl_tw = spectral_tw_control(pristine[l].numpy(), Wn, vecs[l])
            tw[str(l)] = ctrl_tw["verdict"]
        rel = float(np.sqrt(sum(float(((sd[oproj_key(l)] - pristine[l]) ** 2).sum())
                                for l, _ in heads)
                            / sum(float((pristine[l] ** 2).sum())
                                  for l, _ in heads)))
        results[name] = {"heads": [[l, h] for l, h in heads],
                         "kl": kl, "changed_bad": changed_bad,
                         "changed_good": changed_good, "rep4_bad": rep4_bad,
                         "hit_bad": hit_bad, "hit_good": hit_good,
                         "rel_frobenius": rel, "tw_verdicts": tw}
        print(f"  {name} {results[name]['heads']}: changed={changed_bad}/{changed_good} "
              f"hit_bad={hit_bad:.2f} hit_good={hit_good:.2f} KL={kl:.4f} "
              f"rel={rel:.2e} tw={tw} ({time.time()-t0:.0f}s)", flush=True)
        for l in pristine:                      # restore pristine weights
            sd[oproj_key(l)].data.copy_(pristine[l])

    # ------------------------------------------------------ 3. decision rule
    c = results["combo3"]
    go = (c["changed_bad"] >= 2 and c["hit_bad"] <= base["hit_bad"] - 0.15
          and c["hit_good"] <= base["hit_good"] + 0.1)
    print(f"decision: benchmark arm D {'GO' if go else 'NO-GO'}", flush=True)

    arm_d = None
    if go:
        for l, h in conditions["combo3"]:
            zero_head(sd, l, h)
        rows = gen_batch(model, tok, _read_prompt_set(
            os.path.join(WORK, "prompts/bad.jsonl")), MAX_NEW)
        rho_d = float(np.mean([r["defect_hit"] for r in rows]))
        ppls = [perplexity(model, tok, it["text"]) for it in ctrl]
        arm_d = {"rho": rho_d, "ppl_ctrl_mean": float(np.nanmean(ppls)),
                 "ppl_ctrl": [None if p is None else float(p) for p in ppls],
                 "rows": [{k: r[k] for k in ("id", "n_tokens", "rep4", "collapse",
                                             "defect_hit")} for r in rows]}
        print(f"arm D: rho={rho_d:.2f} ppl={arm_d['ppl_ctrl_mean']:.1f}",
              flush=True)
        for l in pristine:
            sd[oproj_key(l)].data.copy_(pristine[l])

    out = {"meta": {"heads": HEADS, "d_head": D_HEAD, "cal_n": CAL_N,
                    "cal_max_new": CAL_MAX_NEW, "max_new": MAX_NEW,
                    "elapsed_s": round(time.time() - t0, 1)},
           "head_profiles": profiles, "targets": targets,
           "placebos": placebos, "baseline": base,
           "conditions": results, "decision_go": bool(go), "arm_d": arm_d}
    with open(os.path.join(OUT, "rcda3_heads.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"saved {OUT}/rcda3_heads.json ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
