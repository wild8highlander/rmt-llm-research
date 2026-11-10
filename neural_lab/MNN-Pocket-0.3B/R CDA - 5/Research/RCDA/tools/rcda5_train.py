#!/usr/bin/env python3
"""RCDA-5 — train the defect out: gradient fine-tuning where the formula IS
the loss. Trainable: o_proj of layers 15/16/18 only (2.4M params, same
tensors the surgical edit touches). Loss per step:

    loss = MSE(cur_logits, base_logits | good prompt+base completion)
         + alpha * mean_l mean_pos (v_l^T h_l / ||v_l||)^2 on bad prefixes

where h_l is the residual stream after layer l (l in {15,16,18}) on the
model's own looping greedy prefix. The penalty term is exactly (I - P_v)
expressed as a training signal: it teaches all weights to stop writing the
loop direction, then the question is whether SGD converges to the same
solution as the surgical projection (TW pass, cos(edit, v) ~ 1) or to
something else.

Usage: rcda5_train.py [--steps 100] [--lr 1e-5] [--alpha-ratio 0.15]
                      [--dry-run]
"""
import argparse
import copy
import json
import sys
import time

import numpy as np
import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, "/home/z/my-project/rcda_work/Code")
sys.path.insert(0, "/home/z/my-project/scripts")
from rmt_intervention import spectral_tw_control  # noqa: E402

WORK = "/home/z/my-project/rcda_work"
OUT = f"{WORK}/rcda5"
MODEL_DIR = f"{WORK}/model"
V = {15: f"{WORK}/rcda4/partB/v_L15.npy",
     16: f"{WORK}/rcda4/partB/v_L16.npy",
     18: f"{WORK}/rcda4/partB/v_L18.npy"}
TRAIN_KEYS = [f"model.layers.{l}.self_attn.o_proj.weight" for l in (15, 16, 18)]
NGRAM = 4
BAD_PREF = 48
MAX_NEW = 96


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


def prompts_from_sentinel():
    S = json.load(open(f"{WORK}/rcda3/rcda3_sentinel.json"))
    det = {}
    for line in open(f"{WORK}/rcda3/rcda3_sentinel_generations.jsonl"):
        r = json.loads(line)
        if r["arm"] == "detect":
            det[r["id"]] = r["prompt"]
    bad_ids = [r["id"] for r in S["benchmark"]["A_baseline"]["rows"]]
    good_ids = [r["id"] for r in S["benchmark"]["FP_good"]["rows"]]
    return [(i, det[i]) for i in bad_ids], [(i, det[i]) for i in good_ids]


def gen(model, tok, prompt, n):
    enc = tok(prompt, return_tensors="pt")
    out = model.generate(**enc, max_new_tokens=n, do_sample=False)
    return out[0][enc["input_ids"].shape[1]:].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--mse-cap", type=float, default=1.0,
                    help="anchor: hard penalty if good-text logit MSE exceeds this")
    ap.add_argument("--beta", type=float, default=10.0,
                    help="weight of the cap violation term")
    ap.add_argument("--bench-at", type=int, nargs="+", default=[0, 50, 100])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32)
    model.eval()

    base_sd = load_file(f"{OUT}/base_state.safetensors")
    global full_sd
    full_sd = {k: v.clone() for k, v in base_sd.items()}
    model.load_state_dict(full_sd, strict=True)

    bad, good = prompts_from_sentinel()
    v = {l: torch.tensor(np.load(V[l]).astype(np.float32)) for l in V}
    vhat = {l: v[l] / v[l].norm() for l in v}

    # ---- build training data from base-model greedy generations ----
    t0 = time.time()
    good_items = []          # (input_ids, base_logits[seq-1])
    for gid, p in good:
        ids = gen(model, tok, p, MAX_NEW)
        text_ids = tok(p, return_tensors="pt").input_ids[0].tolist() + ids
        good_items.append(text_ids)
    bad_items = []           # (prompt_len, prefix_ids)
    for gid, p in bad:
        ids = gen(model, tok, p, BAD_PREF)
        plen = len(tok(p, return_tensors="pt").input_ids[0])
        bad_items.append((plen, tok(p, return_tensors="pt").input_ids[0].tolist() + ids))
    print(f"data built in {time.time()-t0:.0f}s: {len(good_items)} good, "
          f"{len(bad_items)} bad", flush=True)

    # ---- trainable: o_proj of L15/16/18 ----
    model.load_state_dict(full_sd, strict=True)
    for name, p in model.named_parameters():
        p.requires_grad_(name in TRAIN_KEYS)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print("trainable params:", sum(p.numel() for p in trainable), flush=True)
    opt = torch.optim.Adam(trainable, lr=args.lr)

    # base logits for the anchor: last 32 positions of each good completion
    # (tail is where the defect would break fluency); full-sequence logits
    # for 20 texts x 152k vocab would OOM the 3 GB sandbox
    ANCHOR_TAIL = 32
    with torch.no_grad():
        base_logits_store = []
        for text_ids in good_items:
            x = torch.tensor([text_ids[:-1]])
            base_logits_store.append(
                model(x).logits[0][-ANCHOR_TAIL:].detach().clone())

    def penalty_on(plen, ids):
        x = torch.tensor([ids])
        out = model(x, output_hidden_states=True)
        hs = out.hidden_states          # hs[l] = residual after layer l
        terms = []
        for l in (15, 16, 18):
            h = hs[l][0, plen:]         # generated positions only
            proj = h @ vhat[l]          # (n_pos,)
            terms.append((proj ** 2).mean())
        return torch.stack(terms).mean(), out.logits

    def anchor_on(idx):
        x = torch.tensor([good_items[idx][:-1]])
        logits = model(x).logits[0][-ANCHOR_TAIL:]
        mse = ((logits - base_logits_store[idx]) ** 2).mean()
        return mse

    # alpha auto-scale from init magnitudes
    pen0 = {}
    with torch.no_grad():
        for j, (pl, ids) in enumerate(bad_items):
            pen0[j] = float(penalty_on(pl, ids)[0])
    pen_ref = float(np.median(list(pen0.values())))
    print(f"init pen per bad prompt: {[round(x, 2) for x in pen0.values()]} "
          f"-> pen_ref={pen_ref:.3f}", flush=True)
    log = {"args": vars(args), "pen0": pen0, "pen_ref": pen_ref,
           "steps": [], "bench": []}

    def bench(tag):
        # NOTE: no load_state_dict here — bench the CURRENT parameters
        with torch.no_grad():
            rows = []
            for gid, p in bad:
                ids = gen(model, tok, p, MAX_NEW)
                ls = float(token_loop_labels(ids).mean())
                rows.append({"id": gid, "loop_share": round(ls, 3),
                             "n_tok": len(ids)})
        rho = sum(1 for r in rows if r["loop_share"] >= 0.5) / len(rows)
        rec = {"tag": tag, "rho_mini": rho,
               "mean_ls": round(float(np.mean([r["loop_share"] for r in rows])), 3),
               "gens": rows}
        log["bench"].append(rec)
        print(f"[bench {tag}] rho_mini={rho} mean_ls={rec['mean_ls']}", flush=True)
        json.dump(log, open(f"{OUT}/training_log.json", "w"), indent=1)
        return rho

    bench("step0")
    gi = bi = 0
    for step in range(1, args.steps + 1):
        t0 = time.time()
        opt.zero_grad()
        mse = anchor_on(gi % len(good_items)); gi += 1
        j = bi % len(bad_items); bi += 1
        pen_raw, _ = penalty_on(*bad_items[j])
        # normalized penalty: this prompt's projection energy relative to its
        # init value, scaled to the reference magnitude
        pen = (pen_raw / max(pen0[j], 1e-6)) * pen_ref
        cap_viol = torch.relu(mse - args.mse_cap)
        loss = pen + args.beta * cap_viol
        loss.backward()
        opt.step()
        rec = {"step": step, "mse": round(float(mse.detach()), 4),
               "pen_raw": round(float(pen_raw.detach()), 4),
               "pen_norm": round(float(pen.detach()), 4),
               "s": round(time.time() - t0, 1)}
        log["steps"].append(rec)
        if step % 10 == 0 or args.dry_run:
            print(f"[step {step}] mse={rec['mse']} pen_raw={rec['pen_raw']} "
                  f"pen_norm={rec['pen_norm']} ({rec['s']}s/step)", flush=True)
        if step in args.bench_at and not args.dry_run:
            bench(f"step{step}")
        if args.dry_run and step >= 2:
            break

    if args.dry_run:
        print("DRY RUN OK", flush=True)
        return

    # save trained state + TW/cos analysis of what SGD actually changed
    # clone the trained tensors BEFORE any restore overwrites them
    trained = {k: p.detach().clone() for k, p in zip(TRAIN_KEYS, trainable)}
    for k in TRAIN_KEYS:
        full_sd[k] = trained[k]
    save_file(full_sd, f"{OUT}/trained_state.safetensors")

    analysis = {}
    for l, k in zip((15, 16, 18), TRAIN_KEYS):
        d = (full_sd[k].float() - base_sd[k].float())
        rel = float(torch.linalg.norm(d) / torch.linalg.norm(base_sd[k].float()))
        tw = spectral_tw_control(base_sd[k].float().numpy(),
                                 full_sd[k].float().numpy(),
                                 np.load(V[l]).astype(np.float64))
        analysis[k] = {"rel_frobenius": round(rel, 5),
                       "tw": tw["verdict"] if isinstance(tw, dict)
                             and "verdict" in tw else tw}
        # how much of the edit direction aligns with the loop direction v
        vh = float(torch.tensor(np.load(V[l]).astype(np.float32)) @ d)
        analysis[k]["cos_v"] = round(vh / max(float(torch.linalg.norm(d))
                                              * float(np.linalg.norm(np.load(V[l]))), 1e-9), 4)
    log["final"] = {"analysis": analysis}
    json.dump(log, open(f"{OUT}/training_log.json", "w"), indent=1)
    print("saved trained_state.safetensors")
    print(json.dumps(analysis, indent=1))


if __name__ == "__main__":
    main()
