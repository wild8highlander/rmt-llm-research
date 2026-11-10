#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rcda4_eval_b.py — final evaluation of a Part B agent edit against the
pre-registered gates: full 20 bad + 20 good benchmark, KL(base||edited) on 5
control texts, Tracy-Widom spectral control per edited tensor, and cosine
between the edit's top direction and the measured loop direction v_l.

Usage: rcda4_eval_b.py --edited-state PATH --tag blind|aided --out PREFIX
"""
import argparse
import json
import sys

import numpy as np
import torch
from safetensors.torch import load_file

sys.path.insert(0, "/home/z/my-project/rcda_work/Code")
sys.path.insert(0, "/home/z/my-project/scripts")
from rmt_intervention import spectral_tw_control  # noqa: E402

MODEL_DIR = "/home/z/my-project/rcda_work/model"
NGRAM = 4
MAX_NEW = 96
V = {15: "/home/z/my-project/rcda_work/rcda4/partB/v_L15.npy",
     16: "/home/z/my-project/rcda_work/rcda4/partB/v_L16.npy",
     18: "/home/z/my-project/rcda_work/rcda4/partB/v_L18.npy"}


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
    S = json.load(open("/home/z/my-project/rcda_work/rcda3/rcda3_sentinel.json"))
    D3 = "/home/z/my-project/rcda_work/rcda3/rcda3_sentinel_generations.jsonl"
    det = {}
    for line in open(D3):
        r = json.loads(line)
        if r["arm"] == "detect":
            det[r["id"]] = r["prompt"]
    bad_ids = [r["id"] for r in S["benchmark"]["A_baseline"]["rows"]]
    good_ids = [r["id"] for r in S["benchmark"]["FP_good"]["rows"]]
    return [(i, det[i], "bad") for i in bad_ids], \
           [(i, det[i], "good") for i in good_ids]


@torch.no_grad()
def gen_ids(model, tok, prompt):
    enc = tok(prompt, return_tensors="pt")
    out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False)
    return out[0][enc["input_ids"].shape[1]:].tolist()


@torch.no_grad()
def kl_ctrl(model, tok, texts, base_logits):
    kls = []
    for text, bl in zip(texts, base_logits):
        enc = tok(text, return_tensors="pt", truncation=True, max_length=512)
        le = torch.log_softmax(model(**enc).logits[0].float(), dim=-1)
        lb = torch.log_softmax(bl, dim=-1)
        kls.append(float((lb.exp() * (lb - le)).sum(-1).mean()))
    return float(np.mean(kls))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edited-state", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    base_sd = load_file(f"{MODEL_DIR}/model.safetensors")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16)
    model.eval()

    bad, good = prompts_from_sentinel()
    ctrl_texts = []
    for i, p, _ in good[:5]:
        t = tok.decode(gen_ids(model, tok, p), skip_special_tokens=True)
        enc = tok(t, return_tensors="pt")
        if int(enc["input_ids"].shape[1]) >= 8:  # skip instant-EOS generations
            ctrl_texts.append(t)
    if not ctrl_texts:                       # fall back to raw good prompts
        ctrl_texts = [p for i, p, _ in good[:5]]
    base_logits = []
    for t in ctrl_texts:
        enc = tok(t, return_tensors="pt", truncation=True, max_length=512)
        base_logits.append(model(**enc).logits[0].float().cpu())

    # apply the agent's edit
    ed = load_file(a.edited_state)
    changed = {}
    for k in base_sd:
        if k in ed and not torch.equal(base_sd[k], ed[k]):
            d = (ed[k].float() - base_sd[k].float())
            rel = float(torch.linalg.norm(d)
                        / max(torch.linalg.norm(base_sd[k].float()), 1e-9))
            changed[k] = {"rel_frobenius": round(rel, 5),
                          "n_weights": int((d != 0).sum())}
    model.load_state_dict(ed, strict=True)

    tw, cosines = {}, {}
    for k, meta in changed.items():
        parts = k.split(".")
        if len(parts) > 2 and parts[1] == "layers":
            lay = int(parts[2])
            if lay in V and lay in [15, 16, 18]:
                v = np.load(V[lay]).astype(np.float64)
                v /= max(np.linalg.norm(v), 1e-9)
                wb = base_sd[k].float().numpy().astype(np.float64)
                wa = ed[k].float().numpy().astype(np.float64)
                try:
                    tw[k] = spectral_tw_control(wb, wa, v)
                except Exception as e:
                    tw[k] = {"error": str(e)}
                u, s, _ = np.linalg.svd(ed[k].float().numpy() - wb,
                                        full_matrices=False)
                u1 = u[:, 0] if u.shape[1] else None
                if u1 is not None:
                    cosines[k] = round(float(abs(np.dot(u1, v))), 4)

    rows = []
    for gid, p, kind in bad + good:
        ids = gen_ids(model, tok, p)
        labels = token_loop_labels(ids)
        n = len(ids)
        ls = float(labels.mean()) if n else 0.0
        collapse = int(np.argmax(labels)) if labels.any() else None
        waste = (n - collapse) if collapse is not None else 0
        rows.append({"id": gid, "kind": kind, "n_tokens": n,
                     "loop_share": round(ls, 4), "looping": ls >= 0.5,
                     "loop_waste": waste})
        print(f"[{a.tag}] {gid} {kind}: ls={ls:.2f} waste={waste}", flush=True)

    def frac(sub, key):
        sub_r = [r for r in rows if r["kind"] == sub]
        return round(sum(r[key] for r in sub_r) / len(sub_r), 4)

    summary = {
        "tag": a.tag, "edited_state": a.edited_state,
        "n_changed_tensors": len(changed), "changed": changed,
        "tw": {k: (v if "verdict" not in v else {"verdict": v["verdict"]})
               for k, v in tw.items()},
        "cosine_edit_vs_v": cosines,
        "rho_bad": frac("bad", "looping"),
        "rho_good": frac("good", "looping"),
        "loop_waste_mean": round(float(np.mean([r["loop_waste"] for r in rows
                                                if r["kind"] == "bad"])), 2),
        "kl_ctrl": round(kl_ctrl(model, tok, ctrl_texts, base_logits), 4),
    }
    json.dump(summary, open(a.out + f"_eval_{a.tag}.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items()
                      if k != "changed"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
