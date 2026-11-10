#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RCDA-2 stage 2 — behavioral dose-response of alpha and the final edit.

Round 1 taught us that the BBP-calibrated alpha (0.15/0.42) was surgically safe
(TW pass, PPL unchanged) but too gentle to flip a single greedy trajectory
(0/20 generations changed). Stage 2 measures the actual dose-response curve:

    for alpha in grid: apply the behavioral-direction edit in memory
      -> mean KL(base || edited) on teacher-forced continuations
      -> n_changed greedy trajectories on a calibration subset
      -> rep4 / defect rate on the calibration subset
      -> restore pristine weights

and picks alpha* = the smallest alpha that changes trajectories AND reduces
the repetition defect; falls back to alpha = 1.0 with an escalation note.
Then applies the final edit, runs the TW control, and saves
edited_model_v2 (safetensors + apply_pack-compatible JSON report).
"""
import json
import os
import shutil
import sys
import time

import numpy as np
import torch

WORK = "/home/z/my-project/rcda_work"
sys.path.insert(0, os.path.join(WORK, "Code"))

from rmt_intervention import (  # noqa: E402
    rcda_edit, select_columns, spectral_tw_control, save_edit_pack,
)
from benchmark_rmt import _defect_flags, _load_model_dir  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

MODEL_DIR = os.path.join(WORK, "model")
OUT = os.path.join(WORK, "rcda2")
EDITED_DIR = os.path.join(WORK, "edited_model_v2")
ALPHA_GRID = [0.1, 0.25, 0.5, 0.75, 1.0]
CAL_N = 8
CAL_MAX_NEW = 48


def gen_batch(model, tok, items, max_new):
    rows = []
    for it in items:
        enc = tok(it["text"], return_tensors="pt")
        n_in = enc["input_ids"].shape[1]
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        ids = out[0, n_in:].tolist()
        if tok.eos_token_id in ids:
            ids = ids[:ids.index(tok.eos_token_id) + 1]
        text = tok.decode(ids, skip_special_tokens=True)
        rows.append({"id": it["id"], "ids": ids, "text": text,
                     **_defect_flags(text, len(ids), max_new, 0.2, 0.8)})
    return rows


@torch.no_grad()
def mean_kl(model, tok, sequences, base_logits):
    """KL(base || edited), mean over generated positions of 2 teacher-forced texts."""
    kls = []
    for seq, bl in zip(sequences, base_logits):
        enc = tok(seq["prompt"] + seq["text"], return_tensors="pt")
        logits = model(**enc).logits[0].float()
        n = min(len(seq["ids"]), logits.shape[0] - 1)
        pos = torch.arange(enc["input_ids"].shape[1] - n,
                           enc["input_ids"].shape[1])
        lb = torch.log_softmax(bl[-n:], dim=-1)
        le = torch.log_softmax(logits[pos], dim=-1)
        kl = (lb.exp() * (lb - le)).sum(-1)
        kls.append(float(kl.mean()))
    return float(np.mean(kls))


def main():
    t0 = time.time()
    probe = json.load(open(os.path.join(OUT, "rcda2_probe.json")))
    scans = [s for s in probe["matrix_scans"] if s["scan"]["supercritical"]]
    if not scans:
        scans = sorted(probe["matrix_scans"],
                       key=lambda s: -s["scan"]["lambda_over_edge"])[:2]
    scans = scans[:3]
    print("dose-response targets:", [(s["layer"], s["matrix"]) for s in scans],
          flush=True)

    model, tok = _load_model_dir(MODEL_DIR)
    sd = model.state_dict()

    # pristine copies + behavioral profiles/columns per target
    pristine, plans = {}, []
    for s in scans:
        key = s["state_dict_key"]
        W = sd[key].detach().float()
        pristine[key] = W.clone()
        dh = torch.load(os.path.join(OUT, s["vec_file"]),
                        weights_only=True).numpy().astype(np.float64)
        v = dh / np.linalg.norm(dh)
        # the round-2 scan JSON does not carry the raw profile: recompute it
        profile = v @ W.numpy().astype(np.float64)
        cols, info = select_columns(profile, s["scan"]["sigma2_w"],
                                    s["scan"]["lambda_plus_w"], s["scan"]["shape"][0],
                                    mode="auto_chi2")
        plans.append({"key": key, "layer": s["layer"], "matrix": s["matrix"],
                      "v": v, "cols": [int(i) for i in cols],
                      "scan": s["scan"], "n_cols": info["n_columns"]})
        print(f"  target {key}: cols={info['n_columns']}", flush=True)

    bad = None  # noqa: F841 (kept for symmetry with stage 1; sets come from jsonl)

    # calibration sets: read via the probe's prompt reader
    from pocket_rmt_probe import _read_prompt_set  # noqa: E402
    cal_bad = _read_prompt_set(os.path.join(WORK, "prompts/bad.jsonl"), CAL_N)
    cal_good = _read_prompt_set(os.path.join(WORK, "prompts/good.jsonl"), CAL_N)

    # teacher-forced KL texts from the round-2 generations (1 looping + 1 clean)
    gens = probe["generations"]
    looping = [g for g in gens if not g["clean_gen"]][:1]
    clean = [g for g in gens if g["clean_gen"]][:1]
    id2prompt = {}
    for p in _read_prompt_set(os.path.join(WORK, "prompts/bad.jsonl")) + \
            _read_prompt_set(os.path.join(WORK, "prompts/good.jsonl")):
        id2prompt[p["id"]] = p["text"]
    kl_seqs = [{"prompt": id2prompt[g["id"]], "text": g["text"]}
               for g in (looping + clean)]
    for s in kl_seqs:
        s["ids"] = tok(s["text"], add_special_tokens=False)["input_ids"][:96]
    # teacher-forced KL needs a non-empty continuation; drop empty ones and
    # backfill the clean slot with the longest clean generation available
    kl_seqs = [s for s in kl_seqs if len(s["ids"]) >= 8]
    if len(kl_seqs) < 2:
        extra = sorted((g for g in gens if g["clean_gen"]),
                       key=lambda g: -len(g["text"]))
        for g in extra:
            ids = tok(g["text"], add_special_tokens=False)["input_ids"][:96]
            if len(ids) >= 8 and id2prompt.get(g["id"]):
                kl_seqs.append({"prompt": id2prompt[g["id"]], "text": g["text"],
                                "ids": ids})
                break
    print(f"KL texts: {[len(s['ids']) for s in kl_seqs]} tokens", flush=True)

    @torch.no_grad()
    def base_logits_for(seq):
        enc = tok(seq["prompt"] + seq["text"], return_tensors="pt")
        return model(**enc).logits[0].float()
    base_logits = [base_logits_for(s) for s in kl_seqs]

    # baseline calibration generations
    base_cal = {"bad": gen_batch(model, tok, cal_bad, CAL_MAX_NEW),
                "good": gen_batch(model, tok, cal_good, CAL_MAX_NEW)}
    base_rep4_bad = float(np.mean([r["rep4"] or 0 for r in base_cal["bad"]]))
    base_hit_bad = float(np.mean([r["defect_hit"] for r in base_cal["bad"]]))
    print(f"calibration baseline: rep4_bad={base_rep4_bad:.3f} "
          f"hit_bad={base_hit_bad:.2f}", flush=True)

    # ------------------------------------------------------------ dose loop
    dose = []
    for alpha in ALPHA_GRID:
        for p in plans:                     # apply
            Wn, _ = rcda_edit(pristine[p["key"]], p["v"], alpha,
                              columns=p["cols"] if p["cols"] else None)
            sd[p["key"]].data.copy_(Wn)
        kl = mean_kl(model, tok, kl_seqs, base_logits)
        cal = {"bad": gen_batch(model, tok, cal_bad, CAL_MAX_NEW),
               "good": gen_batch(model, tok, cal_good, CAL_MAX_NEW)}
        changed = {"bad": int(sum(a["text"] != b["text"] for a, b in
                                  zip(cal["bad"], base_cal["bad"]))),
                   "good": int(sum(a["text"] != b["text"] for a, b in
                                   zip(cal["good"], base_cal["good"])))}
        rep4_bad = float(np.mean([r["rep4"] or 0 for r in cal["bad"]]))
        hit_bad = float(np.mean([r["defect_hit"] for r in cal["bad"]]))
        hit_good = float(np.mean([r["defect_hit"] for r in cal["good"]]))
        dose.append({"alpha": alpha, "kl": kl, "changed": changed,
                     "rep4_bad": rep4_bad, "hit_bad": hit_bad,
                     "hit_good": hit_good})
        print(f"  alpha={alpha:.2f}: KL={kl:.4f} changed={changed} "
              f"rep4_bad={rep4_bad:.3f} hit_bad={hit_bad:.2f} "
              f"hit_good={hit_good:.2f} ({time.time()-t0:.0f}s)", flush=True)
        for p in plans:                     # restore
            sd[p["key"]].data.copy_(pristine[p["key"]])

    # ------------------------------------------------------- choice of alpha*
    ok = [d for d in dose if d["changed"]["bad"] >= 2
          and d["rep4_bad"] <= base_rep4_bad - 0.15]
    alpha_star = ok[0]["alpha"] if ok else (dose[-1]["alpha"] if dose else 1.0)
    note = ("smallest alpha with >=2 changed bad trajectories and rep4 -0.15"
            if ok else
            "no alpha in the grid fixed the defect on the calibration set; "
            "alpha=1.0 recorded for the full benchmark (escalation candidate: "
            "more layers or --full ablation)")
    print(f"alpha* = {alpha_star} ({note})", flush=True)

    # ----------------------------------------------------------- final edit
    reports, tensors = [], {}
    for p in plans:
        # TW-aware per-matrix alpha: back off from alpha* until the healthy
        # top spectrum stays within the plan's +-3 TW units (section 7.2)
        chosen = None
        for alpha_p in [alpha_star, 0.75, 0.5, 0.25, 0.1]:
            Wn, n_changed = rcda_edit(pristine[p["key"]], p["v"], alpha_p,
                                      columns=p["cols"] if p["cols"] else None)
            ctrl = spectral_tw_control(pristine[p["key"]].numpy(), Wn.numpy(), p["v"])
            chosen = (alpha_p, Wn, n_changed, ctrl)
            if ctrl["verdict"] == "pass" or alpha_p <= 0.1:
                break
        alpha_p, Wn, n_changed, ctrl = chosen
        rel = float(np.linalg.norm(Wn.numpy() - pristine[p["key"]].numpy())
                    / np.linalg.norm(pristine[p["key"]].numpy()))
        reports.append({
            "layer": p["layer"], "matrix": p["matrix"],
            "state_dict_key": p["key"], "vector": p["v"],
            "alpha": float(alpha_p), "alpha_mode": "dose_response+tw_backoff",
            "columns": {"mode": "auto_chi2", "n_columns": p["n_cols"],
                        "indices": p["cols"]},
            "scan": {k: p["scan"][k] for k in p["scan"]},
            "n_weights_changed": int(n_changed), "rel_frobenius_change": rel,
            "tw_control": ctrl, "status": "applied",
        })
        tensors[p["key"]] = Wn
        print(f"  final edit {p['key']}: alpha={alpha_p} cols={p['n_cols']} "
              f"weights={n_changed} tw={ctrl['verdict']}", flush=True)

    for p, r in zip(plans, reports):        # bake into the model
        sd[p["key"]].data.copy_(tensors[p["key"]])

    os.makedirs(EDITED_DIR, exist_ok=True)
    sd_save = {k: v.detach() for k, v in sd.items()
               if k != "lm_head.weight"}   # tied to embed_tokens (base file omits it)
    save_file(sd_save,
              os.path.join(EDITED_DIR, "model.safetensors"),
              metadata={"format": "pt", "rcda": "2"})
    for fname in os.listdir(MODEL_DIR):
        lp = os.path.join(MODEL_DIR, fname)
        if (os.path.isfile(lp) and not fname.endswith(".safetensors")
                and not fname.startswith("model.safetensors.part")
                and not fname.endswith(".sha256")):
            shutil.copy2(lp, os.path.join(EDITED_DIR, fname))
    save_edit_pack(os.path.join(OUT, "rcda2_edit_report.json"), reports,
                   meta={"round": 2, "alpha_star": alpha_star, "note": note,
                         "dose_response": dose,
                         "base_cal": {"rep4_bad": base_rep4_bad,
                                      "hit_bad": base_hit_bad}})
    print(f"saved {EDITED_DIR}/model.safetensors and {OUT}/rcda2_edit_report.json",
          flush=True)


if __name__ == "__main__":
    main()
