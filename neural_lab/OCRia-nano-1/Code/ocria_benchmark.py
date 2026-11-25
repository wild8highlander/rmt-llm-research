"""The one evaluator every OCRia arm ends in (the MNN/Gemma protocol, ported).

For one arm (base / classic / rcda_x) it measures, in one protocol:
  * F1 phantom_rate / phantom_char_rate / blank specificity
  * F2 mean loop_share + rho_loop on the emitted strings (char 4-gram, thr 0.5)
  * F3 CER on clean / degraded / control + exact-match rate (abilities)
  * F6 confidence-vs-geometry AUC on the phantom frames (Utility Trap)
  * logits MSE + frame-level KL(base||arm) on the control corpus (parity)
  * per edited tensor: rel_frobenius, relative TW gate (dKS vs base),
    cos(u1 of the edit, v) alignment with the probe direction

F4 (routing) and F5 (flip) live in the frozen router - identical across arms,
reported from the probe.  The detector side of F1 is likewise frozen.

Usage:
  python ocria_benchmark.py --arm base
  python ocria_benchmark.py --arm classic --state ../Results/classic_trained_state.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from marchenko_pastur import ks_distance, wishart_eigs, fit_sigma2
from ocria_common import (BATCHES, KIT_DIR, NONTEXT_BATCHES, RESULTS_DIR,
                          cer, ctc_decode, dump_json, load_batch, load_image,
                          loop_stats, rho_of_texts, seed_all)
from rebuild_latin import LatinRecognizer, extract_latin_tensors

CONF_EMIT = 0.5


def evaluate_arm(model: LatinRecognizer, base_model: LatinRecognizer,
                 arm: str) -> dict:
    """Full protocol for one arm.  Returns the eval dict (also written out)."""
    # ------------------------------------------------ per-crop generations
    gens = {}
    for batch in BATCHES:
        rows = []
        for r in load_batch(batch):
            crop = load_image(KIT_DIR / r["file"])
            with torch.no_grad():
                logits = model(torch.tensor(crop[None, None]))
                probs = F.softmax(logits, dim=1)[0]
                conf, ids = probs.max(dim=0)
            ids = ids.numpy(); conf = conf.numpy()
            text = ctc_decode(ids)
            st = loop_stats(text)
            row = {"batch": batch, "id": r["id"], "text": text,
                   "n_chars": len(text),
                   "mean_conf": round(float(conf[ids != 0].mean()) if (ids != 0).any()
                                      else float(conf.mean()), 4),
                   "loop_share": round(st["loop_share"], 4), "gt": r["text"]}
            if r["text"]:
                row["cer"] = round(cer(text, r["text"]), 4)
            rows.append(row)
        gens[batch] = rows

    # ------------------------------------------------ family metrics
    nontext = gens["structured"] + gens["noise"] + gens["blank"]
    phantom = [r for r in nontext if r["n_chars"] > 0]
    fam = {
        "F1_phantom_rate": round(len(phantom) / len(nontext), 4),
        "F1_phantom_char_rate": round(float(np.mean([r["n_chars"] for r in nontext])), 4),
        "F1_blank_specificity": round(sum(1 for r in gens["blank"] if r["n_chars"] == 0) / len(gens["blank"]), 4),
        "F2_loop": rho_of_texts([r["text"] for r in phantom]),
        "F3_cer_clean": round(float(np.mean([r["cer"] for r in gens["clean"]])), 4),
        "F3_cer_degraded": round(float(np.mean([r["cer"] for r in gens["degraded"]])), 4),
        "F3_cer_control": round(float(np.mean([r["cer"] for r in gens["control"]])), 4),
        "F3_exact_match_clean": round(float(np.mean([1.0 if r["text"] == r["gt"] else 0.0
                                                     for r in gens["clean"]])), 4),
    }

    # ------------------------------------------------ F6 utility trap
    v_file = RESULTS_DIR / "ocria_vectors.npz"
    vecs = {}
    if v_file.exists():
        with np.load(v_file) as z:
            vecs = {k[2:]: z[k] for k in z.files}
    confs, geoms, labs = [], [], []
    if "head" in vecs:
        v_head = vecs["head"] / (np.linalg.norm(vecs["head"]) + 1e-12)
        v_mix2 = vecs.get("mix2")
        for r in nontext:
            crop = load_image(KIT_DIR / next(x["file"] for x in
                                             (load_batch(r["batch"])) if x["id"] == r["id"]))
            with torch.no_grad():
                logits = model(torch.tensor(crop[None, None]))
                probs = F.softmax(logits, dim=1)[0]
                conf, ids = probs.max(dim=0)
                model(torch.tensor(crop[None, None]))
                h = model.stage_maps["mix2"]
                if h.dim() == 4:
                    h = h.squeeze(2)
                h = h[0].numpy().T
                lg = logits[0].numpy().T
            for t in range(len(ids)):
                if ids[t] != 0 and conf[t] >= CONF_EMIT:
                    labs.append(1); confs.append(float(conf[t]))
                    geoms.append(float(lg[t] @ v_head))
                elif ids[t] == 0 and conf[t] >= CONF_EMIT:
                    labs.append(0); confs.append(float(conf[t]))
                    geoms.append(float(lg[t] @ v_head))
    from ocria_probe import auc_score
    labs = np.array(labs); confs = np.array(confs); geoms = np.array(geoms)
    if labs.size and (labs == 1).any() and (labs == 0).any():
        fam["F6_auc_confidence"] = round(auc_score(confs[labs == 1], confs[labs == 0]), 4)
        fam["F6_auc_geometry"] = round(auc_score(geoms[labs == 1], geoms[labs == 0]), 4)
        fam["F6_n_frames"] = int(labs.size)

    # ------------------------------------------------ abilities parity
    mse_rows, kl_rows = [], []
    for r in load_batch("control"):
        crop = load_image(KIT_DIR / r["file"])
        with torch.no_grad():
            la = model(torch.tensor(crop[None, None]))[0]        # [792,T]
            lb = base_model(torch.tensor(crop[None, None]))[0]
        mse_rows.append(float(F.mse_loss(la, lb)))
        pa = F.softmax(la.double(), dim=0)
        pb = F.softmax(lb.double(), dim=0)
        kl_rows.append(float((pa * ((pa + 1e-12).log() - (pb + 1e-12).log())).sum(0).mean()))
    fam["control_logits_mse"] = round(float(np.mean(mse_rows)), 6)
    fam["control_kl_base_to_arm"] = round(float(np.mean(kl_rows)), 6)

    result = {"arm": arm, **fam, "generations": gens}
    return result


def spectral_section(model, base_model, state: dict, vecs) -> dict:
    """Per edited tensor: rel_frobenius, relative TW gate, cos(u1, v)."""
    base_sd = dict(base_model.named_parameters())
    out = {}
    stage_of = {"mix_fc2.0.weight": "mix0", "mix_fc2.1.weight": "mix1",
                "mix_fc2.2.weight": "mix2"}
    for name, w in state.items():
        W_t = w.numpy().reshape(w.shape[0], -1).astype(np.float64)
        W_b = base_sd[name].detach().numpy().reshape(w.shape[0], -1).astype(np.float64)
        dW = W_t - W_b
        rel = float(np.linalg.norm(dW) / (np.linalg.norm(W_b) + 1e-12))
        # relative TW gate: bulk KS(trained) - KS(base)
        def bulk_ks(W):
            lam, q = wishart_eigs(W)
            lam_s = np.sort(lam)[::-1]
            s2 = fit_sigma2(lam_s)
            return ks_distance(lam_s, min(q, 1.0), s2)
        d_ks = round(bulk_ks(W_t) - bulk_ks(W_b), 4)
        # alignment: u1 of the edit vs the probe direction of its output space
        U, S, Vh = np.linalg.svd(dW, full_matrices=False)
        stage = stage_of.get(name)
        cos_v = None
        if stage in vecs and vecs[stage].size == W_t.shape[0]:
            v = vecs[stage] / (np.linalg.norm(vecs[stage]) + 1e-12)
            cos_v = round(float(abs(U[:, 0] @ v)), 4)
        out[name] = {"rel_frobenius": round(rel, 5), "n_weights": int(dW.size),
                     "tw_gate_dks": d_ks,
                     "tw_verdict": "pass" if abs(d_ks) < 0.05 else "violation",
                     "cos_u1_v": cos_v,
                     "sigma1_share": round(float(S[0] ** 2 / max((S ** 2).sum(), 1e-12)), 4)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["base", "classic", "rcda_x"])
    ap.add_argument("--state", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    seed_all(args.seed)

    model = LatinRecognizer(extract_latin_tensors())
    base_model = LatinRecognizer(extract_latin_tensors())
    model.eval(); base_model.eval()

    state = {}
    if args.arm != "base":
        if not args.state:
            args.state = str(RESULTS_DIR / f"{args.arm}_trained_state.pt")
        state = torch.load(args.state, map_location="cpu")
        with torch.no_grad():
            sd = dict(model.named_parameters())
            for name, w in state.items():
                sd[name].copy_(w.float())

    with np.load(RESULTS_DIR / "ocria_vectors.npz") as z:
        vecs = {k[2:]: z[k] for k in z.files}

    res = evaluate_arm(model, base_model, args.arm)
    gens = res.pop("generations")
    res["n_changed_tensors"] = len(state)
    if state:
        res["spectral"] = spectral_section(model, base_model, state, vecs)

    out = RESULTS_DIR / f"eval_{args.arm}.json"
    dump_json(res, out)
    rows = [dict(r, batch=b) for b, rs in gens.items() for r in rs]
    with open(RESULTS_DIR / f"eval_{args.arm}_rows.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[bench:{args.arm}] F1 phantom={res['F1_phantom_rate']} "
          f"(char {res['F1_phantom_char_rate']}) spec={res['F1_blank_specificity']} | "
          f"F3 cer_clean={res['F3_cer_clean']} cer_degr={res['F3_cer_degraded']} | "
          f"mse={res['control_logits_mse']} kl={res['control_kl_base_to_arm']}")
    if "spectral" in res:
        for n, s in res["spectral"].items():
            print(f"    {n}: rel_frob={s['rel_frobenius']} dKS={s['tw_gate_dks']} "
                  f"cos(u1,v)={s['cos_u1_v']} verdict={s['tw_verdict']}")
    print(f"[bench:{args.arm}] wrote {out.name} + rows jsonl")


if __name__ == "__main__":
    main()
