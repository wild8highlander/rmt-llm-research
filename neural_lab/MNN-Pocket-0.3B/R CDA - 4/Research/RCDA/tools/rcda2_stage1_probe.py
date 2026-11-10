#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RCDA-2 stage 1 — behavioral contrast probe (my own plan, maximum creativity).

Key idea: stop labeling prompts, label TOKENS. The baseline model itself is the
labeler: a generated token is "in a loop" if its 4-gram (token-level) already
occurred earlier in the same generation. Then

    Delta_h_behavioral^(l) = mu_loop^(l) - mu_clean^(l)

contrasts tokens from the looping regime with tokens from clean generations of
the SAME prompt sets (bad and good are both present in both pools), which
removes the style/topic confound of the round-1 contrast (long CoT vs short
factual).

Also produced here:
  - cosine(style Delta_h, behavioral Delta_h) at the round-1 candidate layers
    (quantitative confound test);
  - token attribution: tied embeddings E give Delta_logits ~ E @ Delta_h, so the
    direction can be read as "which tokens does the loop direction boost";
  - spectral precursors: sliding-window participation ratio PR = (sum s^2)^2 /
    sum s^4 of the token-window SVD, aligned at the collapse onset.

Outputs: rcda2_work/rcda2_probe.json, vectors, and the prompt/generation log.
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

from pocket_rmt_probe import _CovAcc, _read_prompt_set  # noqa: E402
from rmt_intervention import RCDA_MATRICES, matrix_scan, select_columns, state_dict_key  # noqa: E402
from marchenko_pastur import mp_upper_edge, rmt_z_score  # noqa: E402
from benchmark_rmt import _defect_flags, _load_model_dir  # noqa: E402

MODEL_DIR = os.path.join(WORK, "model")
OUT = os.path.join(WORK, "rcda2")
os.makedirs(OUT, exist_ok=True)

MAX_NEW = 96
POS_LO = 8            # position window for both token pools
WINDOW = 16           # sliding window for the precursor analysis


class CovAcc32:
    """Memory-diet covariance accumulator: second moments in float32
    (28 layers x 896^2 fp64 would not fit in the 4 GB sandbox)."""

    def __init__(self, H):
        self.H = H
        self.n = 0
        self.s1 = np.zeros(H, dtype=np.float64)
        self.s2 = np.zeros((H, H), dtype=np.float32)

    def add(self, X):
        X = X.astype(np.float32, copy=False)
        self.n += X.shape[0]
        self.s1 += X.sum(axis=0, dtype=np.float64)
        self.s2 += X.T @ X

    def moments(self):
        if self.n < 2:
            return None, None
        mu = self.s1 / self.n
        s2 = self.s2.astype(np.float64)
        cov = (s2 - self.n * np.outer(mu, mu)) / (self.n - 1)
        return mu, cov


def pooled_moments(a, b):
    """Moments of the union sample of two CovAcc32 without copying."""
    n = a.n + b.n
    mu = (a.s1 + b.s1) / n
    s2 = a.s2.astype(np.float64) + b.s2.astype(np.float64)
    cov = (s2 - n * np.outer(mu, mu)) / (n - 1)
    return mu, cov, n


def token_loop_labels(ids, ngram=4):
    """Per-token boolean: the n-gram ending here already appeared earlier."""
    labels = np.zeros(len(ids), dtype=bool)
    seen = {}
    for t in range(len(ids)):
        if t + 1 >= ngram:
            g = tuple(ids[t - ngram + 1:t + 1])
            if g in seen and seen[g] <= t - ngram:   # non-overlapping earlier hit
                labels[t] = True
            if g not in seen:
                seen[g] = t
    return labels


@torch.no_grad()
def generate_and_probe(model, tok, items, tag, acc, gens_file):
    """Generate, label tokens, accumulate per-layer covariances into the SHARED
    acc (both prompt sets feed the same loop/clean pools) and stream the
    per-generation records to a JSONL file (crash-safe)."""
    L = model.config.num_hidden_layers
    t0 = time.time()
    for i, it in enumerate(items):
        enc = tok(it["text"], return_tensors="pt")
        n_in = enc["input_ids"].shape[1]
        out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        ids = out[0, n_in:].tolist()
        if tok.eos_token_id in ids:
            ids = ids[:ids.index(tok.eos_token_id) + 1]
        labels = token_loop_labels(ids)
        collapse = int(np.argmax(labels)) if labels.any() else None
        clean_gen = collapse is None
        text = tok.decode(ids, skip_special_tokens=True)
        flags = _defect_flags(text, len(ids), MAX_NEW, 0.2, 0.8)

        # one forward over prompt+generation, hidden states of generated tokens
        hs = model(input_ids=out[0][None, :n_in + len(ids)],
                   output_hidden_states=True).hidden_states  # tuple len L+1
        for l in range(L):
            X = hs[l + 1][0, n_in:n_in + len(ids)].float().numpy()  # block l output
            pos = np.arange(len(ids))
            loop_mask = labels.astype(bool) & (pos >= POS_LO)
            if loop_mask.any():
                acc["loop"][l].add(X[loop_mask])
            if clean_gen:
                cm = (~labels) & (pos >= POS_LO)
                if cm.any():
                    acc["clean"][l].add(X[cm])
        gens = {"id": it["id"], "split": tag, "prompt": it["text"],
                "gen_ids": ids, "labels": [int(x) for x in labels],
                "collapse": collapse, "clean_gen": clean_gen,
                "text": text, **flags}
        with open(gens_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(gens, ensure_ascii=False) + "\n")
        print(f"  [{tag} {i+1}/{len(items)}] {it['id']}: ntok={len(ids)} "
              f"collapse={collapse} hit={flags['defect_hit']} "
              f"({time.time()-t0:.0f}s)", flush=True)
        del hs
    return


def pool_probe(acc, H, L, style_vecs):
    """Per-layer Delta_h_behavioral + RMT passport + cosine vs the style vector.
    Frees each layer's accumulators after use and returns (stats, dh_by_layer)."""
    layers, dh_by_layer = [], {}
    for l in range(L):
        mu_l, cov_l = acc["loop"][l].moments()
        mu_c, cov_c = acc["clean"][l].moments()
        _, cov_all, N_all = pooled_moments(acc["loop"][l], acc["clean"][l])
        Nl, Nc = acc["loop"][l].n, acc["clean"][l].n
        acc["loop"][l] = acc["clean"][l] = None       # memory diet
        if mu_l is None or mu_c is None or N_all < 2:
            continue
        dh = mu_l - mu_c
        dh_by_layer[l] = dh
        eigs, U = np.linalg.eigh(cov_all)
        eigs = eigs[::-1].clip(min=0)
        U = U[:, ::-1]
        lam1, u1 = float(eigs[0]), U[:, 0]
        sigma2 = float(np.median(eigs))
        c = H / N_all
        lam_p = mp_upper_edge(sigma2, c)
        cov_p = (((Nl - 1) * cov_l + (Nc - 1) * cov_c) / (Nl + Nc - 2)
                 if Nl > 1 and Nc > 1 else cov_all)
        ridge = cov_p + 1e-6 * np.trace(cov_p) / H * np.eye(H)
        t2 = float(Nl * Nc / (Nl + Nc) * (dh @ np.linalg.solve(ridge, dh)))
        cos = {}
        for name, vec in style_vecs.items():
            v = vec / max(np.linalg.norm(vec), 1e-12)
            cos[name] = float(v @ dh / max(np.linalg.norm(dh), 1e-12))
        layers.append({
            "layer": l, "n_loop": Nl, "n_clean": Nc, "n_all": N_all,
            "q": c, "sigma2_bulk": sigma2, "lambda_max": lam1,
            "lambda_plus_mp": float(lam_p), "z_rmt": rmt_z_score(lam1, lam_p, N_all),
            "delta_h_norm": float(np.linalg.norm(dh)),
            "beta": float(dh @ u1), "beta_noise_floor": float(np.sqrt(lam1 / N_all)),
            "hotelling_t2": t2, "z_diff": float((t2 - H) / np.sqrt(2 * H)),
            "cosine_vs_style": cos,
        })
        del cov_all, cov_l, cov_c, cov_p, ridge, eigs, U
    return layers, dh_by_layer


def token_attribution(model, tok, dh, top=15):
    """Delta_logits ~ E @ Delta_h (tied embeddings, up to the final RMSNorm)."""
    E = model.model.embed_tokens.weight.detach().float().numpy()
    score = E @ dh
    top_i = np.argsort(-score)[:top]
    bot_i = np.argsort(score)[:top]
    return {
        "boosted": [(tok.decode([int(i)]).strip() or repr(int(i)), float(score[i]))
                    for i in top_i],
        "suppressed": [(tok.decode([int(i)]).strip() or repr(int(i)), float(score[i]))
                       for i in bot_i],
    }


@torch.no_grad()
def rebuild_accumulators(model, tok, gens, acc):
    """Resume path: rebuild the loop/clean accumulators from saved generations
    (deterministic greedy generations are reused, not regenerated)."""
    L = model.config.num_hidden_layers
    for gi, gen in enumerate(gens):
        ids = gen["gen_ids"]
        labels = np.array(gen["labels"], dtype=bool)
        enc = tok(gen["prompt"], return_tensors="pt")
        n_in = enc["input_ids"].shape[1]
        hs = model(input_ids=torch.cat([enc["input_ids"],
                                        torch.tensor([ids], dtype=torch.long)], dim=1),
                   output_hidden_states=True).hidden_states
        for l in range(L):
            X = hs[l + 1][0, n_in:n_in + len(ids)].detach().float().numpy()
            pos = np.arange(len(ids))
            loop_mask = labels & (pos >= POS_LO)
            if loop_mask.any():
                acc["loop"][l].add(X[loop_mask])
            if gen["clean_gen"]:
                cm = (~labels) & (pos >= POS_LO)
                if cm.any():
                    acc["clean"][l].add(X[cm])
        del hs
        if (gi + 1) % 10 == 0:
            print(f"  [resume {gi+1}/{len(gens)}]", flush=True)


def main():
    t_start = time.time()
    model, tok = _load_model_dir(MODEL_DIR)
    print("model loaded", flush=True)
    H = model.config.hidden_size
    L = model.config.num_hidden_layers

    bad = _read_prompt_set(os.path.join(WORK, "prompts/bad.jsonl"))
    good = _read_prompt_set(os.path.join(WORK, "prompts/good.jsonl"))

    gens_file = os.path.join(OUT, "rcda2_generations.jsonl")
    acc = {"loop": [CovAcc32(H) for _ in range(L)],
           "clean": [CovAcc32(H) for _ in range(L)]}
    if os.path.exists(gens_file):
        gens = [json.loads(line) for line in open(gens_file, encoding="utf-8")]
    else:
        gens = []
    if len(gens) >= len(bad) + len(good):
        print(f"resume: reusing {len(gens)} saved generations", flush=True)
        rebuild_accumulators(model, tok, gens, acc)
    else:
        open(gens_file, "w").close()
        generate_and_probe(model, tok, bad, "bad", acc, gens_file)
        generate_and_probe(model, tok, good, "good", acc, gens_file)
        gens = [json.loads(line) for line in open(gens_file, encoding="utf-8")]
    n_loop = sum(1 for g in gens if not g["clean_gen"])
    n_loop_tok, n_clean_tok = acc["loop"][0].n, acc["clean"][0].n
    print(f"\ngenerations: {len(gens)}, looping: {n_loop}, "
          f"loop tokens: {n_loop_tok}, clean tokens: {n_clean_tok}", flush=True)

    # ------------------------------------------------------------------ probe
    style_vecs = {}
    for f in os.listdir(os.path.join(WORK, "rcda_results/vectors")):
        if f.endswith("_delta_h.pt"):
            l = int(f.split("L")[1].split("_")[0])
            style_vecs[f"L{l}"] = torch.load(
                os.path.join(WORK, "rcda_results/vectors", f),
                weights_only=True).numpy().astype(np.float64)
    probe_layers, dh_by_layer = pool_probe(acc, H, L, style_vecs)
    del acc
    gc.collect()
    rank = sorted(probe_layers, key=lambda d: -d["z_diff"])
    top = [d["layer"] for d in rank[:4]]
    print("\nRCDS-2 top layers by z_diff:",
          [(d["layer"], round(d["z_diff"], 1)) for d in rank[:6]], flush=True)
    for d in rank[:4]:
        print(f"  L{d['layer']}: z_diff={d['z_diff']:.1f} |dh|={d['delta_h_norm']:.2f} "
              f"cos_vs_style={d['cosine_vs_style']}", flush=True)

    # -------------------------------------------------- matrix scans + vectors
    os.makedirs(os.path.join(OUT, "vectors"), exist_ok=True)
    sd = model.state_dict()
    scans = []
    for l in top:
        dh = dh_by_layer[l]
        torch.save(torch.from_numpy(dh.astype(np.float32)),
                   os.path.join(OUT, "vectors", f"L{l}_delta_h_behav.pt"))
        v = dh / max(np.linalg.norm(dh), 1e-12)
        for matrix in RCDA_MATRICES:
            key = state_dict_key(l, matrix)
            scan = matrix_scan(sd[key].detach().float().numpy(), v)
            previews = {}
            for mode in ("auto_tw", "auto_chi2"):
                idx, info = select_columns(scan["profile"], scan["sigma2_w"],
                                           scan["lambda_plus_w"], scan["shape"][0],
                                           mode=mode, tau=2.0)
                previews[mode] = {"n_columns": info["n_columns"],
                                  "top10": [float(x) for x in scan["profile"][idx[:10]]]}
            scans.append({"layer": l, "matrix": matrix, "state_dict_key": key,
                          "scan": {k: scan[k] for k in scan if k != "profile"},
                          "column_previews": previews,
                          "vec_file": f"vectors/L{l}_delta_h_behav.pt"})
            print(f"  [scan] L{l} {matrix}: lam_v/lam+={scan['lambda_over_edge']:.2f} "
                  f"theta_v={scan['theta_v']:.3f} theta_c={scan['theta_c']:.3f} "
                  f"delta={scan['delta']:.3f} alpha0={scan['alpha0_kappa0.5']:.3f} "
                  f"cols(tw)={previews['auto_tw']['n_columns']} "
                  f"cols(chi2)={previews['auto_chi2']['n_columns']}", flush=True)

    # ------------------------------------------------------ token attribution
    attribution = {}
    for l in top[:2]:
        attribution[f"L{l}"] = token_attribution(model, tok, dh_by_layer[l])
        print(f"  [attrib L{l}] boosted:", [t for t, _ in attribution[f'L{l}']['boosted'][:8]],
              flush=True)

    # ------------------------------------------------------------- precursors
    sel = top[:3]
    pr = {"pre": {l: [] for l in sel}, "loop": {l: [] for l in sel},
          "clean": {l: [] for l in sel}}
    n_prec = 0
    with torch.no_grad():
        for gen in gens:
            ids = gen["gen_ids"]
            if len(ids) < WINDOW + 4:
                continue
            enc = tok(gen["prompt"], return_tensors="pt")
            n_in = enc["input_ids"].shape[1]
            hs = model(input_ids=torch.cat([enc["input_ids"],
                                            torch.tensor([ids], dtype=torch.long)], dim=1),
                       output_hidden_states=True).hidden_states
            n_prec += 1
            for l in sel:
                X = hs[l + 1][0, n_in:n_in + len(ids)].detach().float().numpy()
                def pr_at(t):
                    s = np.linalg.svd(X[max(0, t - WINDOW + 1):t + 1], compute_uv=False)
                    p = s ** 2
                    return (float((p.sum()) ** 2 / max((p ** 2).sum(), 1e-12)),
                            float(p[0] / max(p.sum(), 1e-12)))
                if gen["collapse"] is not None:
                    c = gen["collapse"]
                    for t in range(max(POS_LO, c - 24), c):          # before collapse
                        if t >= WINDOW:
                            pr["pre"][l].append((t - c, *pr_at(t)))
                    for t in range(c, min(len(ids), c + 32)):        # looping regime
                        pr["loop"][l].append((t - c, *pr_at(t)))
                elif gen["clean_gen"]:
                    for t in range(WINDOW, min(len(ids), 72)):       # clean control
                        pr["clean"][l].append((t, *pr_at(t)))
            del hs
    print(f"precursor windows done over {n_prec} generations "
          f"({time.time()-t_start:.0f}s total)", flush=True)

    # ------------------------------------------------------------------- save
    report = {
        "meta": {"max_new": MAX_NEW, "pos_lo": POS_LO, "window": WINDOW,
                 "n_generations": len(gens), "n_looping": n_loop,
                 "n_tokens": {"loop": n_loop_tok, "clean": n_clean_tok},
                 "elapsed_s": round(time.time() - t_start, 1)},
        "layers": probe_layers,
        "ranking_by_z_diff": [d["layer"] for d in rank],
        "top_layers": top,
        "matrix_scans": scans,
        "token_attribution": attribution,
        "precursor": {k: {str(l): v for l, v in pr[k].items()} for k in pr},
        "generations": [{k: g[k] for k in ("id", "split", "collapse", "clean_gen",
                                           "n_tokens", "rep4", "defect_hit", "text")}
                        for g in gens],
    }
    with open(os.path.join(OUT, "rcda2_probe.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1, default=float)
    print(f"\nsaved {OUT}/rcda2_probe.json", flush=True)


if __name__ == "__main__":
    main()
