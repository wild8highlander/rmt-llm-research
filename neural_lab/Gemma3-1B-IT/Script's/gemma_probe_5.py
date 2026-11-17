#!/usr/bin/env python3
"""gemma_probe.py — stage 1: behavioral + spectral probing of the repetition
defect on Gemma3-1B-IT (the RCDA-5 analog of pocket_rmt_probe.py).

For every bad prompt: greedy generation (96 tokens, chat template), token-level
n-gram loop labeling; for looping trajectories a cacheless forward with
output_hidden_states pools

    A_l = residual states h_l at generated positions INSIDE the loop (t >= c)
    B_l = residual states h_l at generated positions BEFORE the loop (t < c)

and extracts, per decoder layer l (0..25):

    dh_l        = mean(A_l) - mean(B_l)                     (behavioral delta)
    v_l         = dh_l / ||dh_l||                           (loop direction)
    z_diff_l    = diagonal Mahalanobis separability of A vs B along dh_l
    beta_l      = Rayleigh spike strength of the loop-state covariance
                  (lambda_1 of Cov(A) vs the MP edge, i.e. how far the loop
                  states are supercritical with respect to random-matrix bulk)

Writer layers (the analog of MNN-Pocket o_proj L15/16/18) are selected by the
deterministic sliding-window rule: the window of 5 consecutive layers with the
highest sum of z_diff, then the 3 best layers inside that window.

Outputs (under $GEMMA_WORK/probe/):
    probe_report.json     per-layer profile, looping generations, writers
    probe_generations.jsonl
    vectors_all.npz       v_l for every layer
    v_L{N}.npy            loop direction per selected writer layer
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gemma_lab as L                                    # noqa: E402


def diag_mahalanobis(dh: np.ndarray, A: np.ndarray, B: np.ndarray) -> float:
    """Separability of class means along dh: sum_k dh_k^2 / pooled_var_k."""
    var_a = A.var(axis=0) if len(A) > 1 else np.ones_like(dh)
    var_b = B.var(axis=0) if len(B) > 1 else np.ones_like(dh)
    pooled = 0.5 * (var_a + var_b) + 1e-8
    return float((dh ** 2 / pooled).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-1b-it")
    ap.add_argument("--max-new", type=int, default=L.MAX_NEW)
    ap.add_argument("--pool-threshold", type=float, default=0.3,
                    help="loop_share needed to enter the v-pool")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else L.resolve_work() / "probe"
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM
    import torch

    torch.set_num_threads(max(1, __import__("os").cpu_count() or 2))
    tok = L.load_tokenizer(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.eval()

    bad = L.load_bad_prompts()
    n_layers = model.config.num_hidden_layers
    print(f"probe: {len(bad)} bad prompts, {n_layers} layers", flush=True)

    rows = []
    gens_path = out_dir / "gens.jsonl"
    done = {}
    if gens_path.exists():
        for line in gens_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["id"]] = r
        print(f"resume: {len(done)} generations already on disk", flush=True)
    for pid, text in bad:
        if pid in done:
            r = done[pid]
            rows.append({"id": pid, "text": text, "gen_ids": r["gen_ids"],
                         "loop_share": r["loop_share"], "collapse": r["collapse"],
                         "n_tok": len(r["gen_ids"])})
            continue
        t0 = time.time()
        ids = L.gen_ids(model, tok, text, args.max_new)
        ls = L.loop_share(ids)
        c = L.loop_collapse_point(ids)
        rows.append({"id": pid, "text": text, "gen_ids": ids, "loop_share": ls,
                     "collapse": c, "n_tok": len(ids)})
        with gens_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"id": pid, "gen_ids": ids, "loop_share": ls,
                                "collapse": c}, ensure_ascii=False) + "\n")
        print(f"[gen] {pid}: ls={ls:.2f} collapse={c} ({time.time()-t0:.0f}s)",
              flush=True)

    pool = [r for r in rows if r["loop_share"] >= args.pool_threshold
            and r["collapse"] is not None]
    gens_meta_path = out_dir / "probe_generations.jsonl"
    with gens_meta_path.open("w", encoding="utf-8") as f:
        for r in rows:
            rec = {k: v for k, v in r.items() if k != "gen_ids"}
            rec["text_out"] = tok.decode(r["gen_ids"], skip_special_tokens=True)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    rho_mini = float(np.mean([1.0 if r["loop_share"] >= L.LOOP_THRESHOLD else 0.0
                              for r in rows]))
    print(f"probe pool: {len(pool)}/{len(rows)} prompts loop (>= {args.pool_threshold}); "
          f"rho_mini(strict)={rho_mini}", flush=True)

    profile = {l: {"n_loop": 0, "n_pre": 0, "z_diff": 0.0, "beta_rayleigh": 0.0,
                   "dh_norm": 0.0} for l in range(n_layers)}
    dh_store: dict[int, np.ndarray] = {}

    if pool:
        # collect hidden states per looping prompt (one cacheless forward each)
        A_all: dict[int, list] = {l: [] for l in range(n_layers)}
        B_all: dict[int, list] = {l: [] for l in range(n_layers)}
        for r in pool:
            plen = int(L.encode_prompt(tok, r["text"]).shape[1])
            full = torch.cat([L.encode_prompt(tok, r["text"])[0],
                              torch.tensor(r["gen_ids"])], dim=0).unsqueeze(0)
            with torch.no_grad():
                out = model(full, output_hidden_states=True)
            hs = out.hidden_states                       # hs[l+1] = after layer l
            c = r["collapse"]                            # index inside generation
            for l in range(n_layers):
                h = hs[l + 1][0].float().numpy()
                gen = h[plen:]                           # generated positions
                lo = min(c, len(gen) - 2)                # keep >= 1 pre token
                A_all[l].append(gen[lo:])
                B_all[l].append(gen[:max(lo, 1)])
            del out, hs

        for l in range(n_layers):
            A = np.concatenate(A_all[l], axis=0)
            B = np.concatenate(B_all[l], axis=0)
            dh = A.mean(axis=0) - B.mean(axis=0)
            z = diag_mahalanobis(dh, A, B)
            # supercriticality of the loop states vs the MP edge (q = H/T)
            T = A.shape[0]
            C = np.cov(A.T) + 1e-6 * np.eye(A.shape[1])
            lam = np.linalg.eigvalsh(C)
            from marchenko_pastur import mp_bounds, mp_fit_sigma2, \
                rayleigh_spike_strength
            q = A.shape[1] / T
            sigma2 = mp_fit_sigma2(lam, q)
            lam_plus = mp_bounds(q, sigma2)[1]
            beta = rayleigh_spike_strength(float(lam[-1]), lam_plus)
            profile[l] = {"n_loop": int(A.shape[0]), "n_pre": int(B.shape[0]),
                          "z_diff": round(z, 3),
                          "beta_rayleigh": round(beta, 3),
                          "dh_norm": round(float(np.linalg.norm(dh)), 5)}
            dh_store[l] = dh
        zd = np.array([profile[l]["z_diff"] for l in range(n_layers)])
        # sliding window of 5 consecutive layers, 3 best inside it
        win = np.array([zd[l:l + 5].sum() for l in range(n_layers - 4)])
        w0 = int(np.argmax(win))
        inner = sorted(range(w0, min(w0 + 5, n_layers)),
                       key=lambda l: -zd[l])[:3]
        writers = sorted(inner)
        print(f"writer layers: {writers} "
              f"(z_diff: {[profile[l]['z_diff'] for l in writers]})", flush=True)
    else:
        writers = []
        print("NO LOOPING PROMPTS FOUND — defect does not manifest; "
              "see probe_generations.jsonl", flush=True)

    vecs = {f"v_L{l}": (dh_store[l] / max(np.linalg.norm(dh_store[l]), 1e-9)
                        if l in dh_store else np.zeros(1152))
            for l in range(n_layers)}
    np.savez(out_dir / "vectors_all.npz", **vecs)
    for l in writers:
        np.save(out_dir / f"v_L{l}.npy", vecs[f"v_L{l}"])

    report = {
        "model": args.model,
        "n_prompts": len(rows), "max_new": args.max_new,
        "pool_threshold": args.pool_threshold,
        "pool_ids": [r["id"] for r in pool],
        "rho_mini_strict": rho_mini,
        "loop_share_per_prompt": {r["id"]: round(r["loop_share"], 3) for r in rows},
        "collapse_per_prompt": {r["id"]: r["collapse"] for r in rows},
        "writers": writers,
        "layer_profile": {str(l): profile[l] for l in range(n_layers)},
    }
    json.dump(report, (out_dir / "probe_report.json").open("w"), indent=1)
    print(f"saved {out_dir}/probe_report.json", flush=True)


if __name__ == "__main__":
    main()
