#!/usr/bin/env python3
"""stage1_probe.py — the RCDA-9 spectral probe on the real Gemma3-1B-IT.

A faithful port of the RCDA-8 stage 1 (which itself ports RCDA-5/6), rerun
from scratch on the same pinned snapshot so the RCDA-9 round is fully
self-contained and the probe numbers are replicated rather than inherited:

  behavioral probe (RCDA-5 port)
    A_l = residual states at generated positions INSIDE the loop (t >= collapse)
    B_l = residual states at generated positions BEFORE the loop (t < collapse)
    -> dh_l, z_diff_l (diagonal Mahalanobis), beta_l (Rayleigh spike strength
       vs the MP edge), writer selection by the 5-layer sliding-window rule

  phase probe (RCDA-6 port, early_cycles = 2)
    pools split pre / early / deep -> v_early_l, v_deep_l, cos(v_e, v_d)
    -> the phase-matched penalty directions of the pen arm

Outputs (rcda9/results/probe/):
  gens.jsonl                  frozen greedy generations (192 tok, bad set)
  probe_report.json           per-layer profile + writers (behavioral)
  probe_phase_report.json     per-layer phase profile + band check
  vectors_all.npz             v_l per layer (collapse direction)
  vectors_phase.npz           v_early_l / v_deep_l per layer
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rcda9_lab as L                                          # noqa: E402
from marchenko_pastur import (mp_bounds, mp_fit_sigma2,        # noqa: E402
                              rayleigh_spike_strength)


def diag_mahalanobis(dh: np.ndarray, A: np.ndarray, B: np.ndarray) -> float:
    var_a = A.var(axis=0) if len(A) > 1 else np.ones_like(dh)
    var_b = B.var(axis=0) if len(B) > 1 else np.ones_like(dh)
    pooled = 0.5 * (var_a + var_b) + 1e-8
    return float((dh ** 2 / pooled).sum())


def os_cpu() -> int:
    import os
    return os.cpu_count() or 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=L.LOCAL_MODEL)
    ap.add_argument("--max-new", type=int, default=L.MAX_NEW)
    ap.add_argument("--pool-threshold", type=float, default=0.3)
    ap.add_argument("--early-cycles", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else L.RESULTS_DIR / "probe"
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM

    torch.set_num_threads(max(1, os_cpu()))
    tok = L.load_tokenizer(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.eval()
    bad = L.load_bad_prompts()
    n_layers = model.config.num_hidden_layers
    d = model.config.hidden_size
    print(f"probe: {len(bad)} bad prompts | {n_layers} layers | d={d} | "
          f"RSS {L.rss_gb():.1f}GB", flush=True)

    # ---------------- stage A: frozen greedy generations (resume) ----------
    gens_path = out_dir / "gens.jsonl"
    done: dict[str, dict] = {}
    if gens_path.exists():
        for line in gens_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["id"]] = r
        print(f"resume: {len(done)} generations already on disk", flush=True)
    rows = []
    for pid, text in bad:
        if pid in done:
            r = done[pid]
        else:
            t0 = time.time()
            ids = L.gen_ids(model, tok, text, args.max_new)
            r = {"id": pid, "gen_ids": ids, "loop_share": L.loop_share(ids),
                 "collapse": L.loop_collapse_point(ids)}
            with gens_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"[gen] {pid}: ls={r['loop_share']:.2f} "
                  f"collapse={r['collapse']} ({time.time()-t0:.0f}s)", flush=True)
        rows.append(r)

    with (out_dir / "probe_generations.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            rec = {k: v for k, v in r.items() if k != "gen_ids"}
            rec["text_out"] = tok.decode(r["gen_ids"], skip_special_tokens=True)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    rho_strict = float(np.mean([1.0 if r["loop_share"] >= L.LOOP_THRESHOLD else 0.0
                                for r in rows]))
    pool = [r for r in rows if r["loop_share"] >= args.pool_threshold
            and r["collapse"] is not None]
    print(f"pool: {len(pool)}/{len(rows)} looping (>= {args.pool_threshold}); "
          f"rho_strict={rho_strict}", flush=True)

    # ------- stage B: one cacheless forward per pooled trajectory ----------
    A_all = {l: [] for l in range(n_layers)}     # inside-loop states
    B_all = {l: [] for l in range(n_layers)}     # pre-loop states
    P_pre = {l: [] for l in range(n_layers)}     # phase pools
    P_early = {l: [] for l in range(n_layers)}
    P_deep = {l: [] for l in range(n_layers)}
    phase_meta = []
    text_of = dict(bad)
    for r in pool:
        enc = L.encode_prompt(tok, text_of[r["id"]])
        plen = int(enc.shape[1])
        full = torch.cat([enc[0], torch.tensor(r["gen_ids"])], dim=0).unsqueeze(0)
        with torch.no_grad():
            out = model(full, output_hidden_states=True)
        hs = out.hidden_states
        c = r["collapse"]
        labels, meta = L.phase_labels(r["gen_ids"], c, args.early_cycles)
        meta["id"] = r["id"]
        phase_meta.append(meta)
        for l in range(n_layers):
            h = hs[l + 1][0].float().numpy()
            gen = h[plen:]
            lo = min(c, len(gen) - 2)
            A_all[l].append(gen[lo:])
            B_all[l].append(gen[:max(lo, 1)])
            P_pre[l].append(gen[labels == 0])
            P_early[l].append(gen[labels == 1])
            P_deep[l].append(gen[labels == 2])
        del out, hs

    # ---------------- behavioral profile + writers (RCDA-5 rule) -----------
    profile, dh_store = {}, {}
    for l in range(n_layers):
        A = np.concatenate(A_all[l], axis=0)
        B = np.concatenate(B_all[l], axis=0)
        dh = A.mean(axis=0) - B.mean(axis=0)
        z = diag_mahalanobis(dh, A, B)
        T = A.shape[0]
        C = np.cov(A.T) + 1e-6 * np.eye(A.shape[1])
        lam = np.linalg.eigvalsh(C)
        q = A.shape[1] / T
        sigma2 = mp_fit_sigma2(lam, q)
        lam_plus = mp_bounds(q, sigma2)[1]
        beta = rayleigh_spike_strength(float(lam[-1]), lam_plus)
        profile[l] = {"n_loop": int(A.shape[0]), "n_pre": int(B.shape[0]),
                      "z_diff": round(z, 3), "beta_rayleigh": round(beta, 3),
                      "dh_norm": round(float(np.linalg.norm(dh)), 5)}
        dh_store[l] = dh
    zd = np.array([profile[l]["z_diff"] for l in range(n_layers)])
    win = np.array([zd[l:l + 5].sum() for l in range(n_layers - 4)])
    w0 = int(np.argmax(win))
    writers = sorted(sorted(range(w0, min(w0 + 5, n_layers)),
                            key=lambda l: -zd[l])[:3])
    print(f"writers (behavioral): {writers} "
          f"z_diff={[profile[l]['z_diff'] for l in writers]}", flush=True)

    # ---------------- phase profile (RCDA-6 directions) --------------------
    phase_profile = {}
    ve_store, vd_store = {}, {}
    for l in range(n_layers):
        pre = np.concatenate(P_pre[l], axis=0) if P_pre[l] else np.zeros((1, d))
        early = np.concatenate(P_early[l], axis=0) if P_early[l] else np.zeros((1, d))
        deep = np.concatenate(P_deep[l], axis=0) if P_deep[l] else np.zeros((1, d))
        dh_e = early.mean(axis=0) - pre.mean(axis=0)
        dh_d = deep.mean(axis=0) - pre.mean(axis=0)
        ne, nd = np.linalg.norm(dh_e), np.linalg.norm(dh_d)
        v_e, v_d = dh_e / max(ne, 1e-9), dh_d / max(nd, 1e-9)
        z_e = diag_mahalanobis(dh_e, early, pre) if len(early) > 1 else 0.0
        z_d = diag_mahalanobis(dh_d, deep, pre) if len(deep) > 1 else 0.0
        phase_profile[l] = {
            "n_early": int(early.shape[0]), "n_deep": int(deep.shape[0]),
            "z_early": round(z_e, 3), "z_deep": round(z_d, 3),
            "cos_v_early_v_deep": round(float(abs(v_e @ v_d)), 4),
            "norm_early": round(float(ne), 4), "norm_deep": round(float(nd), 4),
        }
        ve_store[l], vd_store[l] = v_e, v_d

    # ---------------- band check (RCDA-6 hypothesis: L8-L15) ---------------
    band = [l for l in range(n_layers) if profile[l]["z_diff"] > 130]
    band = band if band else writers
    print(f"band (z_diff > 130): {band}", flush=True)

    report = {
        "model": L.MODEL_SNAPSHOT,
        "stage": "RCDA-9 formula round / stage 1 probe",
        "n_prompts": len(rows), "max_new": args.max_new,
        "pool_threshold": args.pool_threshold,
        "pool_ids": [r["id"] for r in pool],
        "rho_mini_strict": rho_strict,
        "loop_share_per_prompt": {r["id"]: round(r["loop_share"], 3) for r in rows},
        "collapse_per_prompt": {r["id"]: r["collapse"] for r in rows},
        "writers": writers,
        "writer_rule": "5-layer sliding window on z_diff, top-3 inside",
        "layer_profile": {str(l): profile[l] for l in range(n_layers)},
    }
    L.dump_json(report, out_dir / "probe_report.json")

    phase_report = {
        "model": L.MODEL_SNAPSHOT,
        "source_generations": str(gens_path),
        "pool_ids": [r["id"] for r in pool],
        "early_cycles": args.early_cycles,
        "phase_meta": phase_meta,
        "band": band,
        "band_check": {str(l): phase_profile[l] for l in band},
        "layer_profile": {str(l): phase_profile[l] for l in range(n_layers)},
    }
    L.dump_json(phase_report, out_dir / "probe_phase_report.json")

    np.savez(out_dir / "vectors_all.npz",
             **{f"v_L{l}": dh_store[l] / max(np.linalg.norm(dh_store[l]), 1e-9)
                for l in range(n_layers)})
    np.savez(out_dir / "vectors_phase.npz",
             **{**{f"v_early_L{l}": ve_store[l] for l in range(n_layers)},
                **{f"v_deep_L{l}": vd_store[l] for l in range(n_layers)}})
    print(f"saved probe_report.json, probe_phase_report.json, vectors | "
          f"RSS {L.rss_gb():.1f}GB", flush=True)


if __name__ == "__main__":
    main()
