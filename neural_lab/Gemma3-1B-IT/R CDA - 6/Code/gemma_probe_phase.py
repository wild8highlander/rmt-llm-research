#!/usr/bin/env python3
"""gemma_probe_phase.py — RCDA-6 stage 1: the PHASE-conditioned loop probe.

The RCDA-5 report's closing hypothesis for Gemma3-1B-IT:

    "the loop is written jointly by attention output *and* MLP down_proj
     across a wider band (L8–L15 all show z_diff > 130), and the probe
     direction must be conditioned on the loop *phase*, not only the
     collapse point."

This script implements the second half of the hypothesis. It reuses the
FROZEN base generations of the RCDA-5 probe (probe/gens.jsonl — no new
sampling), splits every looping trajectory into three position phases,

    pre    : generated positions before the collapse point,
    early  : the first `early_cycles` cycles after the collapse
             (cycle period p fitted from the token tail),
    deep   : every loop position beyond that,

and extracts per decoder layer l (0..25):

    dh_early_l / dh_deep_l    phase mean differences of the residual stream
                              vs the pre-collapse states
    v_early_l / v_deep_l      the normalized phase directions
    z_early_l / z_deep_l      diagonal-Mahalanobis separability per phase
    cos(v_early, v_deep)      how much the two phases really differ

Outputs (under $GEMMA_WORK/probe_phase/):
    probe_phase_report.json  phase profile of every layer + band check
    vectors_all_phase.npz    v_early_l / v_deep_l for every layer
    v_early_L{N}.npy, v_deep_L{N}.npy   per band layer
    v_L{N}.npy               := v_deep (benchmark compatibility: the deep
                              phase dominates long loops; the collapse-only
                              direction of RCDA-5 ships as v_collapse_L{N}.npy)

Usage:
  gemma_probe_phase.py --model DIR --probe WORK/probe --band 8 9 10 11 12 13 14 15
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

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
    ap.add_argument("--probe", default=None,
                    help="RCDA-5 probe dir holding gens.jsonl (frozen generations)")
    ap.add_argument("--band", type=int, nargs="+",
                    default=[8, 9, 10, 11, 12, 13, 14, 15],
                    help="candidate writer band (RCDA-5 report hypothesis: L8-L15)")
    ap.add_argument("--pool-threshold", type=float, default=0.3)
    ap.add_argument("--early-cycles", type=int, default=2)
    ap.add_argument("--keep-irregular", action="store_true",
                    help="keep trajectories without a fitted cycle period in "
                         "the direction pools (default: exclude them — their "
                         "phase structure is undefined)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else L.resolve_work() / "probe_phase"
    out_dir.mkdir(parents=True, exist_ok=True)
    probe_dir = Path(args.probe) if args.probe else L.resolve_work() / "probe"

    gens_path = probe_dir / "gens.jsonl"
    rows = L.load_jsonl(gens_path)
    text_of = dict(L.load_bad_prompts())        # frozen sentinel texts
    pool = [r for r in rows
            if r["loop_share"] >= args.pool_threshold
            and r.get("collapse") is not None]
    print(f"phase probe: reusing {len(pool)}/{len(rows)} frozen looping "
          f"generations from {gens_path}", flush=True)

    band = sorted(args.band)
    n_layers = 26
    assert max(band) < n_layers, "band exceeds the 26-layer model"

    from transformers import AutoModelForCausalLM

    torch.set_num_threads(max(1, __import__("os").cpu_count() or 2))
    tok = L.load_tokenizer(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.eval()
    print(f"RSS after load: {L.rss_gb():.1f} GB", flush=True)

    # per-phase state pools, per layer
    pool_pre: dict[int, list] = {l: [] for l in range(n_layers)}
    pool_early: dict[int, list] = {l: [] for l in range(n_layers)}
    pool_deep: dict[int, list] = {l: [] for l in range(n_layers)}
    meta_rows = []

    t0 = time.time()
    for k, r in enumerate(pool):
        plen = int(L.encode_prompt(tok, text_of[r["id"]]).shape[1])
        full = torch.cat([L.encode_prompt(tok, text_of[r["id"]])[0],
                          torch.tensor(r["gen_ids"])], dim=0).unsqueeze(0)
        with torch.no_grad():
            out = model(full, output_hidden_states=True)
        hs = out.hidden_states                      # hs[l+1] = after layer l
        labels, meta = L.phase_labels(r["gen_ids"], r["collapse"],
                                      early_cycles=args.early_cycles)
        meta_rows.append({"id": r["id"], **meta,
                          "n_tok": len(r["gen_ids"]),
                          "n_pre": int((labels == 0).sum()),
                          "n_early": int((labels == 1).sum()),
                          "n_deep": int((labels == 2).sum()),
                          "excluded_irregular": meta["period"] is None
                          and not args.keep_irregular})
        if meta["period"] is None and not args.keep_irregular:
            print(f"[phase] {r['id']}: collapse={meta['collapse']} period=None "
                  f"-> EXCLUDED from direction pools (irregular tail)",
                  flush=True)
            del out, hs
            continue
        print(f"[phase] {r['id']}: collapse={meta['collapse']} "
              f"period={meta['period']} pre/early/deep="
              f"{(labels == 0).sum()}/{(labels == 1).sum()}/{(labels == 2).sum()}",
              flush=True)
        for l in range(n_layers):
            h = hs[l + 1][0].float().numpy()[plen:]
            pool_pre[l].append(h[labels == 0])
            pool_early[l].append(h[labels == 1])
            pool_deep[l].append(h[labels == 2])
        del out, hs
    print(f"phase forwards done in {time.time() - t0:.0f}s", flush=True)

    # ---- per-layer phase profile ----
    vecs: dict[str, np.ndarray] = {}
    profile: dict[str, dict] = {}
    for l in range(n_layers):
        B = np.concatenate([x for x in pool_pre[l] if len(x)], axis=0)
        A_e = np.concatenate([x for x in pool_early[l] if len(x)], axis=0)
        A_d = np.concatenate([x for x in pool_deep[l] if len(x)], axis=0)
        rec = {"n_pre": int(B.shape[0]), "n_early": int(A_e.shape[0]),
               "n_deep": int(A_d.shape[0])}
        dh_e = dh_d = None
        if len(A_e) and len(B):
            dh_e = A_e.mean(axis=0) - B.mean(axis=0)
            rec["z_early"] = round(diag_mahalanobis(dh_e, A_e, B), 3)
            rec["dh_early_norm"] = round(float(np.linalg.norm(dh_e)), 5)
            v_e = dh_e / max(np.linalg.norm(dh_e), 1e-9)
            vecs[f"v_early_L{l}"] = v_e.astype(np.float32)
        if len(A_d) and len(B):
            dh_d = A_d.mean(axis=0) - B.mean(axis=0)
            rec["z_deep"] = round(diag_mahalanobis(dh_d, A_d, B), 3)
            rec["dh_deep_norm"] = round(float(np.linalg.norm(dh_d)), 5)
            v_d = dh_d / max(np.linalg.norm(dh_d), 1e-9)
            vecs[f"v_deep_L{l}"] = v_d.astype(np.float32)
            vecs[f"v_L{l}"] = v_d.astype(np.float32)      # benchmark compat
        if dh_e is not None and dh_d is not None:
            c = float(abs(np.dot(v_e, v_d)) / max(np.linalg.norm(v_e)
                        * np.linalg.norm(v_d), 1e-9))
            rec["cos_v_early_v_deep"] = round(c, 4)
        profile[str(l)] = rec

    band_check = {}
    for l in band:
        rec = profile[str(l)]
        band_check[str(l)] = {
            "z_early": rec.get("z_early"), "z_deep": rec.get("z_deep"),
            "in_band_z_min": min(x for x in (rec.get("z_early"),
                                             rec.get("z_deep")) if x is not None)}
    report = {
        "model": args.model,
        "source_generations": str(gens_path),
        "pool_ids": [r["id"] for r in pool],
        "pool_threshold": args.pool_threshold,
        "early_cycles": args.early_cycles,
        "band": band,
        "band_check": band_check,
        "phase_meta": meta_rows,
        "layer_profile": profile,
    }
    json.dump(report, (out_dir / "probe_phase_report.json").open("w"), indent=1)
    np.savez(out_dir / "vectors_all_phase.npz", **vecs)
    for l in band:
        if f"v_early_L{l}" in vecs:
            np.save(out_dir / f"v_early_L{l}.npy", vecs[f"v_early_L{l}"])
            np.save(out_dir / f"v_deep_L{l}.npy", vecs[f"v_deep_L{l}"])
            np.save(out_dir / f"v_L{l}.npy", vecs[f"v_L{l}"])
    print(f"saved {out_dir}/probe_phase_report.json", flush=True)
    for l in band:
        rec = profile[str(l)]
        print(f"L{l:02d}: z_early={rec.get('z_early')} z_deep={rec.get('z_deep')} "
              f"cos(v_e,v_d)={rec.get('cos_v_early_v_deep')}", flush=True)


if __name__ == "__main__":
    main()
