#!/usr/bin/env python3
"""rcda5_analyze.py — spectral forensics of the SGD edit (Gemma3-1B-IT RCDA-5).

Recomputes the rich per-layer record from the trained state and the probe
vectors and saves:

  * rcds5_vectors.npz      — v_l per writer layer + per-layer top singular
                             values of the edit delta + summary stats
  * rcda5_spectral_digest.json — per writer layer: rel_frobenius, TW control
                             (verdict + z_resid / target shift in TW units),
                             cos(u1, v), top singular values of the delta,
                             columns changed > 1%

Usage: rcda5_analyze.py --model DIR --trained-state PATH --probe DIR --out DIR
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "Code"))
import gemma_lab as L                                    # noqa: E402
from marchenko_pastur import (mp_bounds, mp_fit_sigma2,  # noqa: E402
                              spectral_tw_control)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-1b-it")
    ap.add_argument("--trained-state", required=True)
    ap.add_argument("--probe", default=None)
    ap.add_argument("--out", required=True, help="results dir")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    probe_dir = Path(args.probe) if args.probe else L.resolve_work() / "probe"

    base_path = Path(args.model)
    base_sd = load_file(str(base_path / "model.safetensors"))
    trained = load_file(args.trained_state)

    layers = sorted({L.parse_layer(k) for k in trained if L.parse_layer(k) is not None})
    digest, vecs = {}, {}
    for l in layers:
        k = L.o_proj_key(l)
        v = np.load(probe_dir / f"v_L{l}.npy").astype(np.float64)
        wb = base_sd[k].float().numpy().astype(np.float64)
        wa = trained[k].float().numpy().astype(np.float64)
        d = wa - wb
        rel = float(np.linalg.norm(d) / max(np.linalg.norm(wb), 1e-12))
        tw = spectral_tw_control(wb, wa, v)
        u, s, _ = np.linalg.svd(d, full_matrices=False)
        u1 = u[:, 0]
        cos_u1_v = float(abs(np.dot(u1, v / np.linalg.norm(v))))

        # residual bulk (edit minus its v-component) vs the MP edge
        vh = v / np.linalg.norm(v)
        resid = d - np.outer(vh, vh @ d)
        transposed = resid.shape[0] > resid.shape[1]
        Rm = resid.T if transposed else resid
        mm, nn = Rm.shape
        lam = np.clip(np.linalg.eigvalsh((Rm @ Rm.T) / nn), 0, None)
        sigma2 = mp_fit_sigma2(lam, mm / nn)
        lam_plus = mp_bounds(mm / nn, sigma2)[1]

        digest[k] = {
            "layer": l,
            "rel_frobenius": round(rel, 5),
            "tw_verdict": tw["verdict"],
            "z_resid_tw": tw["z_resid_tw"],
            "target_shift_tw_units": tw["target_shift_tw_units"],
            "cos_u1_v": round(cos_u1_v, 4),
            "cos_top_left_sv_vs_v": round(cos_u1_v, 4),
            "cols_changed_gt_1pct": int((np.abs(d) > 0.01 *
                                         np.abs(wb).mean()).sum()),
            "n_weights": int(d.size),
            "sigma2_resid": float(sigma2),
            "lam_plus_resid": float(lam_plus),
            "top16_singular_values_delta":
                [round(float(x), 6) for x in s[:16]],
            "tw_detail": tw,
        }
        vecs[f"v_L{l}"] = v.astype(np.float32)
        vecs[f"delta_top16_s_L{l}"] = s[:16].astype(np.float32)
        vecs[f"u1_L{l}"] = u1.astype(np.float32)

    np.savez(out_dir / "rcds5_vectors.npz", **vecs)
    json.dump({"layers": layers, "digest": digest},
              (out_dir / "rcda5_spectral_digest.json").open("w"), indent=1)
    print(f"saved {out_dir}/rcds5_vectors.npz and rcda5_spectral_digest.json")
    print(json.dumps(digest, indent=1)[:2000])


if __name__ == "__main__":
    main()
