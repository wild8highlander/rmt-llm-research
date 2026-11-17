#!/usr/bin/env python3
"""rcda6_analyze.py — spectral forensics digest of the RCDA-6 edit.

The per-tensor forensics (rel_Frobenius, TW control, cos(u1, v_early/v_deep),
columns changed > 1 %) are computed by gemma_rcds6_train.py's final pass with
incremental persistence in analysis_partial.json. This tool converts that
record into the shipping artifacts:

  * rcda6_spectral_digest.json — per edited tensor summary
  * rcds6_vectors.npz          — v_early/v_deep per band layer + u1 + top-16
                                 singular values of each edit delta

Usage: rcda6_analyze.py --work WORK/rcda6 --probe-phase WORK/probe_phase --out DIR
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[3] / "Code"
sys.path.insert(0, str(CODE))
import gemma_lab as L                                    # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, help="rcda6 runtime dir")
    ap.add_argument("--probe-phase", default=None)
    ap.add_argument("--out", required=True, help="results dir")
    args = ap.parse_args()

    work = Path(args.work)
    pp_dir = Path(args.probe_phase) if args.probe_phase \
        else L.resolve_work() / "probe_phase"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    partial = json.load((work / "analysis_partial.json").open())
    band = sorted({rec["layer"] for rec in partial.values()})
    types = sorted({rec["type"] for rec in partial.values()})
    print(f"digest: {len(partial)} tensors, band {band}, types {types}")

    digest = {}
    for k in sorted(partial):
        rec = dict(partial[k])
        rec["tw_verdict"] = rec.pop("tw", None)
        tw_detail = rec.pop("tw_detail", {})
        rec["z_resid_tw"] = tw_detail.get("z_resid_tw")
        rec["target_shift_tw_units"] = tw_detail.get("target_shift_tw_units")
        rec["target_index"] = tw_detail.get("target_index")
        rec["target_cos"] = tw_detail.get("target_cos")
        digest[k] = rec

    vecs: dict[str, np.ndarray] = {}
    for l in band:                                   # phase directions
        for ph in ("early", "deep"):
            f = pp_dir / f"v_{ph}_L{l}.npy"
            if f.exists():
                vecs[f"v_{ph}_L{l}"] = np.load(f).astype(np.float32)

    # u1 + top-16 singular values of each edit delta (from disk, per key)
    from safetensors import safe_open
    from safetensors.torch import load_file
    trained = load_file(str(work / "trained_state.safetensors"))
    model_hint = json.load((work / "training_log.json").open())["args"]["model"]
    base_st = Path(model_hint) / "model.safetensors"
    if not base_st.exists():
        from huggingface_hub import hf_hub_download
        base_st = Path(hf_hub_download(model_hint, "model.safetensors"))
    with safe_open(str(base_st), framework="pt") as f:
        for k in sorted(partial):
            d = (trained[k].float() - f.get_tensor(k).float()).numpy()
            u, s, _ = np.linalg.svd(d, full_matrices=False)
            vecs[f"delta_top16_s_{k}"] = s[:16].astype(np.float32)
            vecs[f"u1_{k}"] = u[:, 0].astype(np.float32)
            del d, u, s
    del trained

    np.savez(out_dir / "rcds6_vectors.npz", **vecs)
    json.dump({"band": band, "tensor_types": types, "digest": digest},
              (out_dir / "rcda6_spectral_digest.json").open("w"), indent=1)
    print(f"saved {out_dir}/rcds6_vectors.npz and rcda6_spectral_digest.json")

    for k in sorted(digest):
        r = digest[k]
        print(f"{k}: rel_F={r['rel_frobenius']:.5f} tw={r['tw_verdict']} "
              f"z_resid={r['z_resid_tw']} cos_u1_v_deep={r['cos_u1_v_deep']} "
              f"cos_u1_v_early={r['cos_u1_v_early']}")


if __name__ == "__main__":
    main()
