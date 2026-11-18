"""The one evaluator every RCDA round ends in (MNN protocol, ported).

For one arm (base or a trained state) it measures, in one protocol:
  * rho_bad / rho_good / mean loop_share / loop_waste  - greedy 96, threshold 0.5
  * KL(base || arm) on the control corpus              - abilities drift
  * PPL per control text and mean                      - abilities parity
  * TW spectral purity of every edited tensor          - marchenko_pastur.tw_check
  * cos(u1 of the edit, v) per edited tensor           - did SGD align with the
                                                         behavioral direction?

Usage (round 5, trained arm):
  python gemma_benchmark.py --round 5 --tag trained --state ../"R CDA - 5"/.../rcda5_trained_state.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rcds_common import (Arch, dump_json, greedy, kl_and_ppl, load_model,
                         load_prompts, rho_of_gens, seed_all)
from marchenko_pastur import tw_check


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai-community/gpt2")
    ap.add_argument("--round", type=int, default=5, choices=[5, 6, 7])
    ap.add_argument("--tag", default="trained", choices=["base", "trained"])
    ap.add_argument("--state", default=None,
                    help="{variant}_trained_state.pt (omit for the base arm)")
    ap.add_argument("--vectors", default=None, help="rcds5_vectors.npz")
    ap.add_argument("--max_new", type=int, default=96)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--round_dir", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seed_all(args.seed)
    kit = Path(__file__).resolve().parent.parent
    round_dir = (Path(args.round_dir) if args.round_dir else
                 kit / f"R CDA - {args.round}")
    out = Path(args.out_dir) if args.out_dir else round_dir / "Research" / "RCDA" / "results"
    out.mkdir(parents=True, exist_ok=True)
    base_results = kit / "R CDA - 5" / "Research" / "RCDA" / "results"

    state = torch.load(args.state, map_location="cpu") if args.state else None
    if state is None and args.tag == "trained":
        raise SystemExit("--tag trained requires --state")

    model, tok = load_model(args.model)
    arch = Arch(model)
    base_model, _ = load_model(args.model)
    prompts = load_prompts()

    edited = {}
    if state:
        sd = model.state_dict()
        base_sd = base_model.state_dict()
        with torch.no_grad():
            for name, w in state.items():
                sd[name].copy_(w.float())
            # rel_frobenius against the pristine base copy
            for name, w in state.items():
                d = (w.float() - base_sd[name].float()).numpy()
                rel = float(np.linalg.norm(d) /
                            (np.linalg.norm(base_sd[name].numpy().astype(np.float32)) + 1e-12))
                edited[name] = {"rel_frobenius": round(rel, 5),
                                "n_weights": int(d.size)}

    print(f"[bench] round {args.round} tag={args.tag} "
          f"edited={len(edited)} tensors, generating ({args.max_new} tokens, greedy)")

    gens_bad = [greedy(model, tok, p, args.max_new) for p in prompts["bad"]]
    gens_good = [greedy(model, tok, p, args.max_new) for p in prompts["good"]]
    rb = rho_of_gens(gens_bad)
    rg = rho_of_gens(gens_good)
    kl = kl_and_ppl(model, base_model, tok, prompts["control"])

    tw = {}
    if edited:
        sd = model.state_dict()
        for name in edited:
            tw[name] = tw_check(sd[name].numpy().astype(np.float64))

    cos_v = {}
    if edited:
        vec_path = Path(args.vectors) if args.vectors else base_results / "rcds5_vectors.npz"
        vs = {int(k[3:]): v for k, v in np.load(vec_path).items()}
        for name in edited:
            l_from_name = None
            for tok_ in name.split("."):
                if tok_.isdigit():
                    l_from_name = int(tok_)
                    break
            v = vs.get(l_from_name)
            if v is None:
                continue
            d = (model.state_dict()[name].float() -
                 base_model.state_dict()[name].float()).numpy()
            # u1 must live in the same space as v (the residual/output side):
            # pick the singular-vector side whose dimension matches len(v)
            U, S, Vh = np.linalg.svd(d, full_matrices=False)
            if d.shape[0] == len(v):
                u1 = U[:, 0]
            elif d.shape[1] == len(v):
                u1 = Vh[0]
            else:
                continue
            cos_v[name] = round(float(abs(u1 @ v)), 4)

    result = {
        "tag": args.tag,
        "model": args.model,
        "round": args.round,
        "protocol": {"n_bad": len(prompts["bad"]), "n_good": len(prompts["good"]),
                     "max_new": args.max_new, "ngram": 4, "rho_threshold": 0.5,
                     "decoding": "greedy", "precision": "fp32"},
        "n_changed_tensors": len(edited),
        "changed": edited,
        "tw": {n: {"verdict": t["verdict"], "ks_distance": round(t["ks_distance"], 4)}
               for n, t in tw.items()},
        "cosine_edit_vs_v": cos_v,
        "rho_bad": rb["rho"],
        "rho_good": rg["rho"],
        "mean_loop_share": round(rb["mean_loop_share"], 4),
        "loop_waste_mean": round(rb["loop_waste_mean"], 1),
        "per_traj_loop_share_bad": [round(x, 3) for x in rb["per_traj_loop_share"]],
        "kl_ctrl": round(kl["kl_ctrl_mean"], 4),
        "ppl_control_mean": round(kl["ppl_control_mean"], 1),
        "ppl_per_text": [round(x, 1) for x in kl["ppl_per_text"]],
    }
    suffix = args.tag
    fname = f"part{args.round}_eval_{suffix}.json"
    dump_json(result, out / fname)
    print(f"[bench] rho_bad={result['rho_bad']:.2f} (base protocol) "
          f"rho_good={result['rho_good']:.2f} KL={result['kl_ctrl']:.3f} "
          f"PPL={result['ppl_control_mean']:.1f} TW={'pass' if all(t['verdict']=='pass' for t in tw.values()) or not tw else 'check'}")

    # merge PPL into the round's ppl_check.json (base + trained arms)
    ppl_file = out / "ppl_check.json"
    ppl = json.loads(ppl_file.read_text(encoding="utf-8")) if ppl_file.exists() else {}
    ppl[args.tag] = {"ppl_control_mean": result["ppl_control_mean"],
                     "ppl_per_text": result["ppl_per_text"]}
    dump_json(ppl, ppl_file)
    print(f"[bench] wrote {fname} and updated ppl_check.json")


if __name__ == "__main__":
    main()
