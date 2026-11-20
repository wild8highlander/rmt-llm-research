"""RCDA-1-style spectral probe for Gemma3-1B-IT (and any causal LM).

The model labels its own defect: hidden-state deltas between looped and clean
generated tokens give a behavioural direction v per layer; a Hotelling-style
z_diff localises the writers; the matrix scans (marchenko_pastur.spike_scan)
decide which weight matrices are supercritical for the loop direction.

Outputs (round-5 canonical locations):
  results/probe_report.json      - per-layer z_diff, scans, writer selection
  results/rcds5_vectors.npz      - v_L{l} unit vectors (input to training)

Production run:  python gemma_probe.py --model google/gemma-3-1b-it
Validation run:  python gemma_probe.py                (GPT-2 stand-in)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rcds_common import (Arch, dump_json, greedy, load_model, load_prompts,
                         loop_stats, seed_all)
from marchenko_pastur import spike_scan, tw_check


@torch.no_grad()
def collect_hidden(model, tok, arch, prompts, max_new=48):
    """Greedy continuations + per-layer hidden states on generated positions.

    Generation first, then one full-sequence forward pass with
    output_hidden_states=True - exact positions, no cache-shape surprises.
    """
    gens, hiddens = [], []
    for text in prompts:
        new = greedy(model, tok, text, max_new)
        gens.append(new)
        ids = tok(text, return_tensors="pt").input_ids
        full = torch.cat([ids[0], torch.tensor(new)])
        out = model(full.unsqueeze(0), output_hidden_states=True)
        per_layer = [out.hidden_states[l][0, ids.shape[1]:].cpu().numpy()
                     for l in range(arch.n_layers + 1)]  # layer -> [new, d]
        hiddens.append(per_layer)
    return gens, hiddens


def probe_directions(gens_bad, hid_bad, gens_good, hid_good, n_layers):
    """v_l from Delta h between loop tokens and clean tokens *labelled by the
    model itself*: the contrast is taken WITHIN the same generations wherever
    possible (looped positions vs unlooped positions of the bad continuations,
    plus whole clean continuations as the control pool) - this strips the
    content difference and keeps the defect. z_diff = separation of the
    projections onto v in pooled-sigma units."""
    loop_h = [[] for _ in range(n_layers + 1)]
    clean_h = [[] for _ in range(n_layers + 1)]
    for g, hs in zip(gens_bad, hid_bad):
        # loop tokens: tokens belonging to repeated 4-grams
        counts = {}
        ngram = 4
        n = len(g)
        for i in range(n - ngram + 1):
            counts.setdefault(tuple(g[i:i + ngram]), 0)
            counts[tuple(g[i:i + ngram])] += 1
        mask = []
        for i in range(n):
            in_loop = (i + ngram <= n and counts[tuple(g[i:i + ngram])] >= 2) or \
                      (i >= ngram and counts[tuple(g[i - ngram:i])] >= 2)
            mask.append(in_loop)
        n_loop = sum(mask)
        if n_loop == 0:
            continue
        idx_loop = np.array(mask)
        for l in range(n_layers + 1):
            loop_h[l].append(hs[l][idx_loop])
            # the within-generation control: unlooped positions of the same text
            n_clean_same = len(g) - n_loop
            take = min(n_clean_same, n_loop)
            if take > 0:
                clean_idx = np.where(~idx_loop)[0][:take]
                clean_h[l].append(hs[l][clean_idx])
    for g, hs in zip(gens_good, hid_good):
        # good generations: all tokens are clean when the gen itself is clean
        if loop_stats(g)["loop_share"] < 0.3:
            for l in range(n_layers + 1):
                clean_h[l].append(hs[l])
    vs, zs = {}, {}
    for l in range(1, n_layers + 1):
        if not loop_h[l]:
            continue
        A = np.concatenate(loop_h[l], 0)
        if clean_h[l]:
            B = np.concatenate(clean_h[l], 0)
        else:
            B = np.concatenate([hs[l] for hs in hid_good], 0)
        mu_d = A.mean(0) - B.mean(0)
        nrm = np.linalg.norm(mu_d)
        if nrm < 1e-8:
            continue
        v = mu_d / nrm
        pa = A @ v
        pb = B @ v
        pooled = np.std(np.concatenate([pa, pb])) + 1e-9
        z = (pa.mean() - pb.mean()) / pooled
        # hidden_states index l is the OUTPUT of transformer block l-1:
        # store vectors on block indices so hooks/weights line up 1:1
        vs[l - 1] = v
        zs[l - 1] = float(z)
    return vs, zs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai-community/gpt2",
                    help="google/gemma-3-1b-it in production")
    ap.add_argument("--out", default=None, help="results dir (default: RCDA-5/Research/RCDA/results)")
    ap.add_argument("--n_writers", type=int, default=3)
    ap.add_argument("--max_new", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seed_all(args.seed)
    kit = Path(__file__).resolve().parent.parent
    out = Path(args.out) if args.out else kit / "RCDA-5" / "Research" / "RCDA" / "results"
    out.mkdir(parents=True, exist_ok=True)

    model, tok = load_model(args.model)
    arch = Arch(model)
    prompts = load_prompts()

    print(f"[probe] model={args.model} layers={arch.n_layers} d={arch.d_model}")
    gens_bad, hid_bad = collect_hidden(model, tok, arch, prompts["bad"], args.max_new)
    gens_good, hid_good = collect_hidden(model, tok, arch, prompts["good"], args.max_new)
    base_rho = sum(1 for g in gens_bad if loop_stats(g)["loop_share"] >= 0.5) / len(gens_bad)
    print(f"[probe] base rho_bad = {base_rho:.2f} on {len(gens_bad)} bad prompts")

    vs, zs = probe_directions(gens_bad, hid_bad, gens_good, hid_good, arch.n_layers)

    scans = {}
    for l in sorted(vs):
        W = arch.attn_out(l).weight.detach().double().numpy()
        sc = spike_scan(W, vs[l])
        tw = tw_check(W)
        scans[l] = {"attn": sc, "tw_base": tw["verdict"],
                    "lambda_ratio": sc["lambda_ratio"], "theta_v": sc["theta_v"]}

    # writer selection: supercritical first, then by z_diff
    order = sorted(
        [l for l in sorted(vs)],
        key=lambda l: ((-1 if scans.get(l, {}).get("attn", {}).get("supercritical") else 0),
                       -zs[l]),
    )
    writers = order[: args.n_writers]

    report = {
        "model": args.model,
        "arch": {"kind": arch.kind, "n_layers": arch.n_layers, "d_model": arch.d_model},
        "protocol": {"bad": len(prompts["bad"]), "good": len(prompts["good"]),
                     "max_new": args.max_new, "ngram": 4, "rho_threshold": 0.5},
        "base_rho_bad": base_rho,
        "layers": {str(l): {"z_diff": zs[l], "scan": scans[l]["attn"],
                            "tw_base": scans[l]["tw_base"]} for l in sorted(vs)},
        "writer_layers": writers,
        "writer_names": {"attn": [arch.name_attn(l) for l in writers],
                         "mlp": [arch.name_mlp(l) for l in writers]},
        "deep_phase_layer": max(writers) if writers else None,
    }
    dump_json(report, out / "probe_report.json")
    np.savez_compressed(out / "rcds5_vectors.npz",
                        **{f"v_L{l}": vs[l] for l in sorted(vs)})
    print(f"[probe] writers = {writers}  (deep phase = L{max(writers)})")
    for l in writers:
        sc = scans[l]["attn"]
        print(f"  L{l}: z_diff={zs[l]:.2f}  lam_v/lam+={sc['lambda_ratio']:.2f}  "
              f"theta_v={sc['theta_v']:.2f} > theta_c={sc['theta_c']:.2f}  "
              f"supercritical={sc['supercritical']}")
    print(f"[probe] wrote {out/'probe_report.json'} and rcds5_vectors.npz")


if __name__ == "__main__":
    main()
