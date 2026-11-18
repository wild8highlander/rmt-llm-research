"""Channel decomposition of the loop direction (RCDA-6 diagnostic).

Question: at the writer layers, how much of the loop direction's energy is
written by the attention o_proj, by the MLP down_proj, and how much is simply
carried by the residual stream? RCDA-5 trains the o_proj channel only; if the
v-energy lives mostly in the MLP channel, the o_proj-only penalty *must*
under-treat the defect - which is precisely the RCDA-6 hypothesis of
RCDA5_REPORT.md ("extend the trainable tensors to down_proj").

Writes results/rcda6_channel_decomposition.json next to the probe report.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rcds_common import Arch, dump_json, greedy, load_model, load_prompts, seed_all


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai-community/gpt2")
    ap.add_argument("--probe", default=None)
    ap.add_argument("--vectors", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    seed_all(0)
    kit = Path(__file__).resolve().parent.parent
    base_results = kit / "R CDA - 5" / "Research" / "RCDA" / "results"
    probe = Path(args.probe) if args.probe else base_results / "probe_report.json"
    vectors = Path(args.vectors) if args.vectors else base_results / "rcds5_vectors.npz"
    out = Path(args.out) if args.out else base_results / "rcda6_channel_decomposition.json"

    report = json.loads(probe.read_text(encoding="utf-8"))
    writers = report["writer_layers"]
    npz = np.load(vectors)
    vs = {int(k[3:]): npz[k].astype(np.float32) for k in npz.files}

    model, tok = load_model(args.model)
    arch = Arch(model)
    prompts = load_prompts()

    per_layer = {l: {"attn": [], "mlp": [], "post": []} for l in writers}
    cap = {"attn": [], "mlp": []}
    for p in prompts["bad"]:
        cap["attn"], cap["mlp"] = [], []
        hook_handles = []
        for l in writers:
            hook_handles.append(arch.attn_out(l).register_forward_hook(
                lambda m, i, o, l=l: cap["attn"].append((l, o.detach()))))
            hook_handles.append(arch.mlp_out(l).register_forward_hook(
                lambda m, i, o, l=l: cap["mlp"].append((l, o.detach()))))
        ids = tok(p, return_tensors="pt").input_ids
        cont = greedy(model, tok, p, 48)
        full = torch.cat([ids[0], torch.tensor(cont)])
        out_h = model(full.unsqueeze(0), output_hidden_states=True)
        for h in hook_handles:
            h.remove()
        T = len(cont)
        for l in writers:
            v = torch.tensor(vs[l])
            v = v / v.norm()
            attn_out = next(o for ll, o in cap["attn"] if ll == l)[0, -T:, :].float()
            mlp_out = next(o for ll, o in cap["mlp"] if ll == l)[0, -T:, :].float()
            post = out_h.hidden_states[l + 1][0, -T:, :].float()
            per_layer[l]["attn"].append(float(((attn_out @ v) ** 2).mean()))
            per_layer[l]["mlp"].append(float(((mlp_out @ v) ** 2).mean()))
            per_layer[l]["post"].append(float(((post @ v) ** 2).mean()))

    result = {
        "model": args.model,
        "writers": writers,
        "note": "mean (v^T out)^2 over generated positions of the bad set; "
                "'post' = residual stream after the block (hidden_states)",
        "per_layer": {str(l): {k: round(float(np.mean(vv)), 4) for k, vv in d.items()}
                      for l, d in per_layer.items()},
    }
    tot_a = sum(result["per_layer"][str(l)]["attn"] for l in writers)
    tot_m = sum(result["per_layer"][str(l)]["mlp"] for l in writers)
    result["totals"] = {"attn": round(tot_a, 4), "mlp": round(tot_m, 4),
                        "mlp_share_of_writes": round(tot_m / (tot_a + tot_m), 4)}
    dump_json(result, out)
    print(json.dumps(result["per_layer"], indent=1))
    print("totals:", result["totals"])
    print(f"[decomp] wrote {out}")


if __name__ == "__main__":
    main()
