"""Diagnostic: pen on training prefixes vs pen on the model's own fresh
trajectories - separates real de-looping from prefix overfitting."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rcds_common import (Arch, dump_json, greedy, load_model, load_prompts,
                         seed_all)
from gemma_rcds_train import PenCapture, pen_value


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai-community/gpt2")
    ap.add_argument("--state", default="../R CDA - 5/Research/RCDA/results/rcda5_trained_state.pt")
    ap.add_argument("--vectors", default="../R CDA - 5/Research/RCDA/results/rcds5_vectors.npz")
    args = ap.parse_args()

    seed_all(0)
    base_model, tok = load_model(args.model)
    model, _ = load_model(args.model)
    arch = Arch(model)
    sd = model.state_dict()
    state = torch.load(Path(__file__).parent / args.state, map_location="cpu")
    with torch.no_grad():
        for name, w in state.items():
            sd[name].copy_(w.float())

    report = json.loads((Path(__file__).parent.parent / "R CDA - 5" / "Research" /
                         "RCDA" / "results" / "probe_report.json").read_text())
    writers = report["writer_layers"]
    npz = np.load(Path(__file__).parent / args.vectors)
    vs = {int(k[3:]): npz[k].astype(np.float32) for k in npz.files}
    points = [("post", l) for l in writers]

    prompts = load_prompts()["bad"][:8]
    cap_b = PenCapture(arch, points)
    cap_f = PenCapture(base_model and Arch(base_model), points) if False else None

    # base model pen on fresh trajectories
    cap0 = PenCapture(Arch(base_model), points)
    pen_base = []
    for p in prompts:
        cont = greedy(base_model, tok, p, 48)
        ids = tok(p, return_tensors="pt").input_ids
        full = torch.cat([ids[0], torch.tensor(cont)]).unsqueeze(0)
        cap0.outputs.clear()
        base_model(full)
        pen_base.append(float(pen_value(cap0.outputs, vs, points, slice(-48, None))))

    # trained model: pen on its OWN fresh trajectories
    pen_fresh, pen_prefix = [], []
    for p in prompts:
        cont = greedy(model, tok, p, 48)
        ids = tok(p, return_tensors="pt").input_ids
        full = torch.cat([ids[0], torch.tensor(cont)]).unsqueeze(0)
        cap_b.outputs.clear()
        model(full)
        pen_fresh.append(float(pen_value(cap_b.outputs, vs, points, slice(-48, None))))
        # and on the BASE's frozen prefix (the training sequence)
        cont0 = greedy(base_model, tok, p, 48)
        full0 = torch.cat([ids[0], torch.tensor(cont0)]).unsqueeze(0)
        cap_b.outputs.clear()
        model(full0)
        pen_prefix.append(float(pen_value(cap_b.outputs, vs, points, slice(-48, None))))

    res = {
        "pen_base_fresh": round(float(np.mean(pen_base)), 2),
        "pen_trained_on_training_prefixes": round(float(np.mean(pen_prefix)), 2),
        "pen_trained_on_fresh_trajectories": round(float(np.mean(pen_fresh)), 2),
        "ratio_prefix": round(float(np.mean(pen_prefix) / np.mean(pen_base)), 3),
        "ratio_fresh": round(float(np.mean(pen_fresh) / np.mean(pen_base)), 3),
        "per_prompt_fresh": [round(x, 1) for x in pen_fresh],
    }
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
