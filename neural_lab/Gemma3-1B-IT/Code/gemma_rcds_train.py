"""RCDA-5/6/7: bake the RMT formula into gradient training.

The penalty is `(I - P_v)` expressed as a gradient signal - literally the
repository's formula: `pen = (v^T h / ||v||)^2` on the outputs of the writer
tensors, normalised per prompt by its init value (`pen_norm`), plus a HARD
abilities anchor `beta * relu(mse - cap)` on the trailing 32 logit positions
against the frozen base logits (MNN RCDA-5 protocol: Adam, lr 3e-5, 150 steps,
batch 1, mini-benchmark rho at steps 0/75/150).

Variants (one script, one evaluator - the only levers that move are the ones
named in the round hypotheses):
  rcda5  baseline: trainable = o_proj of writer layers W; pen at the o_proj
         outputs of W; anchor beta=10, cap=1.0
  rcda6  RCDA5_REPORT.md hypothesis: extend the trainable tensors to down_proj
         (receiver channel) - trainable and pen points double
  rcda7  next lever: deep-phase-only penalty (pen enforced only at the deepest
         writer layer, where the loop is read out) + tightened anchor
         (beta=20, cap=0.5 - half the tolerated drift)

Artifacts per run (canonical locations):
  results/{variant}_training_log.json   steps + pen0 + bench + args
  results/{variant}_trained_state.pt    full edited tensors (fp32, for eval)
  results/{variant}_deltas.npz          fp16 deltas (Model-ZIP content)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rcds_common import (Arch, dump_json, greedy, load_model, load_prompts,
                         rho_of_gens, seed_all)


VARIANTS = {
    "rcda5": {"train_mlp": False, "deep_only": False, "beta": 10.0, "cap": 1.0},
    "rcda6": {"train_mlp": True, "deep_only": False, "beta": 10.0, "cap": 1.0},
    "rcda7": {"train_mlp": True, "deep_only": True, "beta": 20.0, "cap": 0.5},
}


class PenCapture:
    """Forward hooks that grab the post-residual block outputs - the hidden
    states the formula in the MNN RCDA-5 report is written against:
    "fine-tuned so that its hidden states on glitched trajectories stop
    writing along the loop direction v". The channel decomposition
    (rcda6_channel_decomposition.py) shows the defect lives in the carried
    state (~500-700 units of v-energy) while single writes contribute 2-7 -
    so the penalty reads the state, and the trainable tensors do the writing.
    """

    def __init__(self, arch: Arch, points: list[tuple[str, int]]):
        self.outputs: dict[tuple[str, int], torch.Tensor] = {}
        self.handles = []
        for kind, l in points:
            mod = arch.layers[l]
            self.handles.append(
                mod.register_forward_hook(self._make(kind, l)))

    def _make(self, kind, l):
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            self.outputs[(kind, l)] = h
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


def pen_value(cap_out: dict, vs: dict, points: list[tuple[str, int]],
              positions: slice) -> torch.Tensor:
    """pen = mean over points of (v^T h / ||v||)^2 on the penalty positions."""
    terms = []
    for key in points:
        h = cap_out[key]  # [B, T, d]
        v = torch.tensor(vs[key[1]], dtype=h.dtype)
        v = v / (v.norm() + 1e-12)
        proj = h[:, positions, :] @ v  # [B, T_pen]
        terms.append((proj ** 2).mean())
    return torch.stack(terms).mean()


@torch.no_grad()
def mini_bench(model, tok, bad_prompts, n=8, max_new=64):
    gens = [greedy(model, tok, p, max_new) for p in bad_prompts[:n]]
    r = rho_of_gens(gens)
    return {"rho_mini": r["rho"], "mean_loop_share": r["mean_loop_share"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai-community/gpt2",
                    help="google/gemma-3-1b-it in production")
    ap.add_argument("--variant", choices=list(VARIANTS), required=True)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--clip", type=float, default=1.0,
                    help="gradient-norm clip (0 disables); stabilises the "
                         "pen-vs-anchor bang-bang on smaller models")
    ap.add_argument("--beta", type=float, default=None,
                    help="override the variant anchor weight")
    ap.add_argument("--cap", type=float, default=None,
                    help="override the variant anchor cap")
    ap.add_argument("--bench-at", type=int, nargs="*", default=[75, 150])
    ap.add_argument("--vectors", default=None)
    ap.add_argument("--probe", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--round_dir", default=None,
                    help="R CDA - N folder owning this run")
    ap.add_argument("--prefix_len", type=int, default=48)
    ap.add_argument("--anchor_len", type=int, default=32)
    ap.add_argument("--anchor_scope", choices=["all", "good"], default="all",
                    help="'all' = hard hinge on every sequence; 'good' = "
                         "anchor only on the ability data (the bad prefixes "
                         "are freed for the penalty - on them the base logits "
                         "encode the loop itself)")
    ap.add_argument("--refresh_every", type=int, default=0,
                    help="regenerate the bad prefixes from the CURRENT model "
                         "every N steps (online self-labelling; 0 = frozen "
                         "base prefixes as in the MNN protocol)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    seed_all(args.seed)
    cfg = dict(VARIANTS[args.variant])
    if args.beta is not None:
        cfg["beta"] = args.beta
    if args.cap is not None:
        cfg["cap"] = args.cap

    kit = Path(__file__).resolve().parent.parent
    round_dir = (Path(args.round_dir) if args.round_dir else
                 kit / {"rcda5": "RCDA-5", "rcda6": "RCDA-6",
                        "rcda7": "RCDA-7"}[args.variant])
    out = Path(args.out_dir) if args.out_dir else round_dir / "Research" / "RCDA" / "results"
    out.mkdir(parents=True, exist_ok=True)
    base_results = kit / "RCDA-5" / "Research" / "RCDA" / "results"

    probe = Path(args.probe) if args.probe else base_results / "probe_report.json"
    vectors = Path(args.vectors) if args.vectors else base_results / "rcds5_vectors.npz"
    report = __import__("json").loads(probe.read_text(encoding="utf-8"))
    vs_npz = np.load(vectors)
    vs = {int(k[3:]): vs_npz[k].astype(np.float32) for k in vs_npz.files}
    writers = report["writer_layers"]
    deep = report["deep_phase_layer"]
    arch0 = report["arch"]
    is_stand_in = "gemma" not in args.model.lower()

    print(f"[train] variant={args.variant} model={args.model} "
          f"writers={writers} deep=L{deep} beta={cfg['beta']} cap={cfg['cap']}")

    # ------------------------------------------------------------------ setup
    model, tok = load_model(args.model)
    arch = Arch(model)
    base_model, base_tok = load_model(args.model)
    for p in base_model.parameters():
        p.requires_grad_(False)

    if cfg["deep_only"]:
        pen_layers = [deep]
    else:
        pen_layers = list(writers)
    points = [("post", l) for l in pen_layers]

    trainable_names, trainable_params = [], []
    for l in writers:
        m = arch.attn_out(l).weight
        m.requires_grad_(True)
        trainable_params.append(m)
        trainable_names.append(arch.name_attn(l))
        if cfg["train_mlp"]:
            m2 = arch.mlp_out(l).weight
            m2.requires_grad_(True)
            trainable_params.append(m2)
            trainable_names.append(arch.name_mlp(l))
    n_train = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[train] trainable: {len(trainable_params)} tensors, "
          f"{n_train:,} params ({100 * n_train / n_total:.2f}%)")
    print(f"[train] pen points: {points}")

    # -------------------------------------------------- prepare the sequences
    prompts = load_prompts()
    t0 = time.time()
    bad_prefix, good_prefix = [], []
    for p in prompts["bad"]:
        ids = tok(p, return_tensors="pt").input_ids
        cont = greedy(model, tok, p, args.prefix_len)
        bad_prefix.append(torch.cat([ids[0], torch.tensor(cont)]))
    for p in prompts["good"]:
        ids = tok(p, return_tensors="pt").input_ids
        cont = greedy(model, tok, p, args.prefix_len)
        good_prefix.append(torch.cat([ids[0], torch.tensor(cont)]))
    print(f"[train] prefixes ready in {time.time() - t0:.0f}s")

    pen_pos_len = args.prefix_len
    pen_slice = slice(-pen_pos_len, None)
    anchor_slice = slice(-args.anchor_len, None)

    cap = PenCapture(arch, points)

    def forward_pen(seq_ids):
        cap.outputs.clear()
        logits = model(seq_ids.unsqueeze(0)).logits
        return logits, pen_value(cap.outputs, vs, points, pen_slice)

    def forward_anchor_only(seq_ids):
        cap.outputs.clear()
        logits = model(seq_ids.unsqueeze(0)).logits
        return logits, None

    # pen0 per bad prompt at init (weights == base here)
    pen0 = {}
    for i, seq in enumerate(bad_prefix):
        _, pen = forward_pen(seq)
        pen0[str(i)] = float(pen.detach())
    pen_ref = float(np.mean(list(pen0.values())))
    print(f"[train] pen0 spread: {min(pen0.values()):.1f} .. {max(pen0.values()):.1f}, "
          f"pen_ref={pen_ref:.1f}")

    # frozen base trailing logits (the hard anchor reference)
    def refresh_base_logits(seqs, keys):
        with torch.no_grad():
            for k, seq in zip(keys, seqs):
                base_logits[k] = base_model(
                    seq.unsqueeze(0)).logits[0, anchor_slice, :].float()

    base_logits = {}
    refresh_base_logits(bad_prefix + good_prefix,
                        list(range(len(bad_prefix) + len(good_prefix))))
    if args.dry_run:
        print("[train] dry run ok - exiting before the training loop")
        return

    # ------------------------------------------------------------ training
    opt = torch.optim.Adam(trainable_params, lr=args.lr)
    steps_log, bench_log = [], []
    order = [("bad", i) for i in range(len(bad_prefix))]
    order += [("good", i) for i in range(len(good_prefix))]

    mb0 = mini_bench(model, tok, prompts["bad"])
    bench_log.append({"step": 0, **{k: round(v, 3) for k, v in mb0.items()}})
    print(f"[train] bench@0: rho_mini={mb0['rho_mini']:.2f} "
          f"loop_share={mb0['mean_loop_share']:.3f}")

    for step in range(1, args.steps + 1):
        kind, idx = order[(step - 1) % len(order)]
        seq = (bad_prefix if kind == "bad" else good_prefix)[idx]
        ts = time.time()
        # the penalty enters the loss as a LIVE tensor (graph intact),
        # normalised by the prompt's init value - pen_norm below is the
        # logged, detached copy of the same quantity
        if kind == "bad":
            logits, pen = forward_pen(seq)
            pen_norm = float(pen.detach()) / (pen0[str(idx)] + 1e-12)
            pen_raw = float(pen.detach())
            loss = pen / (pen0[str(idx)] + 1e-12)
        else:
            logits, _ = forward_anchor_only(seq)
            pen_norm, pen_raw = 0.0, 0.0
            loss = torch.zeros(())
        # anchor: the hard abilities constraint. scope 'good' applies the
        # hinge on the ability data only; on the bad prefixes the base logits
        # encode the loop, so pinning them would defend the defect
        if kind == "bad" and args.anchor_scope == "good":
            mse = torch.zeros(())
        else:
            mse = torch.nn.functional.mse_loss(
                logits[0, anchor_slice, :], base_logits[(idx if kind == "bad"
                                                         else len(bad_prefix) + idx)].float())
        loss = loss + cfg["beta"] * torch.relu(mse - cfg["cap"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, args.clip)
        opt.step()
        steps_log.append({"step": step, "kind": kind, "mse": round(float(mse), 6),
                          "pen_raw": round(pen_raw, 4),
                          "pen_norm": round(pen_norm, 4),
                          "loss": round(float(loss), 4),
                          "s": round(time.time() - ts, 2)})
        if step in args.bench_at:
            mb = mini_bench(model, tok, prompts["bad"])
            bench_log.append({"step": step, **{k: round(v, 3) for k, v in mb.items()}})
            print(f"[train] bench@{step}: rho_mini={mb['rho_mini']:.2f} "
                  f"loop_share={mb['mean_loop_share']:.3f}")
        # online self-labelling: re-draw the bad prefixes from the current
        # model so the penalty always attacks the trajectory the model would
        # actually generate (the defect is re-labelled, not memorised)
        if args.refresh_every and step % args.refresh_every == 0 and step < args.steps:
            with torch.no_grad():
                for i, p in enumerate(prompts["bad"]):
                    ids = tok(p, return_tensors="pt").input_ids
                    cont = greedy(model, tok, p, args.prefix_len)
                    bad_prefix[i] = torch.cat([ids[0], torch.tensor(cont)])
            refresh_base_logits(bad_prefix, range(len(bad_prefix)))
            print(f"[train] step {step}: bad prefixes refreshed from the current model")
        if step % 10 == 0:
            print(f"[train] step {step}: loss={float(loss):.3f} pen_norm={pen_norm:.2f} "
                  f"mse={float(mse):.4f}")

    # ------------------------------------------------------------- artifacts
    deltas, state = {}, {}
    base_sd = dict(base_model.state_dict())
    with torch.no_grad():
        for l in writers:
            for name, mod in ((arch.name_attn(l), arch.attn_out(l)),
                              (arch.name_mlp(l), arch.mlp_out(l))):
                if name not in trainable_names:
                    continue
                cur = mod.weight.detach()
                base_w = base_sd[name].float()
                d = (cur - base_w)
                deltas[name] = d.half().numpy()
                state[name] = cur.clone()
    rel_frob = {n: float(np.linalg.norm(deltas[n].astype(np.float32)) /
                         (np.linalg.norm(base_sd[n].numpy().astype(np.float32)) + 1e-12))
                for n in deltas}

    log = {
        "variant": args.variant,
        "model": args.model,
        "provenance": {
            "run_model": args.model,
            "target_model": "litert-community/Gemma3-1B-IT "
                            "(HF training mirror: google/gemma-3-1b-it)",
            "stand_in": is_stand_in,
            "note": ("CPU validation run on a GPT-2 124M stand-in; the same "
                     "command with --model google/gemma-3-1b-it produces the "
                     "Gemma-scale artifacts (26 layers, d=1152).") if is_stand_in else "",
        },
        "args": {"steps": args.steps, "lr": args.lr, "mse_cap": cfg["cap"],
                 "beta": cfg["beta"], "bench_at": args.bench_at,
                 "grad_clip": args.clip, "anchor_scope": args.anchor_scope,
                 "refresh_every": args.refresh_every,
                 "variant": args.variant, "trainable": trainable_names,
                 "pen_points": [f"{k}:{l}" for k, l in points],
                 "trainable_params": n_train,
                 "trainable_share": round(n_train / n_total, 5),
                 "prefix_len": args.prefix_len, "anchor_len": args.anchor_len,
                 "seed": args.seed, "dry_run": args.dry_run},
        "writers": writers,
        "deep_phase_layer": deep,
        "pen0": pen0,
        "pen_ref": pen_ref,
        "steps": steps_log,
        "bench": bench_log,
        "deltas_rel_frobenius": {k: round(v, 5) for k, v in rel_frob.items()},
    }
    dump_json(log, out / f"{args.variant}_training_log.json")
    torch.save(state, out / f"{args.variant}_trained_state.pt")
    np.savez_compressed(out / f"{args.variant}_deltas.npz", **deltas)
    final_bench = [b for b in bench_log if b["step"] == args.steps]
    print(f"[train] done: rho_mini {bench_log[0]['rho_mini']} -> "
          f"{final_bench[0]['rho_mini'] if final_bench else 'n/a'}, "
          f"pen_norm(last) {steps_log[-1]['pen_norm']:.2f}")
    print(f"[train] wrote {out}/{args.variant}_training_log.json")


if __name__ == "__main__":
    main()
