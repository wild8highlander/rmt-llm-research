#!/usr/bin/env python3
"""gemma_rcds_train.py — RCDA-5 for Gemma3-1B-IT: train the defect out, with
the formula AS the loss. Trainable tensors: the o_proj of the writer layers
found by gemma_probe.py (same tensors the surgical edit would touch).

Loss per step (identical structure to the MNN-Pocket rcda5_train.py):

    loss = pen_norm + beta * relu(mse - mse_cap)

    pen_norm = (pen_raw / pen0_j) * pen_ref
    pen_raw  = mean_l mean_t (v_l^T h_t / ||v_l||)^2
               over writer layers l, over the model's own looping greedy
               prefix positions t of a bad prompt (h = residual after layer l);
    mse      = MSE(trailing 32 logits, base logits) on a good prompt + the
               base greedy completion (the abilities anchor, hard cap 1.0);

The penalty term is exactly (I - P_v) expressed as a training signal: all
columns of the writer o_proj matrices learn to stop writing the loop
direction v. Per-prompt init normalization keeps one hot prompt from
dominating; Gemma3's residual norms are ~1000x Pocket-0.3B's, so the
normalized penalty is kept DIMENSIONLESS (pen_norm = pen_raw / pen0_j,
init 1.0) and the loss is

    loss = pen_norm + beta * relu(mse - mse_cap)

with gradient clipping at max-norm 1.0 for step stability.

Optimizer: SGD with momentum (flag-controlled, default 0.9) over fp32 master
weights; the model computes in bf16 with the fp32 masters cast per forward
(exact fp32 SGD updates, bf16 matmuls — Gemma3 runs in bf16).

Data (built once, cached with per-item resume in $GEMMA_WORK/rcda5/):
  * good: 20 factoid prompts + base greedy MAX_NEW completions (anchor);
  * bad:  20 sentinel prompts + base greedy 48-token prefixes (penalty).

Usage:
  gemma_rcds_train.py --model DIR --probe WORK/probe --steps 150 --lr 3e-4
                      [--momentum 0.9] [--bench-at 75 150] [--prepare-only]
                      [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gemma_lab as L                                    # noqa: E402

ANCHOR_TAIL = 32


class FP32Linear(torch.nn.Module):
    """o_proj wrapper: fp32 master weight, bf16 compute, exact fp32 grads."""

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = torch.nn.Parameter(weight.detach().float())

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight.to(x.dtype))


def replace_o_proj(model, layers):
    """Swap nn.Linear o_proj of `layers` for FP32Linear; return masters dict."""
    masters = {}
    for l in layers:
        mod = model.model.layers[l].self_attn.o_proj
        w = mod.weight if isinstance(mod, torch.nn.Linear) else mod.weight.data
        lin = FP32Linear(w)
        model.model.layers[l].self_attn.o_proj = lin
        masters[l] = lin.weight
    return masters


def build_data(model, tok, args, cache_dir: Path):
    """Good completions + bad prefixes, per-item resume in data_gens.jsonl."""
    bad = L.load_bad_prompts()
    good = L.load_good_prompts()
    gens_path = cache_dir / "data_gens.jsonl"
    done: dict[str, dict] = {}
    if gens_path.exists() and not args.fresh_data:
        for line in gens_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["id"]] = r
        print(f"data: {len(done)} cached generations", flush=True)

    def gen_cached(gid, text, n):
        if gid in done and len(done[gid]["ids"]) == n:
            return done[gid]["ids"]
        ids = L.gen_ids(model, tok, text, n)
        with gens_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"id": gid, "ids": ids}, ensure_ascii=False) + "\n")
        return ids

    good_items = []
    for gid, p in good:
        ids = gen_cached(gid, p, args.good_new)
        seq = L.encode_prompt(tok, p)[0].tolist() + ids
        good_items.append({"id": gid, "seq": seq})
    bad_items = []
    for gid, p in bad:
        plen = int(L.encode_prompt(tok, p).shape[1])
        ids = gen_cached(gid, p, L.BAD_PREF)
        bad_items.append({"id": gid, "plen": plen,
                          "ids": L.encode_prompt(tok, p)[0].tolist() + ids})
    return good_items, bad_items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-1b-it")
    ap.add_argument("--probe", default=None, help="probe dir with v_L{N}.npy")
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="writer layers (default: from probe_report.json)")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--optim", choices=["sgd", "adam"], default="sgd",
                    help="sgd (user-spec default) or adam (the MNN RCDA-5 recipe)")
    ap.add_argument("--mse-cap", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=10.0)
    ap.add_argument("--bench-at", type=int, nargs="+", default=[75, 150])
    ap.add_argument("--bench-n", type=int, default=8)
    ap.add_argument("--good-new", type=int, default=L.MAX_NEW)
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--fresh-data", action="store_true")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore an existing ckpt.pt and start from step 0")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else L.resolve_work() / "rcda5"
    out_dir.mkdir(parents=True, exist_ok=True)
    probe_dir = Path(args.probe) if args.probe else L.resolve_work() / "probe"

    from transformers import AutoModelForCausalLM

    tok = L.load_tokenizer(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.eval()

    # ---- writers and loop directions ----
    if args.layers:
        layers = sorted(args.layers)
    else:
        layers = json.load((probe_dir / "probe_report.json").open())["writers"]
    v = {l: torch.tensor(np.load(probe_dir / f"v_L{l}.npy").astype(np.float32))
         for l in layers}
    vhat = {l: v[l] / v[l].norm() for l in layers}
    print(f"writers: {layers}; pen = (v^T h)^2 on layers {layers}", flush=True)

    # ---- data: good prompt + base completion (anchor), bad prefixes (pen) ----
    t0 = time.time()
    good_items, bad_items = build_data(model, tok, args, out_dir)
    bad_prompts = L.load_bad_prompts()
    print(f"data ready in {time.time()-t0:.0f}s: {len(good_items)} good, "
          f"{len(bad_items)} bad", flush=True)
    if args.prepare_only:
        print("PREPARE ONLY — data cached, exiting", flush=True)
        return

    # ---- trainable: writer o_proj ----
    masters = replace_o_proj(model, layers)
    for n_, p in model.named_parameters():
        p.requires_grad_(False)
    for l in layers:
        masters[l].requires_grad_(True)
    trainable = [masters[l] for l in layers]
    print(f"trainable params: {sum(p.numel() for p in trainable):,}", flush=True)
    if args.optim == "adam":
        opt = torch.optim.Adam(trainable, lr=args.lr)
        opt_desc = f"adam(lr={args.lr})"
    else:
        opt = torch.optim.SGD(trainable, lr=args.lr, momentum=args.momentum)
        opt_desc = f"sgd(lr={args.lr}, momentum={args.momentum}, grad_clip=1.0)"
    print(f"optimizer: {opt_desc}", flush=True)

    # ---- base logits for the anchor: trailing 32 positions per good item ----
    with torch.no_grad():
        base_logits_store = []
        for item in good_items:
            x = torch.tensor([item["seq"][:-1]])
            out = model(x, logits_to_keep=ANCHOR_TAIL + 1)
            base_logits_store.append(
                out.logits[0][-ANCHOR_TAIL:].half().detach().clone())
    print(f"anchor store: {len(base_logits_store)} x "
          f"{tuple(base_logits_store[0].shape)} (fp16) | RSS {L.rss_gb():.1f}GB",
          flush=True)

    def penalty_on(item):
        x = torch.tensor([item["ids"]])
        out = model(x, output_hidden_states=True, logits_to_keep=1)
        hs = out.hidden_states                      # hs[l+1] = after layer l
        terms = {}
        for l in layers:
            h = hs[l + 1][0, item["plen"]:]         # generated positions only
            terms[l] = ((h @ vhat[l].to(h.dtype)) ** 2).mean()
        return torch.stack(list(terms.values())).mean(), terms

    def anchor_on(idx):
        x = torch.tensor([good_items[idx]["seq"][:-1]])
        out = model(x, logits_to_keep=ANCHOR_TAIL + 1)
        logits = out.logits[0][-ANCHOR_TAIL:].float()
        mse = ((logits - base_logits_store[idx].float()) ** 2).mean()
        return mse

    # ---- pen auto-scale from init magnitudes ----
    pen0 = {}
    with torch.no_grad():
        for j, item in enumerate(bad_items):
            pen0[j] = float(penalty_on(item)[0])
    pen_ref = float(np.median(list(pen0.values())))
    print(f"init pen per bad prompt: {[round(x, 2) for x in pen0.values()]}"
          f" -> pen_ref={pen_ref:.3f}", flush=True)
    log = {"args": {**vars(args), "layers": layers,
                    "optimizer": opt_desc,
                    "loss": "pen_norm + beta*relu(mse-cap); pen_norm=pen_raw/pen0_j (dimensionless)"},
           "pen0": {bad_items[j]["id"]: round(pen0[j], 4) for j in pen0},
           "pen_ref": pen_ref, "steps": [], "bench": []}

    # ---- resume support (checkpoint every 10 steps, like the MNN kit) ----
    ckpt_path = out_dir / "ckpt.pt"
    start_step, gi, bi = 0, 0, 0
    if ckpt_path.exists() and not args.fresh and not args.dry_run:
        ck = torch.load(ckpt_path, weights_only=False)
        start_step, gi, bi = ck["step"], ck["gi"], ck["bi"]
        log = ck["log"]
        with torch.no_grad():
            for l in layers:
                masters[l].copy_(ck["masters"][str(l)])
        opt.load_state_dict(ck["opt"])
        print(f"resumed from step {start_step} (gi={gi}, bi={bi})", flush=True)

    def bench(tag):
        if any(b["tag"] == tag for b in log["bench"]):
            print(f"[bench {tag}] already recorded — skip", flush=True)
            return None
        with torch.no_grad():
            rows = []
            for gid, p in bad_prompts[:args.bench_n]:
                ids = L.gen_ids(model, tok, p, L.MAX_NEW)
                ls = L.loop_share(ids)
                rows.append({"id": gid, "loop_share": round(ls, 3),
                             "n_tok": len(ids)})
        rho = sum(1 for r in rows if r["loop_share"] >= L.LOOP_THRESHOLD) / len(rows)
        rec = {"tag": tag, "rho_mini": rho,
               "mean_ls": round(float(np.mean([r["loop_share"] for r in rows])), 3),
               "gens": rows}
        log["bench"].append(rec)
        print(f"[bench {tag}] rho_mini={rho} mean_ls={rec['mean_ls']}", flush=True)
        json.dump(log, (out_dir / "training_log.json").open("w"), indent=1)
        return rho

    if not args.dry_run and start_step == 0:
        bench("step0")
    for step in range(start_step + 1, args.steps + 1):
        t0 = time.time()
        opt.zero_grad()
        mse = anchor_on(gi % len(good_items)); gi += 1
        j = bi % len(bad_items); bi += 1
        pen_raw, terms = penalty_on(bad_items[j])
        pen = pen_raw / max(pen0[j], 1e-6)     # dimensionless, init = 1.0
        cap_viol = torch.relu(mse - args.mse_cap)
        loss = pen + args.beta * cap_viol
        loss.backward()
        if args.optim == "sgd":
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        opt.step()
        # FP32Linear casts the fp32 master to bf16 on every forward, so the
        # compute weights are always the current masters — nothing to copy.
        rec = {"step": step, "mse": round(float(mse.detach()), 4),
               "pen_raw": round(float(pen_raw.detach()), 4),
               "pen_norm": round(float(pen.detach()), 4),
               "pen_layers": {str(l): round(float(terms[l].detach()), 4)
                              for l in layers},
               "s": round(time.time() - t0, 1)}
        log["steps"].append(rec)
        if step % 10 == 0 or args.dry_run:
            print(f"[step {step}] mse={rec['mse']} pen_raw={rec['pen_raw']} "
                  f"pen_norm={rec['pen_norm']} ({rec['s']}s/step)", flush=True)
        if step in args.bench_at and not args.dry_run:
            bench(f"step{step}")
        if args.dry_run and step >= 2:
            break
        # incremental save: checkpoint every 10 steps (session may be cut)
        if step % 10 == 0:
            torch.save({"step": step, "gi": gi, "bi": bi,
                        "masters": {str(l): masters[l].detach().clone()
                                    for l in layers},
                        "opt": opt.state_dict(), "log": log},
                       ckpt_path)
            json.dump(log, (out_dir / "training_log.json").open("w"), indent=1)

    if args.dry_run:
        print("DRY RUN OK", flush=True)
        return

    # ---- save the edited tensors (delta state over the base checkpoint) ----
    trained = {L.o_proj_key(l): masters[l].detach().to(torch.bfloat16).clone()
               for l in layers}
    save_file(trained, out_dir / "trained_state.safetensors")

    # ---- TW / cos forensics of what SGD actually changed ----
    from safetensors.torch import load_file
    from marchenko_pastur import spectral_tw_control
    base_path = Path(args.model)
    base_st = base_path / "model.safetensors" if base_path.is_dir() else None
    if base_st is None or not base_st.exists():
        from huggingface_hub import hf_hub_download
        base_st = hf_hub_download(args.model, "model.safetensors")
    base_sd_w = load_file(str(base_st))

    analysis = {}
    for l in layers:
        k = L.o_proj_key(l)
        wb = base_sd_w[k].float()
        wa = trained[k].float()
        d = (wa - wb)
        rel = float(torch.linalg.norm(d) / torch.linalg.norm(wb))
        tw = spectral_tw_control(wb.numpy(), wa.numpy(),
                                 np.load(probe_dir / f"v_L{l}.npy").astype(np.float64))
        u, s_, _ = np.linalg.svd(d.numpy(), full_matrices=False)
        vnp = np.load(probe_dir / f"v_L{l}.npy")
        u1 = u[:, 0]
        cos_u1_v = float(abs(np.dot(u1, vnp)) / max(np.linalg.norm(u1)
                        * np.linalg.norm(vnp), 1e-9))
        analysis[k] = {"rel_frobenius": round(rel, 5),
                       "tw": tw["verdict"], "tw_detail": tw,
                       "cos_top_left_sv_vs_v": round(cos_u1_v, 4),
                       "cos_u1_v": round(cos_u1_v, 4),
                       "cols_changed_gt_1pct":
                           int((d.abs() > 0.01 * wb.abs().mean()).sum())}
    log["final"] = {"analysis": analysis}
    json.dump(log, (out_dir / "training_log.json").open("w"), indent=1)
    print("saved trained_state.safetensors")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "tw_detail"}
                      for k, v in analysis.items()}, indent=1))


if __name__ == "__main__":
    main()
