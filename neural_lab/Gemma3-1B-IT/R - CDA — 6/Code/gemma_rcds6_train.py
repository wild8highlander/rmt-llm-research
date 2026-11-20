#!/usr/bin/env python3
"""gemma_rcds6_train.py — RCDA-6 for Gemma3-1B-IT: the EXPANDED trainable set
with PHASE-conditioned loop directions (the hypothesis stated at the end of
the RCDA-5 report):

    "the loop is written jointly by attention output *and* MLP down_proj
     across a wider band (L8–L15 all show z_diff > 130), and the probe
     direction must be conditioned on the loop *phase*, not only the
     collapse point."

Two changes against RCDA-5 (gemma_rcds_train.py), everything else identical:

  1. Trainable tensors: the o_proj AND the down_proj of every band layer
     (default band L8-L15: 8 x (1152x1152 + 6912x1152) = 74.3 M parameters,
     7.1 % of the model, 21x the RCDA-5 trainable count).

  2. The penalty direction is conditioned on the loop PHASE of each position:
     phase_labels() splits every looping trajectory into pre / early / deep
     (deep = beyond the first `early_cycles` cycles; irregular tails are
     deep); gemma_probe_phase.py measured v_early_l / v_deep_l per band layer
     on the frozen RCDA-5 generations. The penalty projects each residual
     state h_t on the direction of ITS phase,

         pen_l = mean_t (v̂_{phase(t), l}ᵀ h_t)² ,
         phase(pre) = phase(early) = early direction, phase(deep) = deep.

     (pre positions carry the early direction: the RCDA-5 continuity — the
     pre-collapse states are penalized away from the earliest loop state.)

Step loss (unchanged from RCDA-5 — the formula is the same, the target is
better measured):

    L = pen_norm + beta * relu(mse - cap)
    pen_norm = pen_raw / pen0_j                  (dimensionless, init 1.0)
    pen_raw  = mean_l mean_t (v̂_{phase(t), l}ᵀ h_t)²   over band layers l
    mse      = MSE(trailing 32 logits, base logits) on a good prompt +
               the base greedy 192-token completion
    beta = 10, cap = 1.0, SGD momentum 0.9 lr 3e-3 grad-clip 1.0 (the
    RCDA-5 primary-arm recipe), fp32 masters over bf16 compute.

Data: identical frozen prompts, base greedy completions (anchor) and 48-token
bad prefixes (penalty); the cache is seeded from the RCDA-5 run (same model,
same greedy decoding — byte-identical data).

Usage:
  gemma_rcds6_train.py --model DIR --probe-phase WORK/probe_phase \
                       --band 8 9 10 11 12 13 14 15 --steps 150 --lr 3e-3
"""
from __future__ import annotations

import argparse
import json
import os
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
    """Linear wrapper: fp32 master weight, bf16 compute, exact fp32 grads."""

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = torch.nn.Parameter(weight.detach().float())

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight.to(x.dtype))


def replace_writer(model, layer: int, ttype: str) -> torch.nn.Parameter:
    """Swap the o_proj / down_proj nn.Linear of `layer` for FP32Linear."""
    mod = (model.model.layers[layer].self_attn.o_proj if ttype == "o_proj"
           else model.model.layers[layer].mlp.down_proj)
    w = mod.weight if isinstance(mod, torch.nn.Linear) else mod.weight.data
    lin = FP32Linear(w)
    if ttype == "o_proj":
        model.model.layers[layer].self_attn.o_proj = lin
    else:
        model.model.layers[layer].mlp.down_proj = lin
    return lin.weight


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
        # accept EOS-terminated completions too: greedy decoding may stop
        # before n tokens, and re-generating them every session is wasted
        # compute (the result is byte-identical)
        if gid in done and len(done[gid]["ids"]) <= n:
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
    ap.add_argument("--probe-phase", default=None,
                    help="probe_phase dir with v_early_L{N}.npy / v_deep_L{N}.npy")
    ap.add_argument("--band", type=int, nargs="+",
                    default=[8, 9, 10, 11, 12, 13, 14, 15],
                    help="writer band (default: the RCDA-5 hypothesis L8-L15)")
    ap.add_argument("--tensors", default="o_proj,down_proj",
                    help="comma-separated subset of {o_proj, down_proj}")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--optim", choices=["sgd", "adam"], default="sgd")
    ap.add_argument("--mse-cap", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=10.0)
    ap.add_argument("--bench-at", type=int, nargs="+", default=[75, 150])
    ap.add_argument("--bench-n", type=int, default=8)
    ap.add_argument("--good-new", type=int, default=L.MAX_NEW)
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--fresh-data", action="store_true")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore an existing ckpt.pt and start from step 0")
    ap.add_argument("--grad-ckpt", action="store_true",
                    help="enable gradient checkpointing for the training "
                         "forwards (RAM safety valve; benches unaffected)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    types = [t.strip() for t in args.tensors.split(",") if t.strip()]
    assert all(t in ("o_proj", "down_proj") for t in types), types
    band = sorted(set(args.band))
    out_dir = Path(args.out) if args.out else L.resolve_work() / "rcda6"
    out_dir.mkdir(parents=True, exist_ok=True)
    pp_dir = Path(args.probe_phase) if args.probe_phase \
        else L.resolve_work() / "probe_phase"

    from transformers import AutoModelForCausalLM

    tok = L.load_tokenizer(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.eval()
    if args.grad_ckpt:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        print("gradient checkpointing: ON (training forwards)", flush=True)

    # ---- phase-conditioned loop directions ----
    v_early, v_deep = {}, {}
    for l in band:
        v_early[l] = torch.tensor(np.load(pp_dir / f"v_early_L{l}.npy")
                                  .astype(np.float32))
        v_deep[l] = torch.tensor(np.load(pp_dir / f"v_deep_L{l}.npy")
                                 .astype(np.float32))
    vhat_early = {l: v_early[l] / v_early[l].norm() for l in band}
    vhat_deep = {l: v_deep[l] / v_deep[l].norm() for l in band}
    print(f"band: {band}; tensors: {types}; pen = phase-matched "
          f"(v^T h)^2 on layers {band}", flush=True)

    # ---- data (seeded from the RCDA-5 cache when present) ----
    t0 = time.time()
    good_items, bad_items = build_data(model, tok, args, out_dir)
    bad_prompts = L.load_bad_prompts()
    print(f"data ready in {time.time()-t0:.0f}s: {len(good_items)} good, "
          f"{len(bad_items)} bad", flush=True)
    if args.prepare_only:
        print("PREPARE ONLY — data cached, exiting", flush=True)
        return

    # ---- trainable: o_proj + down_proj of the band ----
    masters = {}
    for l in band:
        for t in types:
            masters[(l, t)] = replace_writer(model, l, t)
    for _, p in model.named_parameters():
        p.requires_grad_(False)
    for key in masters:
        masters[key].requires_grad_(True)
    trainable = list(masters.values())
    n_train = sum(p.numel() for p in trainable)
    print(f"trainable params: {n_train:,} "
          f"({100 * n_train / sum(p.numel() for p in model.parameters()):.2f} % "
          f"of the model) | RSS {L.rss_gb():.1f}GB", flush=True)
    if args.optim == "adam":
        opt = torch.optim.Adam(trainable, lr=args.lr)
        opt_desc = f"adam(lr={args.lr})"
    else:
        opt = torch.optim.SGD(trainable, lr=args.lr, momentum=args.momentum)
        opt_desc = f"sgd(lr={args.lr}, momentum={args.momentum}, grad_clip=1.0)"
    print(f"optimizer: {opt_desc}", flush=True)

    # ---- anchor logits cache: fp16 npy + memmap (file-backed, ~0 MB anon) ----
    anchor_npy = out_dir / "anchor_logits.npy"
    anchor_lens = out_dir / "anchor_logits_lens.npy"
    if anchor_npy.exists() and anchor_lens.exists() and not args.fresh_data:
        base_logits_store = np.load(anchor_npy, mmap_mode="r")
        base_logits_lens = np.load(anchor_lens)
        print(f"anchor store: memmap {base_logits_store.shape} fp16 "
              f"(rows per item: {base_logits_lens.tolist()})", flush=True)
    else:
        rows = []
        lens = []
        with torch.no_grad():
            for item in good_items:
                x = torch.tensor([item["seq"][:-1]])
                out = model(x, logits_to_keep=ANCHOR_TAIL + 1)
                lg = out.logits[0][-ANCHOR_TAIL:].half().detach().cpu().numpy()
                rows.append(lg)
                lens.append(lg.shape[0])
        T = max(lens)
        arr = np.zeros((len(rows), T, rows[0].shape[1]), dtype=np.float16)
        for i, lg in enumerate(rows):
            arr[i, :len(lg)] = lg
        np.save(anchor_npy, arr)
        np.save(anchor_lens, np.array(lens, dtype=np.int64))
        base_logits_store = np.load(anchor_npy, mmap_mode="r")
        base_logits_lens = np.load(anchor_lens)
        print(f"anchor store: built + memmap {arr.shape} fp16 "
              f"(rows per item: {lens}) | RSS {L.rss_gb():.1f}GB", flush=True)

    def penalty_on(item):
        """Phase-matched penalty over generated positions of a bad prefix."""
        x = torch.tensor([item["ids"]])
        out = model(x, output_hidden_states=True, logits_to_keep=1,
                    use_cache=False)
        hs = out.hidden_states                      # hs[l+1] = after layer l
        gen = item["ids"][item["plen"]:]
        labels, _ = L.phase_labels(gen, L.loop_collapse_point(gen))
        terms, phase_share = {}, {}
        for l in band:
            h = hs[l + 1][0, item["plen"]:]         # generated positions only
            he = ((h @ vhat_early[l].to(h.dtype)) ** 2)
            hd = ((h @ vhat_deep[l].to(h.dtype)) ** 2)
            early_mask = torch.tensor(labels != 2, dtype=h.dtype)
            deep_mask = torch.tensor(labels == 2, dtype=h.dtype)
            n_deep = int(deep_mask.sum())
            terms[l] = ((he * early_mask).sum()
                        + (hd * deep_mask).sum()) / max(len(gen), 1)
            phase_share[l] = round(n_deep / max(len(gen), 1), 3)
        return torch.stack(list(terms.values())).mean(), terms, phase_share

    def anchor_on(idx):
        x = torch.tensor([good_items[idx]["seq"][:-1]])
        n = int(base_logits_lens[idx])
        out = model(x, logits_to_keep=ANCHOR_TAIL + 1, use_cache=False)
        logits = out.logits[0][-n:].float()
        bl = torch.from_numpy(np.ascontiguousarray(base_logits_store[idx][:n]))
        mse = ((logits - bl.float()) ** 2).mean()
        return mse

    # ---- resume support (checkpoint every 10 steps) ----
    # ckpt_masters.pt: fp32 masters, mmap-loaded (READ-ONLY usage — safe);
    # ckpt_opt.pt: optimizer state + loop cursors, PLAIN load — mmap-backed
    # tensors must never end up inside the optimizer: the first in-place
    # momentum update on a mapped storage hangs the process;
    # training_log.json: the log itself (incl. pen0 + benches).
    ckpt_m = out_dir / "ckpt_masters.pt"
    ckpt_o = out_dir / "ckpt_opt.pt"
    log_path = out_dir / "training_log.json"
    start_step, gi, bi = 0, 0, 0
    resumed = False
    if (ckpt_m.exists() and ckpt_o.exists() and log_path.exists()
            and not args.fresh and not args.dry_run):
        log = json.load(log_path.open())
        start_step = log["steps"][-1]["step"] if log["steps"] else 0
        ck = torch.load(ckpt_m, weights_only=True, mmap=True,
                        map_location="cpu")
        with torch.no_grad():
            for key in masters:
                masters[key].copy_(ck[f"{key[0]}_{key[1]}"])
        del ck
        import gc
        gc.collect()
        co = torch.load(ckpt_o, weights_only=True, map_location="cpu")
        opt.load_state_dict(co["opt"])
        gi, bi = co["gi"], co["bi"]
        del co
        gc.collect()
        # paranoia: force optimizer state into plain RAM (no mapped storages)
        for group in opt.param_groups:
            for p in group["params"]:
                st = opt.state.get(p, {})
                for k2, v2 in list(st.items()):
                    if torch.is_tensor(v2):
                        st[k2] = v2.clone()
        resumed = True
        print(f"resumed from step {start_step} (gi={gi}, bi={bi})", flush=True)

    # ---- pen auto-scale from init magnitudes (reused from the log on resume) ----
    if resumed:
        pen0 = {j: float(log["pen0"][bad_items[j]["id"]])
                for j in range(len(bad_items))}
        pen_ref = float(np.median(list(pen0.values())))
        print(f"pen0 reused from log -> pen_ref={pen_ref:.3f}", flush=True)
    else:
        pen0 = {}
        with torch.no_grad():
            for j, item in enumerate(bad_items):
                pen0[j] = float(penalty_on(item)[0])
        pen_ref = float(np.median(list(pen0.values())))
        print(f"init pen per bad prompt: {[round(x, 2) for x in pen0.values()]}",
              flush=True)
        print(f"-> pen_ref={pen_ref:.3f}", flush=True)
        log = {"args": {**vars(args), "band": band, "tensor_types": types,
                        "trainable_params": n_train, "optimizer": opt_desc,
                        "loss": "pen_norm + beta*relu(mse-cap); pen_norm=pen_raw/pen0_j; "
                                "pen_raw = phase-matched (v_early/v_deep^T h)^2 over "
                                "band layers",
                        "hypothesis": "RCDA-5 report closing hypothesis: joint "
                                      "o_proj+down_proj writers, L8-L15 band, "
                                      "phase-conditioned directions"},
               "pen0": {bad_items[j]["id"]: round(pen0[j], 4) for j in pen0},
               "pen_ref": pen_ref, "steps": [], "bench": []}

    def bench(tag):
        if any(b["tag"] == tag for b in log["bench"]):
            print(f"[bench {tag}] already recorded — skip", flush=True)
            return None
        with torch.no_grad():
            rows = []
            for gid, p in bad_prompts[:args.bench_n]:
                if os.environ.get("GEMMA_DEBUG"):
                    print(f"  [dbg bench] generating {gid}...", flush=True)
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

    if not args.dry_run and start_step == 0 and not resumed:
        bench("step0")
    for step in range(start_step + 1, args.steps + 1):
        t0 = time.time()
        opt.zero_grad()
        mse = anchor_on(gi % len(good_items)); gi += 1
        _dbg = os.environ.get("GEMMA_DEBUG")
        if _dbg:
            print(f"  [dbg step {step}] anchor done ({time.time()-t0:.1f}s)",
                  flush=True)
        j = bi % len(bad_items); bi += 1
        pen_raw, terms, phase_share = penalty_on(bad_items[j])
        if _dbg:
            print(f"  [dbg step {step}] penalty done ({time.time()-t0:.1f}s)",
                  flush=True)
        pen = pen_raw / max(pen0[j], 1e-6)     # dimensionless, init = 1.0
        cap_viol = torch.relu(mse - args.mse_cap)
        loss = pen + args.beta * cap_viol
        loss.backward()
        if args.optim == "sgd":
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)      # free grads immediately
        rec = {"step": step, "mse": round(float(mse.detach()), 4),
               "pen_raw": round(float(pen_raw.detach()), 4),
               "pen_norm": round(float(pen.detach()), 4),
               "pen_layers": {f"{l}": round(float(terms[l].detach()), 4)
                              for l in band},
               "deep_share": phase_share[band[0]],
               "s": round(time.time() - t0, 1)}
        log["steps"].append(rec)
        if step % 10 == 0 or args.dry_run or os.environ.get("GEMMA_DEBUG"):
            print(f"[step {step}] mse={rec['mse']} pen_raw={rec['pen_raw']} "
                  f"pen_norm={rec['pen_norm']} ({rec['s']}s/step)", flush=True)
        if step in args.bench_at and not args.dry_run:
            bench(f"step{step}")
        if args.dry_run and step >= 2:
            break
        # incremental save every 10 steps: ATOMIC (tmp + os.replace), split
        # into a masters file (mmap-friendly) and an optimizer-state file
        if step % 10 == 0:
            tmp_m = ckpt_m.with_suffix(".pt.tmp")
            torch.save({f"{k[0]}_{k[1]}": masters[k].detach()
                        for k in masters}, tmp_m)
            os.replace(tmp_m, ckpt_m)
            tmp_o = ckpt_o.with_suffix(".pt.tmp")
            torch.save({"gi": gi, "bi": bi, "opt": opt.state_dict()}, tmp_o)
            os.replace(tmp_o, ckpt_o)
            tmp_j = (out_dir / "training_log.json.tmp")
            with tmp_j.open("w") as f:
                json.dump(log, f, indent=1)
            os.replace(tmp_j, log_path)

    if args.dry_run:
        print(f"DRY RUN OK | RSS {L.rss_gb():.1f}GB", flush=True)
        return

    # ---- save the edited tensors (delta state over the base checkpoint) ----
    trained = {}
    for (l, t) in masters:
        k = L.o_proj_key(l) if t == "o_proj" else L.down_proj_key(l)
        trained[k] = masters[(l, t)].detach().to(torch.bfloat16).clone()
    save_file(trained, out_dir / "trained_state.safetensors")
    print(f"saved trained_state.safetensors ({len(trained)} tensors, "
          f"{sum(t.numel() for t in trained.values()):,} weights)", flush=True)

    edited_keys = sorted(masters)          # [(layer, type)] for the forensics

    # ---- free everything the forensics do not need (the SVDs want RAM) ----
    del model, opt, trainable, base_logits_store
    for key in list(masters):
        del masters[key]
    import gc
    gc.collect()
    print(f"freed model/optimizer | RSS {L.rss_gb():.1f}GB", flush=True)

    # ---- TW / cos forensics of what the expanded training actually changed ----
    from safetensors import safe_open
    from marchenko_pastur import spectral_tw_control
    base_st = Path(args.model) / "model.safetensors"
    if not base_st.exists():
        from huggingface_hub import hf_hub_download
        base_st = Path(hf_hub_download(args.model, "model.safetensors"))

    analysis = {}
    partial_path = out_dir / "analysis_partial.json"
    if partial_path.exists():
        analysis = json.load(partial_path.open())
        print(f"analysis resume: {len(analysis)}/16 tensors already done",
              flush=True)
    with safe_open(str(base_st), framework="pt") as f:
        for (l, t) in edited_keys:
            k = L.o_proj_key(l) if t == "o_proj" else L.down_proj_key(l)
            if k in analysis:
                continue
            wb = f.get_tensor(k).float()
            wa = trained[k].float()
            d = (wa - wb)
            rel = float(torch.linalg.norm(d) / torch.linalg.norm(wb))
            v_d = np.load(pp_dir / f"v_deep_L{l}.npy").astype(np.float64)
            v_e = np.load(pp_dir / f"v_early_L{l}.npy").astype(np.float64)
            tw = spectral_tw_control(wb.numpy(), wa.numpy(), v_d)
            u, _, _ = np.linalg.svd(d.numpy(), full_matrices=False)
            u1 = u[:, 0]
            cos_d = float(abs(np.dot(u1, v_d)) / max(np.linalg.norm(u1), 1e-9)
                          * np.linalg.norm(v_d))
            cos_e = float(abs(np.dot(u1, v_e)) / max(np.linalg.norm(u1), 1e-9)
                          * np.linalg.norm(v_e))
            analysis[k] = {"layer": l, "type": t,
                           "rel_frobenius": round(rel, 5),
                           "tw": tw["verdict"],
                           "tw_detail": tw,
                           "cos_u1_v_deep": round(cos_d, 4),
                           "cos_u1_v_early": round(cos_e, 4),
                           "cols_changed_gt_1pct":
                               int((d.abs() > 0.01 * wb.abs().mean()).sum())}
            del wb, wa, d
            # incremental persistence: each tensor costs ~1-2 min of SVD —
            # a killed session must not recompute them
            with partial_path.open("w") as fj:
                json.dump(analysis, fj, indent=1)
            print(f"[analysis] {k}: tw={tw['verdict']} "
                  f"cos_u1_v_deep={round(cos_d, 4)}", flush=True)
    log["final"] = {"analysis": analysis}
    with (out_dir / "training_log.json.tmp").open("w") as f:
        json.dump(log, f, indent=1)
    os.replace(out_dir / "training_log.json.tmp",
               out_dir / "training_log.json")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "tw_detail"}
                      for k, v in analysis.items()}, indent=1))


if __name__ == "__main__":
    main()
