#!/usr/bin/env python3
"""stage4_eval.py — the one evaluator of the LlamaFactory round (RCDA-8 port).

Every arm ends here: the base model, the RCDA-8 LoRA (formula-trained) and the
SFT LoRA (data-trained). One frozen protocol, byte-compatible with the RCDA-5/6
real-Gemma rounds:

  * 20 bad + 20 good prompts, raw completion mode, greedy 192 tokens
  * 4-gram loop labels, loop threshold 0.5 -> rho_bad / rho_good
  * loop_share / loop_waste means + per-trajectory rows (jsonl, resume-safe)
  * hard-12 / mild-8 split and fully-repaired count vs the base arm
  * KL(base || arm) + PPL on the 5 control texts (base logits cached, fp16
    memmap — the box has 4 GB RAM, two 2 GB models do not fit)
  * per-tensor forensics of the LoRA edit: rel_frobenius, spectral TW control,
    cos(u1, v_deep), cols changed > 1% (only for LoRA arms)

Usage:
  stage4_eval.py --tag base
  stage4_eval.py --tag rcda8 --lora ../results/rcda8_arm/adapter
  stage4_eval.py --tag sft    --lora ../sft_arm/gemma3_sft_lora
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rcda8_lab as L                                          # noqa: E402
from marchenko_pastur import spectral_tw_control               # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=["base", "rcda8", "sft"])
    ap.add_argument("--model", default=L.LOCAL_MODEL)
    ap.add_argument("--lora", default=None, help="peft adapter dir")
    ap.add_argument("--max-new", type=int, default=L.MAX_NEW)
    ap.add_argument("--n-good", type=int, default=20,
                    help="good prompts to generate (protocol default 20)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM

    torch.set_num_threads(2)
    out_dir = Path(args.out) if args.out else L.RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    if args.lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora)
        print(f"[eval] LoRA adapter loaded: {args.lora}", flush=True)
    model.eval()
    tok = L.load_tokenizer(args.model)

    bad = L.load_bad_prompts()
    good = L.load_good_prompts()[: args.n_good]
    ctrl = L.load_control_texts()

    # ---------------- 1. generations (resume-safe per prompt) --------------
    gens_path = out_dir / f"gens_{args.tag}.jsonl"
    done: dict[str, dict] = {}
    if gens_path.exists():
        for line in gens_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["id"]] = r
        print(f"[eval] resume: {len(done)} generations on disk", flush=True)

    def gen_row(pid, text, kind):
        if pid in done:
            return done[pid]
        t0 = time.time()
        ids = L.gen_ids(model, tok, text, args.max_new)
        r = {"id": pid, "kind": kind, "loop_share": round(L.loop_share(ids), 4),
             "loop_waste": L.loop_waste(ids), "n_tok": len(ids),
             "text_out": tok.decode(ids, skip_special_tokens=True)[:600]}
        with gens_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        done[pid] = r
        print(f"[gen] {pid}: ls={r['loop_share']:.2f} ({time.time()-t0:.0f}s)",
              flush=True)
        return r

    rows_bad = [gen_row(pid, text, "bad") for pid, text in bad]
    rows_good = [gen_row(pid, text, "good") for pid, text in good]

    rho_bad = float(np.mean([r["loop_share"] >= L.LOOP_THRESHOLD for r in rows_bad]))
    rho_good = float(np.mean([r["loop_share"] >= L.LOOP_THRESHOLD for r in rows_good]))
    ls_bad = [r["loop_share"] for r in rows_bad]

    # hard/mild split + repaired count vs the base arm (needs base rows)
    base_rows_path = out_dir / "gens_base.jsonl"
    hard12, mild8, repaired = None, None, None
    if args.tag != "base" and base_rows_path.exists():
        base_rows = {json.loads(line)["id"]: json.loads(line)
                     for line in base_rows_path.read_text(encoding="utf-8").splitlines()
                     if line.strip()}
        hard_ids = [i for i, r in base_rows.items()
                    if r["loop_share"] >= 0.5]
        hard12 = [round(r["loop_share"], 3) for r in rows_bad
                  if r["id"] in hard_ids]
        mild8 = [round(r["loop_share"], 3) for r in rows_bad
                 if r["id"] not in hard_ids]
        repaired = sum(1 for r in rows_bad
                       if r["id"] in hard_ids and r["loop_share"] < L.LOOP_THRESHOLD)

    # ---------------- 2. KL + PPL on the control corpus --------------------
    base_store = out_dir / "ctrl_base_logits.npy"
    base_lens = out_dir / "ctrl_base_lens.npy"
    if not base_store.exists():
        # first arm to run here builds the fp16 cache with the PRISTINE model
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16)
        base_model.eval()
        rows, lens = [], []
        with torch.no_grad():
            for cid, text in ctrl:
                ids = L.encode_prompt(tok, text)
                lg = base_model(ids).logits[0].half().numpy()
                rows.append(lg)
                lens.append(lg.shape[0])
        T = max(lens)
        arr = np.zeros((len(rows), T, rows[0].shape[1]), dtype=np.float16)
        for i, lg in enumerate(rows):
            arr[i, :len(lg)] = lg
        np.save(base_store, arr)
        np.save(base_lens, np.array(lens, dtype=np.int64))
        del base_model
        print(f"[eval] base control-logit cache built {arr.shape}", flush=True)
    store = np.load(base_store, mmap_mode="r")
    lens = np.load(base_lens)

    kls, ppls = [], []
    with torch.no_grad():
        for i, (cid, text) in enumerate(ctrl):
            ids = L.encode_prompt(tok, text)
            lg_arm = model(ids).logits[0].float()
            n = int(lens[i])
            lg_base = torch.from_numpy(
                np.ascontiguousarray(store[i][:n])).float()
            logp = torch.log_softmax(lg_arm[:n], -1)
            logq = torch.log_softmax(lg_base, -1)
            kl = (logp.exp() * (logp - logq)).sum(-1).mean().item()
            nll = torch.nn.functional.cross_entropy(
                lg_arm[:n][:-1], ids[0, 1:]).item()
            kls.append(kl)
            ppls.append(float(np.exp(nll)))
            print(f"[ctrl] {cid}: KL={kl:.4f} PPL={np.exp(nll):.1f}", flush=True)

    # base arm PPL comes from the same cache (arm == base)
    if args.tag == "base":
        ppls = []
        with torch.no_grad():
            for i, (cid, text) in enumerate(ctrl):
                ids = L.encode_prompt(tok, text)
                lg = torch.from_numpy(
                    np.ascontiguousarray(store[i][:int(lens[i])])).float()
                nll = torch.nn.functional.cross_entropy(
                    lg[:-1], ids[0, 1:]).item()
                ppls.append(float(np.exp(nll)))

    # ---------------- 3. LoRA forensics: the edit as a matrix --------------
    analysis = {}
    partial_path = out_dir / f"forensics_partial_{args.tag}.json"
    if partial_path.exists():
        analysis = json.loads(partial_path.read_text(encoding="utf-8"))
        print(f"[eval] forensics resume: {len(analysis)} tensors done", flush=True)
    if args.lora:
        probe_dir = L.RESULTS_DIR / "probe"
        vs = np.load(probe_dir / "vectors_phase.npz")
        peft_model = model.base_model.model if hasattr(model, "base_model") else model
        base_sd = None
        # delta per target module: dW = (B @ A) * scale
        for name, mod in peft_model.named_modules():
            if not hasattr(mod, "lora_A") or "default" not in getattr(
                    mod, "lora_A", {}):
                continue
            fore_key = name.replace(".lora_A.default", "").replace(
                ".base_layer", "")
            fore_key = fore_key + ".weight"
            if fore_key in analysis:
                continue
            lA = mod.lora_A["default"].weight.detach().float()
            lB = mod.lora_B["default"].weight.detach().float()
            scale = mod.scaling["default"]
            dW = (lB @ lA) * scale
            if float(dW.abs().max()) == 0.0:
                continue        # band-frozen adapter (never trained)
            if base_sd is None:
                base_sd = {}
                from safetensors import safe_open
                st_path = Path(args.model) / "model.safetensors"
                with safe_open(str(st_path), framework="pt") as f:
                    for k in f.keys():
                        if k.endswith("o_proj.weight") or k.endswith("down_proj.weight"):
                            base_sd[k] = f.get_tensor(k)
            key = name.replace("base_model.model.", "").replace(".base_layer", "")
            key = key + ".weight" if not key.endswith(".weight") else key
            wb = base_sd.get(key)
            if wb is None:
                # peft name mapping fallback: reconstruct from module path
                key = name.split(".base_layer")[0].replace(
                    "base_model.model.", "") + ".weight"
                wb = base_sd.get(key)
            if wb is None:
                print(f"[forensics] skip {name}: no base tensor", flush=True)
                continue
            wb = wb.float()
            rel = float(torch.linalg.norm(dW) / (torch.linalg.norm(wb) + 1e-12))
            lnum = L.parse_layer(key)
            v_d = vs[f"v_deep_L{lnum}"].astype(np.float64) if f"v_deep_L{lnum}" in vs.files else None
            tw = spectral_tw_control(wb.numpy().astype(np.float64),
                                     (wb.numpy() + dW.numpy()).astype(np.float64),
                                     v_d)
            u, _, _ = np.linalg.svd(dW.numpy(), full_matrices=False)
            u1 = u[:, 0]
            cos_d = float(abs(u1 @ (v_d / np.linalg.norm(v_d)))) if v_d is not None else None
            analysis[key] = {
                "layer": lnum, "rel_frobenius": round(rel, 5),
                "tw": tw["verdict"],
                "tw_z_resid": round(tw.get("z_resid_tw", tw.get("z_resid", 0.0)), 2),
                "cos_u1_v_deep": round(cos_d, 4) if cos_d is not None else None,
                "cols_changed_gt_1pct":
                    int((dW.abs() > 0.01 * wb.abs().mean()).sum()),
            }
            L.dump_json(analysis, partial_path)
            print(f"[forensics] {key}: rel={rel:.5f} tw={tw['verdict']} "
                  f"cos_u1_v_deep={analysis[key]['cos_u1_v_deep']}", flush=True)

    result = {
        "tag": args.tag,
        "model": L.MODEL_SNAPSHOT,
        "lora": str(args.lora) if args.lora else None,
        "round": "RCDA-8 LlamaFactory round",
        "protocol": {"n_bad": len(rows_bad), "n_good": len(rows_good),
                     "max_new": args.max_new, "ngram": 4,
                     "rho_threshold": L.LOOP_THRESHOLD,
                     "decoding": "greedy", "mode": "raw completion",
                     "precision": "bf16 weights / fp32 logits"},
        "rho_bad": rho_bad, "rho_good": rho_good,
        "loop_share_mean_bad": round(float(np.mean(ls_bad)), 4),
        "loop_waste_mean_bad": round(float(np.mean(
            [r["loop_waste"] for r in rows_bad])), 1),
        "per_traj_loop_share_bad": [round(r["loop_share"], 3) for r in rows_bad],
        "hard12_loop_share": hard12, "mild8_loop_share": mild8,
        "prompts_fully_repaired": repaired,
        "kl_ctrl_mean": round(float(np.mean(kls)), 4),
        "kl_per_text": [round(k, 4) for k in kls],
        "ppl_control_mean": round(float(np.mean(ppls)), 1),
        "ppl_per_text": [round(p, 1) for p in ppls],
        "lora_forensics": analysis,
    }
    L.dump_json(result, out_dir / f"part8_eval_{args.tag}.json")
    print(f"[eval] {args.tag}: rho_bad={rho_bad:.2f} rho_good={rho_good:.2f} "
          f"ls={result['loop_share_mean_bad']:.3f} KL={result['kl_ctrl_mean']:.4f} "
          f"PPL={result['ppl_control_mean']:.1f} "
          f"repaired={repaired}/{12 if hard12 is not None else '-'}", flush=True)


if __name__ == "__main__":
    main()
