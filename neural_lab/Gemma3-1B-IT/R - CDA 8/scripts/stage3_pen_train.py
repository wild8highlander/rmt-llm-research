#!/usr/bin/env python3
"""stage3_pen_train.py — the formula as a loss, on LlamaFactory rails (RCDA-9).

A faithful re-run of the RCDA-8 formula arm inside the RCDA-9 round: the
RCDA-6 recipe (phase-matched penalty directions over the o_proj+down_proj
writer band, hard abilities anchor) as a LoRA adapter trained through
LlamaFactory's trainer machinery:

  * model + LoRA injection ... llamafactory.model.load_model
    (lora_target=o_proj,down_proj, r=16, alpha=32)
  * trainer .................. llamafactory.train.sft.trainer.CustomSeq2SeqTrainer
    subclassed with the RCDA compute_loss (the same extension pattern as
    LlamaFactory's own use_asft_loss branch: run the model inside
    compute_loss, add the custom term)
  * optimizer/scheduler/
    checkpointing/logging ..... inherited from the LlamaFactory trainer
  * the band mask freezes every LoRA adapter outside the writer band, so
    the effective trainable set is exactly the RCDA-6 band (L8-L15)

The formula (unchanged since RCDA-5, per the repo's monograph):

    L = pen_norm + beta * relu(mse - cap)
    pen_norm = pen_raw / pen0_j                    (dimensionless, init 1.0)
    pen_raw  = mean_l mean_t ( v̂_{phase(t), l}^T h_t )^2   over band layers
    mse      = MSE(trailing-32 logits, base logits) on a good prompt

Data: the frozen protocol sets — 20 bad prefixes (48 greedy tokens, reusing
the RCDA-9 stage-1 frozen generations) and 20 good completions (192 greedy
tokens), alternating in the RCDA-6 order. Anchor logits cached fp16 on disk.

Outputs (rcda9/results/pen_arm/):
  training_log.json   steps (mse/pen_raw/pen_norm per layer) + benches
  adapter/            the LoRA adapter
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rcda9_lab as L                                          # noqa: E402

BAND = [8, 9, 10, 11, 12, 13, 14, 15]


class RCDADataset(torch.utils.data.Dataset):
    """Interleaved RCDA items: 20 bad prefixes then 20 good anchor seqs."""

    def __init__(self, bad_items, good_items):
        self.items = bad_items + good_items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class PassthroughCollator:
    def __call__(self, batch):
        assert len(batch) == 1, "RCDA-9 trains with per-device batch 1"
        item = dict(batch[0])
        item["input_ids"] = torch.tensor(item["input_ids"], dtype=torch.long)
        if "phase" in item:
            item["phase"] = torch.tensor(item["phase"], dtype=torch.long)
        return item


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--beta", type=float, default=10.0)
    ap.add_argument("--cap", type=float, default=1.0)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--band", type=int, nargs="+", default=BAND)
    ap.add_argument("--bench-at", type=int, nargs="+", default=[50, 100, 150])
    ap.add_argument("--bench-n", type=int, default=8)
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    band = sorted(args.band)
    out_dir = Path(args.out) if args.out else L.RESULTS_DIR / "pen_arm"
    out_dir.mkdir(parents=True, exist_ok=True)
    probe_dir = L.RESULTS_DIR / "probe"

    from llamafactory.hparams import (DataArguments, FinetuningArguments,
                                      GeneratingArguments, ModelArguments,
                                      TrainingArguments as LFTrainingArguments)
    from llamafactory.model import load_model, load_tokenizer
    from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer

    # ---------------- LlamaFactory model + LoRA ----------------------------
    model_args = ModelArguments(model_name_or_path=L.LOCAL_MODEL)
    fin_args = FinetuningArguments(
        stage="sft", finetuning_type="lora",
        lora_target="o_proj,down_proj",
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        disable_shuffling=True,
    )
    tokenizer_module = load_tokenizer(model_args)
    tok = tokenizer_module["tokenizer"]
    model = load_model(tok, model_args, fin_args, is_trainable=True)
    print(f"[pen] model loaded | RSS {L.rss_gb():.1f}GB", flush=True)

    # band mask: freeze every LoRA adapter outside the writer band
    def lnum_of(name: str):
        parts = name.split(".")
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                return int(parts[i + 1])
        return None

    n_train = 0
    n_frozen = 0
    for name, p in model.named_parameters():
        if "lora_" in name:
            keep = lnum_of(name) in band
            p.requires_grad_(keep)
            if keep:
                n_train += p.numel()
            else:
                n_frozen += 1
    print(f"[pen] band mask: LoRA trainable on L{band[0]}-L{band[-1]} "
          f"-> {n_train:,} params ({n_frozen} adapters frozen outside)",
          flush=True)
    assert n_train > 0, "band mask froze every adapter - check layer parsing"

    # ---------------- frozen data (stage-1 generations reused) ------------
    bad = L.load_bad_prompts()
    good = L.load_good_prompts()
    gens = {}
    for line in (probe_dir / "gens.jsonl").read_text(
            encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            gens[r["id"]] = r
    good_cache = L.DATA_DIR / "good_gens_cache.jsonl"
    for line in good_cache.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            gens.setdefault(r["id"], r)

    bad_items = []
    for pid, text in bad:
        ids = gens[pid]["gen_ids"][:L.BAD_PREF]
        plen = int(L.encode_prompt(tok, text).shape[1])
        seq = L.encode_prompt(tok, text)[0].tolist() + ids
        labels, meta = L.phase_labels(ids, L.loop_collapse_point(ids))
        bad_items.append({"input_ids": seq, "plen": plen,
                          "phase": labels.tolist(), "kind": "bad",
                          "id": pid, "pen0": 0.0})
    good_items = []
    for gid, text in good:
        ids = gens[gid]["ids"][:L.MAX_NEW]
        seq = L.encode_prompt(tok, text)[0].tolist() + ids
        good_items.append({"input_ids": seq, "kind": "good", "id": gid,
                           "anchor_len": L.ANCHOR_TAIL})
    print(f"[pen] data: {len(bad_items)} bad prefixes + "
          f"{len(good_items)} good anchors", flush=True)

    # ---------------- phase directions from the stage-1 probe -------------
    vs = np.load(probe_dir / "vectors_phase.npz")
    vhat = {}
    for l in band:
        for ph, key in (("early", f"v_early_L{l}"), ("deep", f"v_deep_L{l}")):
            v = torch.tensor(vs[key].astype(np.float32))
            vhat[(ph, l)] = v / v.norm()

    # ---------------- anchor logits cache (fp16 memmap) --------------------
    anchor_npy = out_dir / "anchor_logits.npy"
    anchor_lens = out_dir / "anchor_logits_lens.npy"
    if not (anchor_npy.exists() and anchor_lens.exists()):
        model.eval()
        rows_, lens_ = [], []
        with torch.no_grad():
            for item in good_items:
                x = torch.tensor([item["input_ids"][:-1]])
                lg = model(x, logits_to_keep=L.ANCHOR_TAIL + 1).logits
                lg = lg[0][-L.ANCHOR_TAIL:].half().numpy()
                rows_.append(lg)
                lens_.append(lg.shape[0])
        T = max(lens_)
        arr = np.zeros((len(rows_), T, rows_[0].shape[1]), dtype=np.float16)
        for i, lg in enumerate(rows_):
            arr[i, :len(lg)] = lg
        np.save(anchor_npy, arr)
        np.save(anchor_lens, np.array(lens_, dtype=np.int64))
        print(f"[pen] anchor cache built {arr.shape} | RSS {L.rss_gb():.1f}GB",
              flush=True)
    base_store = np.load(anchor_npy, mmap_mode="r")
    base_lens = np.load(anchor_lens)
    good_index = {item["id"]: i for i, item in enumerate(good_items)}

    # ---------------- pen0 per bad prompt (init = 1.0 normalisation) -------
    log_path = out_dir / "training_log.json"
    if log_path.exists():
        log = json.loads(log_path.read_text(encoding="utf-8"))
    else:
        log = None

    def as_batch(x):
        if isinstance(x, torch.Tensor):
            return x.unsqueeze(0) if x.dim() == 1 else x
        return torch.tensor([x])

    def pen_raw_on(model, item, vhat, band):
        """The formula, live tensor: phase-matched (v^T h)^2 on band layers."""
        x = as_batch(item["input_ids"])
        out = model(x, output_hidden_states=True, logits_to_keep=1,
                    use_cache=False)
        hs = out.hidden_states
        plen, gen = item["plen"], item["input_ids"][item["plen"]:]
        labels = item["phase"]
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)
        terms = {}
        for l in band:
            h = hs[l + 1][0, plen:]
            he = (h @ vhat[("early", l)].to(h.dtype)) ** 2
            hd = (h @ vhat[("deep", l)].to(h.dtype)) ** 2
            early_mask = (labels != 2).to(h.dtype)
            deep_mask = (labels == 2).to(h.dtype)
            terms[l] = ((he * early_mask).sum()
                        + (hd * deep_mask).sum()) / max(len(gen), 1)
        return torch.stack(list(terms.values())).mean(), terms

    if log is None or "pen0" not in log:
        model.eval()
        pen0 = {}
        with torch.no_grad():
            for j, item in enumerate(bad_items):
                pen0[j] = float(pen_raw_on(model, item, vhat, band)[0])
        print(f"[pen] pen0: {[round(x, 2) for x in pen0.values()]}", flush=True)
        log = {
            "variant": "rcda9_pen",
            "framework": "LlamaFactory 0.9.5 (CustomSeq2SeqTrainer + load_model)",
            "model": L.MODEL_SNAPSHOT,
            "formula": "L = pen_norm + beta*relu(mse-cap); pen_norm = "
                       "phase-matched (v_early/v_deep^T h)^2 / pen0_j over band",
            "args": {"steps": args.steps, "lr": args.lr, "beta": args.beta,
                     "cap": args.cap, "band": band,
                     "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha,
                     "lora_target": "o_proj,down_proj (band-masked)",
                     "trainable_params": n_train,
                     "optimizer": "adamw_torch (LlamaFactory default)",
                     "bench_at": args.bench_at},
            "pen0": {bad_items[j]["id"]: round(v, 4) for j, v in pen0.items()},
            "steps": [], "bench": [],
        }
        L.dump_json(log, log_path)
    pen0 = {j: log["pen0"][bad_items[j]["id"]]
            for j in range(len(bad_items))}

    # ---------------- the trainer ------------------------------------------
    t_args = LFTrainingArguments(
        output_dir=str(out_dir / "trainer_out"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        max_steps=args.steps,
        learning_rate=args.lr,
        lr_scheduler_type="linear",
        warmup_steps=10,
        logging_steps=1,
        save_strategy="steps", save_steps=10, save_total_limit=2,
        bf16=False, fp16=False,                      # bf16 weights, no autocast
        report_to=[], seed=0, dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    class RCDASeqTrainer(CustomSeq2SeqTrainer):
        """LlamaFactory trainer + the RCDA formula in compute_loss."""

        def __init__(self, *a, rcda_ctx=None, **kw):
            super().__init__(*a, **kw)
            self.rcda_ctx = rcda_ctx or {}

        def compute_loss(self, model, inputs, return_outputs=False,
                         num_items_in_batch=None):
            ctx = self.rcda_ctx
            if inputs["kind"] == "bad":
                j = ctx["bad_index"][inputs["id"]]
                pen_raw, terms = pen_raw_on(model, inputs, ctx["vhat"],
                                            ctx["band"])
                pen_norm = pen_raw / max(ctx["pen0"][j], 1e-6)
                loss = pen_norm
                self._last = {"kind": "bad", "id": inputs["id"],
                              "pen_raw": float(pen_raw.detach()),
                              "pen_norm": float(pen_norm.detach()),
                              "mse": 0.0}
            else:
                i = ctx["good_index"][inputs["id"]]
                x = as_batch(inputs["input_ids"])[:, :-1]
                out = model(x, logits_to_keep=L.ANCHOR_TAIL + 1, use_cache=False)
                n = int(base_lens[i])
                logits = out.logits[0][-n:].float()
                bl = torch.from_numpy(
                    np.ascontiguousarray(base_store[i][:n])).float()
                mse = ((logits - bl) ** 2).mean()
                loss = ctx["beta"] * torch.relu(mse - ctx["cap"])
                self._last = {"kind": "good", "id": inputs["id"],
                              "mse": float(mse.detach()),
                              "pen_norm": 0.0, "pen_raw": 0.0}
            return (loss, None) if return_outputs else loss

    from transformers import TrainerCallback

    class StepLogCallback(TrainerCallback):
        """Append each step to training_log.json (RCDA-6 schema)."""

        def __init__(self, trainer_ref, log, log_path):
            self.trainer_ref, self.log, self.log_path = trainer_ref, log, log_path
            self.t0 = time.time()

        def on_step_end(self, targs, state, control, **kw):
            tr = self.trainer_ref()
            if tr is None or not hasattr(tr, "_last"):
                return control
            rec = {"step": state.global_step, **tr._last,
                   "s": round(time.time() - self.t0, 1)}
            rec.pop("id", None)
            self.log["steps"].append(rec)
            if state.global_step % 10 == 0:
                L.dump_json(self.log, self.log_path)
                print(f"[step {state.global_step}] {tr._last['kind']} "
                      f"pen_norm={tr._last['pen_norm']:.3f} "
                      f"mse={tr._last['mse']:.4f} ({rec['s']}s cum)", flush=True)
            return control

    class BenchCallback(TrainerCallback):
        """Mini-benchmark (8 bad, greedy 192) at the protocol steps."""

        def __init__(self, trainer_ref, log, log_path, model, tok, bad,
                     at, n):
            self.trainer_ref, self.log, self.log_path = trainer_ref, log, log_path
            self.model, self.tok, self.bad = model, tok, bad
            self.at, self.n = set(at), n

        def on_step_end(self, targs, state, control, **kw):
            if state.global_step not in self.at:
                return control
            if any(b["tag"] == f"step{state.global_step}" for b in self.log["bench"]):
                return control
            self.model.eval()
            rows = []
            with torch.no_grad():
                for pid, p in self.bad[: self.n]:
                    ids = L.gen_ids(self.model, self.tok, p, L.MAX_NEW)
                    rows.append({"id": pid, "loop_share": round(L.loop_share(ids), 3),
                                 "n_tok": len(ids)})
            self.model.train()
            rho = sum(1 for r in rows if r["loop_share"] >= L.LOOP_THRESHOLD) / len(rows)
            rec = {"tag": f"step{state.global_step}", "rho_mini": rho,
                   "mean_ls": round(float(np.mean([r["loop_share"] for r in rows])), 3),
                   "gens": rows}
            self.log["bench"].append(rec)
            L.dump_json(self.log, self.log_path)
            print(f"[bench step{state.global_step}] rho_mini={rho} "
                  f"mean_ls={rec['mean_ls']}", flush=True)
            return control

    dataset = RCDADataset(bad_items, good_items)
    ctx = {"vhat": vhat, "band": band, "pen0": pen0, "beta": args.beta,
           "cap": args.cap, "bad_index": {it["id"]: j for j, it in enumerate(bad_items)},
           "good_index": good_index}

    import weakref
    holder = {}
    trainer = RCDASeqTrainer(
        model=model,
        args=t_args,
        finetuning_args=fin_args,
        processor=None,
        train_dataset=dataset,
        data_collator=PassthroughCollator(),
        tokenizer=tok,
        rcda_ctx=ctx,
    )
    holder["t"] = trainer
    tref = weakref.ref(trainer)
    trainer.add_callback(StepLogCallback(tref, log, log_path))
    trainer.add_callback(BenchCallback(tref, log, log_path, model, tok, bad,
                                       args.bench_at, args.bench_n))

    # step0 bench from the stage-1 frozen generations (byte-identical data)
    if not any(b["tag"] == "step0" for b in log["bench"]):
        first8 = bad[: args.bench_n]
        rows0 = [{"id": pid, "loop_share": round(gens[pid]["loop_share"], 3),
                  "n_tok": len(gens[pid]["gen_ids"])} for pid, _ in first8]
        rho0 = sum(1 for r in rows0 if r["loop_share"] >= L.LOOP_THRESHOLD) / len(rows0)
        log["bench"].append({"tag": "step0", "rho_mini": rho0,
                             "mean_ls": round(float(np.mean(
                                 [r["loop_share"] for r in rows0])), 3),
                             "gens": rows0, "source": "stage-1 frozen gens"})
        L.dump_json(log, log_path)

    ckpt = out_dir / "trainer_out"
    resume_from = None
    if args.resume:
        import glob
        cks = sorted(glob.glob(str(ckpt / "checkpoint-*")))
        if cks:
            resume_from = cks[-1]
            print(f"[pen] resuming from {resume_from}", flush=True)

    trainer.train(resume_from_checkpoint=resume_from)
    trainer.save_model(str(out_dir / "adapter"))
    # ensure the final log is flushed
    L.dump_json(log, log_path)
    print(f"[pen] done | adapter -> {out_dir/'adapter'} | RSS {L.rss_gb():.1f}GB",
          flush=True)


if __name__ == "__main__":
    main()
