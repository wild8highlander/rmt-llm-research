"""RCDA formula-as-loss training for OCRia-nano-1 (two arms).

The parent program's lever: bake the RMT formula into gradient training.

    L = pen_norm + beta * relu(mse - cap)

with pen = (v^T h / ||v||)^2 evaluated on the writer-tensor outputs, per-prompt
normalised by its init value, plus a HARD abilities anchor on the trailing
logits against the frozen base (MNN RCDA-5 protocol: Adam, 150 steps,
batch 1, mini-benchmark at 0/75/150, anchor_scope 'good' - the anchor applies
to ability data only, the hallucination-inducing crops stay free for the
penalty).

This kit runs the same lever on the OCR recogniser and extends it with the
parent repository's OTHER formulas in the rcda_x arm:

  classic : pen at the deep readout (mix2) on emitted frames of the CURRENT
            model's own decoding (online self-labelling, refresh every 32
            steps), trainable = mix2.fc2.
  rcda_x  : pen at all three mix writers, trainable = mix{0,1,2}.fc2, PLUS
            - MP bulk leash on the edit spectrum (Steps 4-5),
            - entropy-response band on clean logits (Step 11),
            - single-spike inverse-BBP projection of the edit spectrum
              every 25 steps (Step 14): the intended top spike is kept,
              any unstructured extra spikes are shrunk back into the bulk.

Artifacts (canonical locations):
  Results/{arm}_training_log.json   steps + pen0 + bench + args
  Results/{arm}_trained_state.pt    full edited tensors (fp32, for eval)
  Results/{arm}_deltas.npz          fp16 deltas
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from marchenko_pastur import fit_sigma2
from rmt_fp.mp import mp_edges
from ocria_common import (KIT_DIR, RESULTS_DIR, ctc_decode, dump_json, load_batch,
                          load_image, loop_stats, seed_all)
from rebuild_latin import LatinRecognizer, extract_latin_tensors
from rmt_fp.spiked import bbp_threshold

VARIANTS = {
    "classic": {"pen_stages": ("head",), "train": ("mix2",),
                "beta": 10.0, "cap": 1.0, "extended": False},
    "rcda_x": {"pen_stages": ("head", "mix2", "mix1", "mix0"),
               "train": ("mix0", "mix1", "mix2"),
               "beta": 10.0, "cap": 1.0, "extended": True},
}

# stage -> (module attr path of the trainable conv, pen readout stage)
TRAINABLE = {
    "mix0": ("mix_fc2", 0, "mix0"),
    "mix1": ("mix_fc2", 1, "mix1"),
    "mix2": ("mix_fc2", 2, "mix2"),
}

BAND = 0.25       # entropy band width (nats) around the base per-crop entropy
LEASH_HEADROOM = 4.0  # lambda_max( edit ) <= 4 * lambda_+( edit bulk )
PROJ_EVERY = 25   # inverse-BBP projection cadence (steps)
PROJ_THRESH = 2.0     # shrink eigenvalues beyond 2 * lambda_+ ...
PROJ_KEEP = 2         # ... but always keep the top-2 spikes (intended edits)
REFRESH_EVERY = 32    # online self-labelling cadence (the parent's refresh_every)


def torch_decode(model, crop: np.ndarray):
    with torch.no_grad():
        logits = model(torch.tensor(crop[None, None]))
        probs = F.softmax(logits, dim=1)[0]
        conf, ids = probs.max(dim=0)
    return ids.numpy(), conf.numpy()


def mini_bench(model, bad_crops: list[np.ndarray], n: int = 12) -> dict:
    texts = [ctc_decode(torch_decode(model, c)[0]) for c in bad_crops[:n]]
    ph = sum(1 for t in texts if t)
    rho = sum(1 for t in texts if loop_stats(t)["loop_share"] >= 0.5)
    return {"phantom_rate": round(ph / len(texts), 3),
            "rho_loop": round(rho / len(texts), 3),
            "mean_chars": round(float(np.mean([len(t) for t in texts])), 2)}


def edit_spectrum_stats(dW: np.ndarray) -> dict:
    m, n = max(dW.shape), min(dW.shape)
    Wsq = dW @ dW.T if dW.shape[0] <= dW.shape[1] else dW.T @ dW
    lam = np.linalg.eigvalsh(Wsq / max(dW.shape))
    lam = np.clip(lam, 0, None)
    sigma2 = fit_sigma2(lam, trim_hi=0.05)
    if sigma2 <= 0:
        return {"lambda_max": float(lam.max()), "lambda_plus": 0.0}
    _, lp = mp_edges(min(m, n) / max(m, n), sigma2)
    return {"lambda_max": float(lam.max()), "lambda_plus": float(lp)}


def single_spike_project_(dW: np.ndarray) -> int:
    """Step-14 style projection: keep the top spike (the intended edit),
    shrink every OTHER eigenvalue beyond 1.1*lambda+ back to the bulk edge.
    Returns the number of shrunk spikes."""
    m, n = dW.shape
    sq = dW @ dW.T if m <= n else dW.T @ dW
    lam, U = np.linalg.eigh(sq / max(m, n))
    order = np.argsort(lam)[::-1]
    lam = lam[order]
    U = U[:, order]
    sigma2 = fit_sigma2(np.clip(lam, 0, None), trim_hi=0.05)
    if sigma2 <= 0:
        return 0
    _, lp = mp_edges(min(m, n) / max(m, n), sigma2)
    shrunk = 0
    lam_new = lam.copy()
    for i in range(PROJ_KEEP, len(lam)):  # keep the top-K intended spikes
        if lam[i] > PROJ_THRESH * lp:
            lam_new[i] = PROJ_THRESH * lp
            shrunk += 1
    if shrunk == 0:
        return 0
    lam_new = np.clip(lam_new, 0, None)
    # reconstruct: scale each left-singular direction by sqrt(lambda'/lambda)
    # (U are the left singular vectors of dW because sq = dW dW^T / max(m,n))
    scale = np.sqrt(np.divide(lam_new, np.maximum(lam, 1e-18),
                              out=np.zeros_like(lam), where=lam > 0))
    dW_new = (U * scale) @ (U.T @ dW)
    dW[:, :] = dW_new
    return shrunk


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS), required=True)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--cap", type=float, default=None)
    ap.add_argument("--bench_at", type=int, nargs="*", default=[75, 150])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    seed_all(args.seed)
    cfg = dict(VARIANTS[args.variant])
    if args.beta is not None:
        cfg["beta"] = args.beta
    if args.cap is not None:
        cfg["cap"] = args.cap

    # ------------------------------------------------------------------
    # model + frozen base
    # ------------------------------------------------------------------
    model = LatinRecognizer(extract_latin_tensors())
    base = LatinRecognizer(extract_latin_tensors())
    model.eval(); base.eval()
    for p in base.parameters():
        p.requires_grad_(False)

    # probe vectors
    with np.load(RESULTS_DIR / "ocria_vectors.npz") as z:
        vecs = {k[2:]: torch.tensor(z[k], dtype=torch.float32) for k in z.files}
    for s in cfg["pen_stages"]:
        if s not in vecs:
            raise SystemExit(f"missing probe vector for {s}")

    # trainable tensors = the mix fc2 conv weights of the pen stages
    trainable_params, trainable_names = [], []
    for stage in cfg["train"]:
        attr, idx, _ = TRAINABLE[stage]
        w = getattr(model, attr)[idx].weight
        w.requires_grad_(True)
        trainable_params.append(w)
        trainable_names.append(f"{attr}.{idx}.weight")
    n_train = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[train] variant={args.variant} trainable={trainable_names} "
          f"({n_train:,} params, {100 * n_train / n_total:.2f}%) "
          f"beta={cfg['beta']} cap={cfg['cap']}")

    # ------------------------------------------------------------------
    # data: bad = all non-text crops (pen freed), good/anchor = clean + control
    # ------------------------------------------------------------------
    bad_batches = ("structured", "noise", "blank")
    bad_recs, bad_crops = [], []
    for b in bad_batches:
        for r in load_batch(b):
            bad_recs.append(r)
            bad_crops.append(np.asarray(load_image(KIT_DIR / r["file"]), np.float32))
    good_recs = load_batch("clean") + load_batch("control")
    good_crops = [np.asarray(load_image(KIT_DIR / r["file"]), np.float32)
                  for r in good_recs]

    # pen0 per bad crop at init (weights == base).
    #
    # Penalty family (Step 10 + Step 14 operationalised as a loss):
    #   head  : edge energy  mean_frames sum_c relu(l_c - l_0)^2 / K - the
    #           frame's logit spectrum vs the ABSTENTION EDGE (the blank
    #           logit): phantom classes are the supercritical spike above
    #           the edge, the penalty is the squared excess - the quantity
    #           the BBP clamp bounds.  Forcing the excess to zero forces
    #           blank to win the argmax (abstention), frame by frame.
    #   mix   : one-sided v-clamp  relu(v^T h - c_s)^2 - pull the writer
    #           state below the ABSTAINED reference level c_s (the mean
    #           v-projection over the base model's confident-blank frames,
    #           computed at init).  The symmetric square is NOT used: it
    #           pushes abstained frames toward the hyperplane from both
    #           sides (control run: phantom 0.42 -> 1.0, see AUDIT.md).
    def pen_frames(h, sel, v, c_ref):
        if c_ref is None:   # head: edge energy vs the blank logit
            lg = h[:, sel]                       # [K, n_frames]
            edge = lg[0:1, :]                    # blank logit per frame
            return (F.relu(lg - edge) ** 2).sum(0).mean() / lg.shape[0]
        proj = h.T[sel] @ v
        return (F.relu(proj - c_ref) ** 2).mean()

    # abstained reference per mix stage (one-sided clamp target)
    c_refs = {}
    if len(cfg["pen_stages"]) > 1 or "mix" in cfg["pen_stages"][0]:
        abst_frames = {s: [] for s in cfg["pen_stages"] if s != "head"}
        for i, c in enumerate(bad_crops):
            ids, conf = torch_decode(base, c)
            sel_abs = (ids == 0) & (conf >= 0.5)
            if not sel_abs.any():
                continue
            with torch.no_grad():
                base(torch.tensor(c[None, None]))
                for s in abst_frames:
                    h = base.stage_maps[s]
                    if h.dim() == 4:
                        h = h.squeeze(2)
                    h = h[0]
                    v = vecs[s] / (vecs[s].norm() + 1e-12)
                    abst_frames[s].append((h.T[sel_abs] @ v).detach())
        for s, frames in abst_frames.items():
            if frames:
                c_refs[s] = float(torch.cat(frames).mean())

    pen0 = {}
    for i, c in enumerate(bad_crops):
        ids, _ = torch_decode(base, c)
        if not (ids != 0).any():
            pen0[str(i)] = None
            continue
        with torch.no_grad():
            base(torch.tensor(c[None, None]))
            terms = []
            for s in cfg["pen_stages"]:
                h = base.stage_maps[s]
                if h.dim() == 4:
                    h = h.squeeze(2)
                h = h[0]  # [C,T]
                v = vecs[s] / (vecs[s].norm() + 1e-12) if s in vecs else None
                sel = (ids != 0)
                terms.append(pen_frames(h, sel, v,
                                        c_refs.get(s) if s != "head" else None))
            pen0[str(i)] = float(torch.stack(terms).mean())
    active = [i for i in range(len(bad_crops)) if pen0[str(i)] is not None]
    pen_ref = float(np.mean([pen0[str(i)] for i in active])) if active else 0.0
    print(f"[train] pen0 spread: {min(pen0[str(i)] for i in active):.4f} .. "
          f"{max(pen0[str(i)] for i in active):.4f} over {len(active)} phantom crops, "
          f"pen_ref={pen_ref:.4f}")

    # frozen base trailing logits on the anchor data (the hard anchor reference)
    def base_logits_of(crop: np.ndarray) -> torch.Tensor:
        with torch.no_grad():
            return base(torch.tensor(crop[None, None]))[0].float()   # [792, T]

    anchor_len = 32
    base_logits = {}
    for j, c in enumerate(good_crops):
        base_logits[j] = base_logits_of(c)[:, -anchor_len:]

    # entropy-band reference (extended arm): per-crop mean entropy at base
    ent_ref = {}
    if cfg["extended"]:
        for j, c in enumerate(good_crops[:32]):
            lg = base_logits_of(c)
            p = F.softmax(lg, dim=0)
            ent_ref[j] = float(-(p * (p + 1e-12).log()).sum(dim=0).mean())

    if args.dry_run:
        print("[train] dry run ok - exiting before the training loop")
        return

    # ------------------------------------------------------------------
    # training loop
    # ------------------------------------------------------------------
    opt = torch.optim.Adam(trainable_params, lr=args.lr)
    steps_log, bench_log, proj_log = [], [], []

    mb0 = mini_bench(model, bad_crops)
    bench_log.append({"step": 0, **mb0})
    print(f"[train] bench@0: {mb0}")

    # frozen emission masks (the parent kit's frozen prefixes + refresh_every):
    # the pen positions are fixed between refreshes so the target is stable,
    # then re-labelled from the CURRENT model every REFRESH_EVERY steps
    emit_mask = {}
    for i in active:
        ids, _ = torch_decode(model, bad_crops[i])
        emit_mask[i] = (ids != 0)

    def refresh_masks(step: int) -> None:
        n_still = 0
        for i in active:
            ids, _ = torch_decode(model, bad_crops[i])
            emit_mask[i] = (ids != 0)
            n_still += int(emit_mask[i].any())
        print(f"[train] step {step}: emission masks refreshed "
              f"({n_still}/{len(active)} crops still emit)")

    # cycle: phantom crops x3 then 40 anchor crops (the parent's batch-1 cycle)
    order = [("bad", i) for i in active] * 3 + [("good", j) for j in range(40)]

    def pen_of(crop_id: int):
        """live pen tensor on the FROZEN emission frames of this crop."""
        c = bad_crops[crop_id]
        logits = model(torch.tensor(c[None, None]))          # [1,792,T] w/ grad
        sel = emit_mask[crop_id]
        if not sel.any():
            return None, 0.0, 0
        terms = []
        n_frames = int(sel.sum())
        for s in cfg["pen_stages"]:
            h = model.stage_maps[s]
            if h.dim() == 4:
                h = h.squeeze(2)
            h = h[0]  # [C,T] with grad
            v = vecs[s] / (vecs[s].norm() + 1e-12) if s in vecs else None
            terms.append(pen_frames(h, sel, v,
                                    c_refs.get(s) if s != "head" else None))
        pen = torch.stack(terms).mean()
        return pen, float(pen.detach()), n_frames

    t0 = time.time()
    for step in range(1, args.steps + 1):
        kind, idx = order[(step - 1) % len(order)]
        ts = time.time()
        mp_leash = 0.0
        ent_pen = 0.0
        if kind == "bad":
            pen, pen_raw, n_frames = pen_of(idx)
            if pen is None:
                pen_norm, pen_raw = 0.0, 0.0
                loss = torch.zeros(())
                mse = torch.zeros(())
            else:
                denom = pen0[str(idx)] if pen0[str(idx)] else pen_ref
                pen_norm = pen_raw / (denom + 1e-12)
                loss = pen / (denom + 1e-12)
                mse = torch.zeros(())
        else:
            logits = model(torch.tensor(good_crops[idx][None, None]))[0]  # [792,T]
            pen_norm, pen_raw, n_frames = 0.0, 0.0, 0
            mse = F.mse_loss(logits[:, -anchor_len:], base_logits[idx])
            loss = cfg["beta"] * torch.relu(mse - cfg["cap"])
            if cfg["extended"] and idx in ent_ref:
                p = F.softmax(logits, dim=0)
                H = float(-(p * (p + 1e-12).log()).sum(dim=0).mean())
                ent_pen = float(torch.relu(torch.tensor(abs(H - ent_ref[idx]) - BAND) ** 2))
                loss = loss + ent_pen
        # extended arm: MP bulk leash on the edit spectrum (reported; the
        # spectrum itself is kept inside the bulk by the projection below)
        if cfg["extended"]:
            ratios = []
            for name, w in zip(trainable_names, trainable_params):
                base_w = dict(base.named_parameters())[name].detach().numpy()
                dW = (w.detach().numpy() - base_w).reshape(w.shape[0], -1)
                st = edit_spectrum_stats(dW)
                if st["lambda_plus"] > 0:
                    ratios.append(st["lambda_max"] /
                                  (LEASH_HEADROOM * st["lambda_plus"]) - 1.0)
            mp_leash = float(np.mean([max(0.0, r) for r in ratios])) if ratios else 0.0

        opt.zero_grad(set_to_none=True)
        if loss.requires_grad:
            loss.backward()
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.clip)
            opt.step()

        # online self-labelling: re-draw the emission masks from the CURRENT
        # model so the penalty always attacks the frames it would emit now
        if step % REFRESH_EVERY == 0 and step < args.steps:
            refresh_masks(step)

        steps_log.append({"step": step, "kind": kind,
                          "mse": round(float(mse.detach()), 6),
                          "pen_raw": round(pen_raw, 4),
                          "pen_norm": round(pen_norm, 4),
                          "mp_leash": round(mp_leash, 4),
                          "ent_band": round(ent_pen, 4),
                          "n_frames": n_frames,
                          "loss": round(float(loss.detach()), 4),
                          "s": round(time.time() - ts, 2)})
        if step in args.bench_at:
            mb = mini_bench(model, bad_crops)
            bench_log.append({"step": step, **mb})
            print(f"[train] bench@{step}: {mb}")
        if step % 10 == 0:
            print(f"[train] step {step}: loss={float(loss.detach()):.4f} pen_norm={pen_norm:.3f} "
                  f"mse={float(mse.detach()):.5f} ({time.time() - t0:.0f}s)")

        # extended arm: single-spike inverse-BBP projection of the edits
        if cfg["extended"] and step % PROJ_EVERY == 0 and step < args.steps:
            n_shrunk = 0
            for name, w in zip(trainable_names, trainable_params):
                base_w = dict(base.named_parameters())[name].detach().numpy()
                dW = (w.detach().numpy() - base_w).reshape(w.shape[0], -1)
                dWc = np.array(dW, copy=True)
                n_shrunk += single_spike_project_(dWc)
                with torch.no_grad():
                    w.copy_(torch.tensor(base_w.reshape(w.shape) + dWc.reshape(w.shape),
                                         dtype=w.dtype))
            proj_log.append({"step": step, "shrunk_spikes": n_shrunk})
            if n_shrunk:
                print(f"[train] step {step}: inverse-BBP projection shrunk {n_shrunk} extra spikes")

    # ------------------------------------------------------------------
    # artifacts
    # ------------------------------------------------------------------
    deltas, state, rel_frob = {}, {}, {}
    base_sd = dict(base.named_parameters())
    with torch.no_grad():
        for name, w in zip(trainable_names, trainable_params):
            d = (w.detach() - base_sd[name].detach()).numpy()
            deltas[name] = d.astype(np.float16)
            state[name] = w.detach().clone()
            rel_frob[name] = float(np.linalg.norm(d.astype(np.float32)) /
                                   (np.linalg.norm(base_sd[name].detach().numpy()) + 1e-12))
    log = {
        "variant": args.variant,
        "model": "llamaindex/OCRia-nano-1 rec_latin (fp32 rebuild)",
        "args": {"steps": args.steps, "lr": args.lr, "mse_cap": cfg["cap"],
                 "beta": cfg["beta"], "bench_at": args.bench_at,
                 "grad_clip": args.clip, "anchor_scope": "good",
                 "refresh_every": REFRESH_EVERY, "extended": cfg["extended"],
                 "trainable": trainable_names, "trainable_params": n_train,
                 "trainable_share": round(n_train / n_total, 5),
                 "pen_points": list(cfg["pen_stages"]), "seed": args.seed},
        "writers": list(cfg["train"]),
        "pen0": pen0, "pen_ref": pen_ref,
        "steps": steps_log, "bench": bench_log,
        "projection_log": proj_log,
        "deltas_rel_frobenius": {k: round(v, 6) for k, v in rel_frob.items()},
        "wall_time_s": round(time.time() - t0, 1),
    }
    dump_json(log, RESULTS_DIR / f"{args.variant}_training_log.json")
    torch.save(state, RESULTS_DIR / f"{args.variant}_trained_state.pt")
    np.savez_compressed(RESULTS_DIR / f"{args.variant}_deltas.npz", **deltas)
    final = [b for b in bench_log if b["step"] == args.steps]
    print(f"[train] done in {log['wall_time_s']}s: phantom_rate "
          f"{bench_log[0]['phantom_rate']} -> {final[0]['phantom_rate'] if final else 'n/a'}, "
          f"pen_norm(last) {steps_log[-1]['pen_norm']:.3f}")
    print(f"[train] wrote {args.variant}_training_log.json / _trained_state.pt / _deltas.npz")


if __name__ == "__main__":
    main()
