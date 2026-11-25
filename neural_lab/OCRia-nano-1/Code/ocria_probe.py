"""The extended RCDA probe for OCRia-nano-1 (six hallucination families).

The parent program probes ONE defect (the repetition loop) with a five-step
pipeline: greedy continuations -> per-layer Delta h between looped and clean
tokens of the SAME generations -> z_diff writer localisation -> BBP matrix
scans -> hallucination-direction vectors.  This probe runs the same pipeline
over an OCR recogniser's *feature states* and extends it in three directions:

  1. six families instead of one - phantom text, CTC collapse loops, misreads,
     script routing, flip blindness, over-confidence - each with its own
     pre-registered metric;
  2. state labelling from the deployed int8 graph: emitted frames (CTC argmax
     != 0) vs abstained frames (argmax == 0, confident) of the SAME
     trajectories, so the contrast strips content and keeps the defect;
  3. a confidence-vs-geometry AUC contest (the parent program's "Utility
     Trap" experiment, ported to OCR): is the model's confidence or its
     internal geometry the better predictor of its own hallucinations?

Outputs (canonical locations):
  Results/probe_report.json     - families, z_diff, scans, writers, AUCs
  Results/ocria_vectors.npz     - v_stage unit vectors (training input)
  Results/probe_gens.jsonl      - every crop's base decoding + labels
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ocria_common import (BATCHES, DATA_DIR, KIT_DIR, NONTEXT_BATCHES, RESULTS_DIR,
                          cer, ctc_decode, dump_json, load_batch, load_image,
                          loop_stats, ort_session, rho_of_texts, run_recognizer,
                          run_router, seed_all)
from marchenko_pastur import spike_scan
from rebuild_latin import LatinRecognizer, extract_latin_tensors

STAGES = ("stem", "body0", "body1", "body2", "body3", "body4",
          "body5", "body6", "body7", "body8", "mix0", "mix1", "mix2", "head")
# the contrast is defined on the CTC timeline: stages whose height collapsed
# to 1 (body8 onward) have exactly one feature state per decoded frame; the
# earlier stages live on coarser timelines and are reported as pre-timeline
CONTRAST_STAGES = ("body8", "mix0", "mix1", "mix2", "head")
# trainable-candidate tensors: the 1x1 pointwise convs that WRITE the final
# state (the OCR analogue of the LM's o_proj / down_proj writer tensors)
WRITER_TENSORS = {
    "body8": "body8_pw", "mix0": "mix0_fc2", "mix1": "mix1_fc2", "mix2": "mix2_fc2",
}
CONF_EMIT = 0.5   # a frame counts as emitted/abstained only when confident


def state_matrix(model, crops: list[np.ndarray], stage: str) -> np.ndarray:
    """[n_frames_total, d] feature states at `stage` for a list of crops."""
    rows = []
    with torch.no_grad():
        for c in crops:
            x = torch.tensor(c[None, None])
            model(x)
            h = model.stage_maps[stage]  # [N,C,H,T] or [N,C,T]
            if h.dim() == 4:
                h = h.squeeze(2)         # [N,C,T]
            rows.append(h[0].numpy().T)  # [T, C]
    return np.concatenate(rows, 0)


def z_diff(proj_loop: np.ndarray, proj_clean: np.ndarray) -> float:
    """Separation of the two projection samples in pooled-sigma units
    (the parent probe's Hotelling-flavoured writer statistic)."""
    if len(proj_loop) < 2 or len(proj_clean) < 2:
        return 0.0
    m1, m0 = proj_loop.mean(), proj_clean.mean()
    s1, s0 = proj_loop.std() + 1e-6, proj_clean.std() + 1e-6
    n1, n0 = len(proj_loop), len(proj_clean)
    se = np.sqrt(s1**2 / n1 + s0**2 / n0)
    return float((m1 - m0) / se)


def auc_score(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank AUC that pos scores higher than neg (ties = 0.5)."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks over ties
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    seed_all(args.seed)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out_dir) if args.out_dir else RESULTS_DIR

    # ------------------------------------------------------------------
    # 1. base behaviour on every corpus (deployed int8 graph, ORT)
    # ------------------------------------------------------------------
    sess = ort_session("rec_latin.onnx")
    charset = None
    gens = {}
    for batch in BATCHES:
        recs = load_batch(batch)
        rows = []
        for r in recs:
            ids, conf = run_recognizer(sess, load_image(KIT_DIR / r["file"]))
            text = ctc_decode(ids)
            st = loop_stats(text)
            row = {"batch": batch, "id": r["id"], "text": text,
                   "n_chars": len(text),
                   "mean_conf": round(float(conf[ids != 0].mean()) if (ids != 0).any()
                                      else float(conf.mean()), 4),
                   "loop_share": round(st["loop_share"], 4),
                   "gt": r["text"]}
            if r["text"]:
                row["cer"] = round(cer(text, r["text"]), 4)
                row["correct"] = bool(text == r["text"])
            rows.append(row)
        gens[batch] = rows

    # family metrics at base
    fam = {}
    for batch in NONTEXT_BATCHES:
        fam.setdefault("F1_phantom", []).extend(gens[batch])
    phantom_rows = [r for r in fam["F1_phantom"] if r["n_chars"] > 0]
    fam["F2_loops"] = phantom_rows
    fam["F3_misreads"] = gens["degraded"]

    n_nontext = len(fam["F1_phantom"])
    base = {
        "phantom_rate": round(len(phantom_rows) / n_nontext, 4),
        "phantom_char_rate": round(float(np.mean([r["n_chars"] for r in fam["F1_phantom"]])), 4),
        "phantom_mean_conf": round(float(np.mean([r["mean_conf"] for r in phantom_rows])) if phantom_rows else 0.0, 4),
        "loop_metrics": rho_of_texts([r["text"] for r in phantom_rows]),
        "cer_clean": round(float(np.mean([r["cer"] for r in gens["clean"]])), 4),
        "cer_degraded": round(float(np.mean([r["cer"] for r in gens["degraded"]])), 4),
        "cer_control": round(float(np.mean([r["cer"] for r in gens["control"]])), 4),
        "specificity_blank": round(sum(1 for r in gens["blank"] if r["n_chars"] == 0) / len(gens["blank"]), 4),
    }
    print(f"[probe] base phantom_rate={base['phantom_rate']:.2f} "
          f"cer_clean={base['cer_clean']:.3f} cer_degraded={base['cer_degraded']:.3f}")

    # ------------------------------------------------------------------
    # 2. state labelling on the fp32 rebuild (same weights, gradients ready)
    # ------------------------------------------------------------------
    model = LatinRecognizer(extract_latin_tensors())
    model.eval()

    def crops_of(batch: str) -> list[np.ndarray]:
        return [load_image(KIT_DIR / r["file"]) for r in load_batch(batch)]

    bad_crops = crops_of("structured") + crops_of("noise") + crops_of("blank")
    bad_meta = (load_batch("structured") + load_batch("noise") + load_batch("blank"))
    clean_crops = crops_of("clean")
    clean_recs = load_batch("clean")

    # per-crop torch decode (must mirror the ORT labels)
    with torch.no_grad():
        torch_texts = []
        for c in bad_crops + clean_crops:
            logits = model(torch.tensor(c[None, None]))
            probs = F.softmax(logits, dim=1)[0]      # [792, T]
            conf, ids = probs.max(dim=0)             # [T]
            torch_texts.append((ids.numpy(), conf.numpy()))

    emit_states = {s: [] for s in CONTRAST_STAGES}   # phantom-emitted frames
    abst_states = {s: [] for s in CONTRAST_STAGES}   # confident-blank frames
    clean_states = {s: [] for s in CONTRAST_STAGES}  # emitted frames, clean

    n_phantom_states = 0
    for k, (ids, conf) in enumerate(torch_texts[:len(bad_crops)]):
        is_phantom = bool((ids != 0).any())
        with torch.no_grad():
            model(torch.tensor(bad_crops[k][None, None]))
            for s in CONTRAST_STAGES:
                h = model.stage_maps[s]
                if h.dim() == 4:
                    h = h.squeeze(2)
                if s == "head":
                    h = h[0].permute(1, 0).numpy()  # [T, 792] logits
                else:
                    h = h[0].numpy().T              # [T, 192]
                if h.shape[0] != len(ids):
                    continue  # pre-timeline stage (defensive; none expected)
                if is_phantom:
                    sel = (ids != 0) & (conf >= CONF_EMIT)
                    if sel.any():
                        emit_states[s].append(h[sel])
                        n_phantom_states += int(sel.sum())
                sel_abs = (ids == 0) & (conf >= CONF_EMIT)
                if sel_abs.any():
                    abst_states[s].append(h[sel_abs])
    for k, (ids, conf) in enumerate(torch_texts[len(bad_crops):]):
        with torch.no_grad():
            model(torch.tensor(clean_crops[k][None, None]))
            for s in CONTRAST_STAGES:
                h = model.stage_maps[s]
                if h.dim() == 4:
                    h = h.squeeze(2)
                if s == "head":
                    h = h[0].permute(1, 0).numpy()
                else:
                    h = h[0].numpy().T
                if h.shape[0] != len(ids):
                    continue
                sel = (ids != 0) & (conf >= CONF_EMIT)
                if sel.any():
                    clean_states[s].append(h[sel])

    print(f"[probe] state frames: emitted={n_phantom_states} "
          f"abst={sum(len(a) for v in abst_states.values() for a in v)}")

    # ------------------------------------------------------------------
    # 3. per-stage hallucination direction v, z_diff, geometry AUC
    # ------------------------------------------------------------------
    vectors = {}
    stages_report = {}
    for s in STAGES:
        if s not in CONTRAST_STAGES:
            stages_report[s] = {"status": "pre_timeline",
                                "note": "height>1: no per-frame state on the CTC timeline"}
            continue
        E = np.concatenate(emit_states[s], 0) if emit_states[s] else None
        A = np.concatenate(abst_states[s], 0) if abst_states[s] else None
        C = np.concatenate(clean_states[s], 0) if clean_states[s] else None
        if E is None or A is None:
            stages_report[s] = {"status": "insufficient_states"}
            continue
        mu = E.mean(0) - A.mean(0)
        n = np.linalg.norm(mu)
        v = (mu / n) if n > 1e-8 else np.zeros_like(mu)
        vectors[s] = v.astype(np.float32)
        pE, pA = E @ v, A @ v
        zd = z_diff(pE, pA)
        auc_geom = auc_score(pE, pA)
        stages_report[s] = {
            "n_emit": int(len(E)), "n_abst": int(len(A)),
            "z_diff": round(zd, 3), "auc_geometry": round(auc_geom, 4),
            "delta_mean": round(float(E.mean() - A.mean()), 4),
        }
        if C is not None:
            stages_report[s]["z_vs_clean"] = round(z_diff(pE, C @ v), 3)

    # confidence vs geometry contest (F6) on the phantom-vs-abstained states:
    # the conf AUC uses frame confidence, the geometry AUC uses |projection on v|
    conf_frames, geom_frames, lab = [], [], []
    for k, (ids, conf) in enumerate(torch_texts[:len(bad_crops)]):
        if not (ids != 0).any():
            continue
        with torch.no_grad():
            model(torch.tensor(bad_crops[k][None, None]))
            h = model.stage_maps["mix2"]
            if h.dim() == 4:
                h = h.squeeze(2)
            h = h[0].numpy().T
        v = vectors.get("mix2")
        if v is None or v.size != h.shape[1]:
            continue
        for t in range(len(ids)):
            if ids[t] != 0 and conf[t] >= CONF_EMIT:
                lab.append(1); conf_frames.append(float(conf[t])); geom_frames.append(float(h[t] @ v))
            elif ids[t] == 0 and conf[t] >= CONF_EMIT:
                lab.append(0); conf_frames.append(float(conf[t])); geom_frames.append(float(h[t] @ v))
    lab = np.array(lab); conf_frames = np.array(conf_frames); geom_frames = np.array(geom_frames)
    auc_conf_frames = auc_score(conf_frames[lab == 1], conf_frames[lab == 0])
    auc_geom_frames = auc_score(geom_frames[lab == 1], geom_frames[lab == 0])
    utility_trap = {
        "n_frames": int(len(lab)),
        "auc_confidence": round(auc_conf_frames, 4),
        "auc_geometry_minus_v": round(auc_geom_frames, 4),
        "verdict": ("geometry wins: confidence is the WORSE predictor of the "
                    "recogniser's own phantom frames - the parent program's "
                    "Utility Trap replicated on OCR"
                    ) if auc_geom_frames > auc_conf_frames else "confidence wins",
    }
    print(f"[probe] Utility Trap: AUC_conf={auc_conf_frames:.3f} "
          f"AUC_geom={auc_geom_frames:.3f} -> {utility_trap['verdict'][:30]}")

    # ------------------------------------------------------------------
    # 4. BBP matrix scans: writer candidates vs the hallucination direction
    # ------------------------------------------------------------------
    tensors = extract_latin_tensors()

    def tensor_weight(stage: str) -> tuple[str, np.ndarray]:
        frag = {"body8": "body.8/act_1", "mix0": "mix.0/fc2",
                "mix1": "mix.1/fc2", "mix2": "mix.2/fc2",
                "head": "head/Conv"}[stage]
        e = next(t for t in tensors if frag in t["node_out"])
        W = e["w"]
        while W.ndim > 2:
            W = W[..., 0]
        return frag.split("/")[-1], W

    scans = {}
    for stage in list(WRITER_TENSORS) + ["head"]:
        if stage not in vectors:
            continue
        v_out = vectors.get(stage)  # output-side direction of this tensor
        if v_out is None:
            continue
        key, W = tensor_weight(stage)
        scan = spike_scan(W.astype(np.float64), v_out.astype(np.float64))
        scans[key] = {k2: (round(v2, 4) if isinstance(v2, float) else v2)
                      for k2, v2 in scan.items()}
        scans[key]["stage"] = stage
    supercritical = [k for k, s in scans.items() if s["supercritical"]]
    print(f"[probe] matrix scans: {len(scans)} tensors, "
          f"supercritical={supercritical}")

    # ------------------------------------------------------------------
    # 5. writer selection: top |z_diff| timeline stages among the writers
    # ------------------------------------------------------------------
    ranked = sorted([s for s in CONTRAST_STAGES[:-1] if s in stages_report
                     and "z_diff" in stages_report[s]],
                    key=lambda s: -abs(stages_report[s]["z_diff"]))
    writers = ranked[:3]
    print(f"[probe] writer selection: {writers} (z_diff "
          f"{[stages_report[s]['z_diff'] for s in writers]})")

    # ------------------------------------------------------------------
    # 6. router + detector probes (F4 / F5 and the detector side of F1)
    # ------------------------------------------------------------------
    router = probe_router()
    detector = probe_detector()

    # ------------------------------------------------------------------
    # artifacts
    # ------------------------------------------------------------------
    report = {
        "model": "llamaindex/OCRia-nano-1 (rec_latin.onnx int8 -> fp32 rebuild)",
        "protocol": {
            "n_nontext": n_nontext, "n_clean": len(clean_crops),
            "conf_emit": CONF_EMIT, "loop_ngram": 4, "loop_threshold": 0.5,
            "probe_seed": args.seed,
        },
        "base": base,
        "stages": stages_report,
        "writers": writers,
        "matrix_scans": scans,
        "supercritical": supercritical,
        "utility_trap": utility_trap,
        "router": router,
        "detector": detector,
    }
    dump_json(report, out_dir / "probe_report.json")
    np.savez_compressed(out_dir / "ocria_vectors.npz",
                        **{f"v_{s}": vectors[s] for s in vectors})
    rows = [dict(r, batch=b) for b, rs in gens.items() for r in rs]
    with open(out_dir / "probe_gens.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[probe] wrote probe_report.json, ocria_vectors.npz "
          f"({len(vectors)} stages), probe_gens.jsonl ({len(rows)} rows)")


def probe_router() -> dict:
    """F4 script-routing + F5 flip-blindness on the deployed router."""
    sess = ort_session("router.onnx")
    recs = load_batch("router")
    group_names = ("latin", "cjk", "rtl", "world")
    route_rows, flip_rows, nontext_rows = [], [], []
    for r in recs:
        g, fl = run_router(sess, load_image(KIT_DIR / r["file"]))
        gi, fi = int(np.argmax(g)), int(np.argmax(fl))
        if r["kind"].startswith("text_") and not r.get("flipped"):
            want = {"arabic": "rtl", "cyr": "latin"}.get(r["script"], r["script"])
            route_rows.append({"want": want, "got": group_names[gi],
                               "ok": bool(want == group_names[gi])})
        elif r.get("flipped"):
            flip_rows.append({"script": r["script"], "detected": bool(fi == 1)})
        else:
            nontext_rows.append({"routed_to": group_names[gi],
                                 "margin": round(float(g[gi] - np.max(np.delete(g, gi))), 3)})
    route_err = round(1 - float(np.mean([x["ok"] for x in route_rows])), 4) if route_rows else None
    flip_miss = {}
    for sc in ("latin", "cyr", "arabic", "cjk"):
        xs = [x for x in flip_rows if x["script"] == sc]
        if xs:
            flip_miss[sc] = round(1 - float(np.mean([x["detected"] for x in xs])), 4)
    return {
        "group_order": list(group_names),
        "route_error_rate": route_err,
        "n_text": len(route_rows),
        "flip_miss_rate": flip_miss,
        "nontext_route_rate": round(float(np.mean([x["routed_to"] is not None for x in nontext_rows])), 4) if nontext_rows else None,
        "nontext_routed_to": {k: sum(1 for x in nontext_rows if x["routed_to"] == k)
                              for k in group_names},
        "n_nontext": len(nontext_rows),
    }


def probe_detector(n_pages: int = 6, seed: int = 3) -> dict:
    """Detector-side phantom boxes on textless pages (F1 at the det stage)."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    sess = ort_session("det.onnx")
    rng = np.random.default_rng(seed)
    from make_corpora import PATTERNS
    out = {}
    for kind in ("text", "blank", "noise", "structured"):
        boxes, thr_frac = 0, []
        for p in range(n_pages):
            w, h = 640, 480
            img = Image.new("RGB", (w, h), (255, 255, 255))
            d = ImageDraw.Draw(img)
            if kind == "text":
                f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
                for i in range(14):
                    d.text((30, 20 + i * 30), f"synthetic document line {i + 1} "
                           f"with numbers {rng.integers(100, 999)}", font=f, fill=(20, 20, 20))
            elif kind == "blank":
                pass
            elif kind == "noise":
                a = rng.normal(128, 60, (h, w, 3))
                img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
            else:
                pat = list(PATTERNS)[p % len(PATTERNS)]
                crop = np.asarray(PATTERNS[pat](rng, w), np.float32)
                a = np.stack([crop] * 3, -1)
                img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
            a = np.asarray(img, np.float32).transpose(2, 0, 1)[None] / 255.0
            prob = sess.run(None, {"x": np.ascontiguousarray(a.astype(np.float32))})[0][0, 0]
            mask = prob > 0.55
            thr_frac.append(float(mask.mean()))
            # connected-component line count (crude but deterministic)
            from scipy import ndimage
            lbl, n = ndimage.label(mask)
            sizes = ndimage.sum(mask, lbl, range(1, n + 1)) if n else []
            boxes += int(np.sum(np.array(sizes) > 40)) if n else 0
        out[kind] = {"phantom_boxes_per_page": round(boxes / n_pages, 2),
                     "hot_fraction": round(float(np.mean(thr_frac)), 4)}
    out["note"] = ("prob>0.55 blobs larger than 40 px counted as line boxes; "
                   "textless pages should score ~0")
    return out


if __name__ == "__main__":
    main()
