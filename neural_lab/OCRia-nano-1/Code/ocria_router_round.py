"""The router round: F4's missing reject option, added as a 5th head class.

This is the kit's THIRD round and its first **head edit** (AUDIT.md
limitation #3, lifted by user request): the deployed router's group head
has only the 4 script classes (latin/cjk/rtl/world), so 100% of textless
crops are routed INTO a recogniser — the phantom chain of F4 starts at
routing.  Curing it is not a weight edit but a head change: the group head
`group.weight [4,192] / group.bias [4]` grows a 5th row, **reject**.

The trunk (stem + 6 mobile blocks) and the flip head stay bit-frozen; the
pooled 192-d features (mean+max over space, the Concat feeding both Gemms)
are extracted once through onnxruntime and cached — training therefore
operates on exact deployed-graph features, not a rebuild.

Measured lessons of the round's control runs (kept in
router_reject_training_log.json):
  1. the reject row first learned *whiteness*, not textlessness — the fit
     corpus text spans the full 512-px frame while the ability corpora
     (clean/control) are 200-480 px wide and are centred on a white canvas
     to fit the router's fixed input; a reject row trained only on
     full-frame text fired on 47% of the padded ability text.  The shipped
     protocol anchors the round on the kit's ability corpora (the parent's
     anchor scope 'good').
  2. text-heavy passes collapse the cure: with 240 text vs 120 non-text
     steps, Adam's per-sign normalization let the tiny-but-consistent
     text-side gradients drag the reject row below the script logits on
     half of the non-text crops (fit reject 1.0 -> 0.5).  The shipped
     protocol therefore trains VIOLATION-DRIVEN (the kit's online
     self-labelling, ported to a classifier head): every 32 steps the
     violation sets are re-labelled from the current head and only crops
     the head still gets wrong are sampled, 3:1 nt:text, with lr x0.5 at
     steps 320/640; the round stops early when no violations remain.

Two arms, one round protocol (the kit's fair-arm convention):

  reject_row  : train ONLY the new row (192 + 1 params).  The 4 script rows
                stay bit-exact -> script logits parity is exact by
                construction; the edit is a single rank-1 BBP spike by
                design (verified in the spectral section).
  reject_full : train all 5 rows; CE on text (script targets) + non-text
                (reject target) + the hard anchor beta*relu(mse4 - cap)
                of the 4 script logits vs the frozen base (scope 'text',
                the router's ability data).

Round protocol (pre-registered addendum 8 of EXPERIMENT_PROTOCOL.md):
Adam, batch 1, grad-clip 1.0, seed 0, violation-driven sampling (refresh
every 32 steps, violators 3:1 nt:text), up to 960 steps with lr x0.5 at
320/640, early stop when both violation sets empty, mini-benchmark at
0/320/640/960.  Width normalisation: the router is fixed 32x512; the fit
corpus is exactly 512 wide, ability/transfer crops are centred on a white
canvas (pad; faithful — resize degrades even the base router to 0.892
latin acc).

Gates (registered):
  G1 parity  - row arm: script rows bitwise unchanged (script logits exact);
               full arm: zero new routing errors on the fit text crops and
               ability route acc >= base - 0.02, anchor mse <= cap.  Both
               arms: the reject row never wins an ability text crop.
  G2 flip    - flip head untouched -> flip outputs bitwise identical.
  G3 trunk   - every non-head initializer array identical to the base graph.
  G4 cure    - reject rate >= 0.90 on the 40 fit non-text crops AND the
               reject row never wins a fit text crop.

The honest generalisation metric is the TRANSFER reject rate on the three
non-text families the router corpus never contains (structured/noise/blank,
210 crops); ability text is anchored (in-sample) by protocol, like the
recogniser arms' clean/control anchor.

Winner: the gate-passing arm with the higher transfer reject rate; tie goes
to reject_row (minimal intervention).

Artifacts (canonical locations):
  Model/router_reject.onnx           extended graph (group -> 5 classes)
  Model/router_reject.meta.json      group order incl. reject + provenance
  Results/router_reject_training_log.json   both arms' steps/bench/gates
  Results/eval_router_reject.json    the winner's full F4/F5 eval
  Results/eval_router_reject_rows.jsonl     per-crop rows of the winner
  Results/router_reject_deltas.npz   per-arm head deltas (fp32)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from marchenko_pastur import fit_sigma2
from ocria_common import (KIT_DIR, MODEL_DIR, RESULTS_DIR, dump_json,
                          load_batch, load_image, seed_all)
from rmt_fp.mp import mp_edges

ROUTER_GROUPS = ("latin", "cjk", "rtl", "world", "reject")
N_SCRIPT = 4

BENCH_AT = (0, 320, 640, 960)
STEPS_DEFAULT = 960                  # 30 refresh cycles of 32 steps
REFRESH_EVERY = 32                   # violation re-labelling cadence (the kit's refresh_every)
LR_DECAY = (320, 640)                # lr x0.5 at these steps (settles the last violators)
ANCHOR_CAP = 0.02                    # full arm: mse of the 4 script logits vs base
ANCHOR_BETA = 10.0
GATE_CURE = 0.90                     # G4: fit reject rate
GATE_ABILITY_SLACK = 0.02            # G1: ability route acc >= base - slack
SCRIPT_MAP = {"arabic": "rtl", "cyr": "latin"}   # probe's calibrated mapping


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# the deployed graph: features + base logits through onnxruntime
# ---------------------------------------------------------------------------

def router_graph_with_features():
    """ORT session on the deployed router.onnx with the pooled 192-d feature
    tensor (the Concat feeding both Gemm heads) exposed as a 3rd output."""
    import onnx
    from onnx import TensorProto, helper
    m = onnx.load(str(MODEL_DIR / "router.onnx"))
    gemms = [n for n in m.graph.node if n.op_type == "Gemm"]
    feat_name = gemms[0].input[0]                    # '/Concat_output_0'
    assert all(g.input[0] == feat_name for g in gemms)
    m.graph.output.append(
        helper.make_tensor_value_info(feat_name, TensorProto.FLOAT, ("N", 192)))
    import onnxruntime as ort
    sess = ort.InferenceSession(m.SerializeToString(),
                                providers=["CPUExecutionProvider"])
    return sess, feat_name


def crop_arrays(records: list[dict]) -> np.ndarray:
    """Crops normalised to the router's fixed 32x512 input: the fit corpus is
    exactly 512 wide; the ability/transfer crops (CTC-width-free) are centred
    on a white 512-wide canvas (pad) or centre-cropped when wider.  White
    padding is the faithful choice: resize degrades even the BASE router
    (clean latin acc 0.892 vs 0.967 padded) — measured in the round's log.
    Deterministic and applied identically to base and new router."""
    out = []
    for r in records:
        img = load_image(KIT_DIR / r["file"])
        h, w = img.shape
        if w < 512:
            pad = 512 - w
            left, right = pad // 2, pad - pad // 2
            img = np.pad(img, ((0, 0), (left, right)),
                         constant_values=1.0)          # white
        elif w > 512:
            lo = (w - 512) // 2
            img = img[:, lo:lo + 512]
        out.append(img)
    return np.stack(out)


def gather_features(sess, crops: np.ndarray, feat_name: str,
                    chunk: int = 32) -> np.ndarray:
    outs = []
    for i in range(0, len(crops), chunk):
        x = np.ascontiguousarray(crops[i:i + chunk, None].astype(np.float32))
        g, fl, f = sess.run(None, {"x": x})
        outs.append(f.astype(np.float32))
    return np.concatenate(outs, 0)


def want_index(rec: dict) -> int | None:
    """Ground-truth group index: text crops -> script class, else reject.
    The generic 'text' kind (the clean/control ability renders) is latin."""
    kind = rec["kind"]
    if kind.startswith("text") and not rec.get("flipped"):
        script = rec.get("script", "latin")   # generic text crops are latin
        want = SCRIPT_MAP.get(script, script)
        return ROUTER_GROUPS.index(want)
    return 4                                    # non-text -> reject


_HW_CACHE: dict[str, np.ndarray] = {}


def head_weights() -> tuple[np.ndarray, np.ndarray]:
    if "W" not in _HW_CACHE:
        import onnx
        from onnx import numpy_helper
        init = {i.name: numpy_helper.to_array(i)
                for i in onnx.load(str(MODEL_DIR / "router.onnx")).graph.initializer}
        _HW_CACHE["W"] = init["group.weight"]
        _HW_CACHE["b"] = init["group.bias"]
    return _HW_CACHE["W"], _HW_CACHE["b"]


# ---------------------------------------------------------------------------
# round data (registered): fit = the router corpus, anchor = the ability
# corpora (clean + control, the parent's anchor scope 'good'),
# transfer = the three non-text families the router corpus never contains
# ---------------------------------------------------------------------------

def build_round_data(sess, feat_name) -> dict:
    fit = load_batch("router")
    fit_nt = [r for r in fit if not r["kind"].startswith("text_")]
    fit_tx = [r for r in fit if r["kind"].startswith("text_") and not r.get("flipped")]
    fit_flip = [r for r in fit if r.get("flipped")]
    feats_nt = gather_features(sess, crop_arrays(fit_nt), feat_name)
    feats_tx = gather_features(sess, crop_arrays(fit_tx), feat_name)
    feats_fl = gather_features(sess, crop_arrays(fit_flip), feat_name)

    # text pool for the anchor segment: fit router text + ability corpora
    pool = [{"rec": r, "src": "fit"} for r in fit_tx]
    for b in ("clean", "control"):
        for r in load_batch(b):
            pool.append({"rec": r, "src": b})
    pool_feats = gather_features(sess, crop_arrays([p["rec"] for p in pool]),
                                 feat_name)
    pool_targets = np.array([want_index(p["rec"]) for p in pool])
    pool_src = [p["src"] for p in pool]

    transfer_nt = {}
    for b in ("structured", "noise", "blank"):
        recs = load_batch(b)
        transfer_nt[b] = {"records": recs,
                          "feats": gather_features(sess, crop_arrays(recs),
                                                   feat_name)}
    return {
        "fit_nt": {"records": fit_nt, "feats": feats_nt},
        "fit_tx": {"records": fit_tx, "feats": feats_tx},
        "fit_flip": {"records": fit_flip, "feats": feats_fl},
        "text_pool": {"feats": pool_feats, "targets": pool_targets,
                      "src": pool_src, "n": len(pool)},
        "transfer_nt": transfer_nt,
    }


def base_group_logits(feats: np.ndarray) -> np.ndarray:
    W4, b4 = head_weights()
    return feats @ W4.T + b4


# ---------------------------------------------------------------------------
# head surgery: the LDA/prototype init of the reject row
# ---------------------------------------------------------------------------

def reject_row_init(feats_nt: np.ndarray, feats_tx: np.ndarray,
                    W4: np.ndarray) -> tuple[np.ndarray, float]:
    """w = unit(mu_nt - mu_text) scaled to the median script-row norm;
    b at the midpoint of the two projected means -> ~zero init margins."""
    mu_nt, mu_tx = feats_nt.mean(0), feats_tx.mean(0)
    w = mu_nt - mu_tx
    n = np.linalg.norm(w)
    if n < 1e-12:
        raise SystemExit("degenerate prototype direction")
    w = w / n * float(np.median(np.linalg.norm(W4, axis=1)))
    b = float(-w @ (mu_tx + mu_nt) / 2.0)
    return w.astype(np.float32), float(b)


# ---------------------------------------------------------------------------
# the two arms
# ---------------------------------------------------------------------------

def logits5(feats: torch.Tensor, W4: torch.Tensor, b4: torch.Tensor,
            w_rej: torch.Tensor, b_rej: torch.Tensor) -> torch.Tensor:
    l4 = feats @ W4.T + b4                       # [N,4] (grad only if W4 grad)
    l_rej = feats @ w_rej[:, None] + b_rej       # [N,1]
    return torch.cat([l4, l_rej], dim=1)


def head_metrics(feats_nt: np.ndarray, feats_tx: np.ndarray, tx_targets: np.ndarray,
                 W5: np.ndarray, b5: np.ndarray, base_tx_logits: np.ndarray | None) -> dict:
    l_nt = feats_nt @ W5.T + b5
    l_tx = feats_tx @ W5.T + b5
    m_nt = l_nt[:, 4] - l_nt[:, :4].max(1)
    m_tx = l_tx[:, 4] - l_tx[:, :4].max(1)
    arg_nt, arg_tx = l_nt.argmax(1), l_tx.argmax(1)
    route_tx = arg_tx == tx_targets
    mse = (float(((l_tx[:, :4] - base_tx_logits) ** 2).mean())
           if base_tx_logits is not None else None)
    return {
        "nontext_reject_rate": round(float((arg_nt == 4).mean()), 4),
        "nontext_residual_routes": {ROUTER_GROUPS[k]: int((arg_nt == k).sum())
                                    for k in range(N_SCRIPT)},
        "reject_margin_nontext_mean": round(float(m_nt.mean()), 3),
        "reject_margin_nontext_min": round(float(m_nt.min()), 3),
        "text_route_acc": round(float(route_tx.mean()), 4),
        "reject_win_rate_text": round(float((arg_tx == 4).mean()), 4),
        "reject_margin_text_mean": round(float(m_tx.mean()), 3),
        "reject_margin_text_max": round(float(m_tx.max()), 3),
        "text_logits_mse_vs_base": None if mse is None else round(mse, 6),
    }


def ability_metrics(pool_feats: np.ndarray, pool_targets: np.ndarray,
                    pool_src: list[str], W5: np.ndarray, b5: np.ndarray) -> dict:
    """Routing of the ability text (clean + control + fit text) — the anchor
    scope.  Per-source route accuracy and the reject-win count."""
    arg = (pool_feats @ W5.T + b5).argmax(1)
    ok = arg == pool_targets
    out = {}
    for src in ("fit", "clean", "control"):
        sel = np.array([s == src for s in pool_src])
        out[f"{src}_route_acc"] = round(float(ok[sel].mean()), 4)
        out[f"{src}_reject_wins"] = int((arg[sel] == 4).sum())
        out[f"{src}_n"] = int(sel.sum())
    out["reject_win_rate_ability"] = round(
        float((arg[np.array([s in ("clean", "control") for s in pool_src])] == 4).mean()), 4)
    return out


def transfer_metrics(transfer_nt: dict, W5: np.ndarray,
                     b5: np.ndarray) -> dict:
    out = {}
    for b, d in transfer_nt.items():
        arg = (d["feats"] @ W5.T + b5).argmax(1)
        out[f"{b}_reject_rate"] = round(float((arg == 4).mean()), 4)
        out[f"{b}_residual_routes"] = {ROUTER_GROUPS[k]: int((arg == k).sum())
                                       for k in range(N_SCRIPT)}
    out["mean_reject_rate"] = round(float(np.mean(
        [out[f"{b}_reject_rate"] for b in transfer_nt])), 4)
    return out


def train_arm(arm: str, data: dict, W4: np.ndarray, b4: np.ndarray,
              w0: np.ndarray, b0: float, steps: int, lr: float,
              seed: int) -> dict:
    """One arm of the router round.  Features are frozen (cached); only the
    head rows selected by `arm` carry gradients."""
    torch.manual_seed(seed)
    W4_t = torch.tensor(W4)
    b4_t = torch.tensor(b4)
    if arm == "reject_row":
        w_rej = torch.tensor(w0).clone().requires_grad_(True)
        params = [w_rej]
    else:
        W5 = torch.tensor(np.vstack([W4, w0[None]])).clone()
        W5.requires_grad_(True)
        w_rej = W5[4]                                         # view for logits
        params = [W5]
    b_rej = torch.tensor(float(b0)).requires_grad_(True)
    params.append(b_rej)

    feats_nt = torch.tensor(data["fit_nt"]["feats"])
    pool_feats = torch.tensor(data["text_pool"]["feats"])
    y_pool = torch.tensor(data["text_pool"]["targets"])
    base_pool_l4 = base_group_logits(data["text_pool"]["feats"])  # [240,4]
    pool_src = data["text_pool"]["src"]

    # violation-driven sampling (the kit's online self-labelling, ported to a
    # classifier head): every REFRESH_EVERY steps the violation sets are
    # re-labelled from the CURRENT head - text crops where reject still wins,
    # non-text crops where reject still loses - and steps sample violators
    # 3:1 nt:text.  Gradient lands only on crops the head gets wrong, so the
    # LDA init's 29% ability-text reject-wins are repaired fast while the
    # correct crops stay untouched; the round stops early when both sets are
    # empty (converged).
    def compute_violations() -> tuple[list[int], list[int]]:
        W5n, b5n = current_W5(), current_b5()
        l_nt = data["fit_nt"]["feats"] @ W5n.T + b5n
        l_tx = data["text_pool"]["feats"] @ W5n.T + b5n
        v_nt = np.where(l_nt.argmax(1) != 4)[0].tolist()
        v_tx = np.where(l_tx.argmax(1) !=
                        data["text_pool"]["targets"])[0].tolist()
        return v_nt, v_tx

    cyc = {"nt": 0, "t": 0}
    converged_at = None

    def current_W5() -> np.ndarray:
        with torch.no_grad():
            if arm == "reject_full":
                return W5.detach().numpy()
            return torch.cat([W4_t, w_rej[None]], 0).detach().numpy()

    def current_b5() -> np.ndarray:
        return np.append(b4_t.numpy(), float(b_rej.detach()))

    def bench() -> dict:
        W5n, b5n = current_W5(), current_b5()
        base_fit_l4 = (base_group_logits(data["fit_tx"]["feats"])
                       if arm == "reject_full" else None)
        fit = head_metrics(data["fit_nt"]["feats"], data["fit_tx"]["feats"],
                           np.array([want_index(r) for r in
                                     data["fit_tx"]["records"]]),
                           W5n, b5n, base_fit_l4)
        ability = ability_metrics(data["text_pool"]["feats"],
                                  data["text_pool"]["targets"], pool_src,
                                  W5n, b5n)
        transfer = transfer_metrics(data["transfer_nt"], W5n, b5n)
        return {"fit": fit, "ability": ability, "transfer": transfer}

    opt = torch.optim.Adam(params, lr=lr)
    steps_log, bench_log = [], []
    t0 = time.time()

    bench_log.append({"step": 0, **bench()})
    b0m = bench_log[0]
    print(f"[round:{arm}] bench@0: fit_rej={b0m['fit']['nontext_reject_rate']} "
          f"fit_route={b0m['fit']['text_route_acc']} "
          f"ability_rejwin={b0m['ability']['reject_win_rate_ability']} "
          f"transfer_rej={b0m['transfer']['mean_reject_rate']}")

    viol_log = []
    v_nt, v_tx = compute_violations()
    viol_log.append({"step": 0, "n_violating_nt": len(v_nt),
                     "n_violating_text": len(v_tx)})
    print(f"[round:{arm}] violations@0: nt={len(v_nt)} text={len(v_tx)}")

    for step in range(1, steps + 1):
        if step > 1 and (step - 1) % REFRESH_EVERY == 0:
            v_nt, v_tx = compute_violations()
            cyc = {"nt": 0, "t": 0}
            viol_log.append({"step": step - 1, "n_violating_nt": len(v_nt),
                             "n_violating_text": len(v_tx)})
            print(f"[round:{arm}] violations@{step - 1}: nt={len(v_nt)} "
                  f"text={len(v_tx)}")
            if not v_nt and not v_tx:
                converged_at = step - 1
                print(f"[round:{arm}] converged at step {step - 1} "
                      f"(no violations left) - stopping early")
                break
        if v_nt and (not v_tx or (step % 4) != 0):
            kind, idx = "nt", v_nt[cyc["nt"] % len(v_nt)]
        elif v_tx:
            kind, idx = "t", v_tx[cyc["t"] % len(v_tx)]
        else:
            kind, idx = "nt", v_nt[cyc["nt"] % len(v_nt)]
        cyc[kind] += 1
        ts = time.time()
        if step in LR_DECAY:
            for gset in opt.param_groups:
                gset["lr"] *= 0.5
            print(f"[round:{arm}] step {step}: lr x0.5 -> {opt.param_groups[0]['lr']:.2e}")
        opt.zero_grad(set_to_none=True)
        if kind == "nt":
            feats = feats_nt[idx:idx + 1]
            if arm == "reject_row":
                logits = logits5(feats, W4_t, b4_t, w_rej, b_rej)
            else:
                logits = logits5(feats, W5[:4], b4_t, W5[4], b_rej)
            loss = F.cross_entropy(logits, torch.tensor([4]))
            mse_v = 0.0
        else:
            if arm == "reject_row":
                logits = logits5(pool_feats[idx:idx + 1], W4_t, b4_t,
                                 w_rej, b_rej)
                loss = F.cross_entropy(logits, y_pool[idx:idx + 1])
                mse_v = 0.0
            else:  # full arm: script rows train, script bias frozen
                l5 = logits5(pool_feats[idx:idx + 1], W5[:4], b4_t,
                             W5[4], b_rej)
                loss = F.cross_entropy(l5, y_pool[idx:idx + 1])
                l4 = l5[:, :4]
                mse = F.mse_loss(l4, torch.tensor(base_pool_l4[idx:idx + 1]))
                mse_v = float(mse.detach())
                loss = loss + ANCHOR_BETA * torch.relu(mse - ANCHOR_CAP)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        steps_log.append({"step": step, "kind": kind,
                          "loss": round(float(loss.detach()), 4),
                          "mse": round(mse_v, 6),
                          "s": round(time.time() - ts, 3)})
        if step in BENCH_AT:
            bl = bench()
            bench_log.append({"step": step, **bl})
            print(f"[round:{arm}] bench@{step}: fit_rej="
                  f"{bl['fit']['nontext_reject_rate']} "
                  f"fit_route={bl['fit']['text_route_acc']} "
                  f"clean_route={bl['ability']['clean_route_acc']} "
                  f"ctrl_route={bl['ability']['control_route_acc']} "
                  f"ability_rejwin={bl['ability']['reject_win_rate_ability']} "
                  f"transfer_rej={bl['transfer']['mean_reject_rate']}")

    final = bench()
    W5_final, b5_final = current_W5(), current_b5()

    dW = (W5_final - np.vstack([W4, np.zeros((1, W4.shape[1]))]))
    rel_row4 = float(np.linalg.norm(dW[4]) /
                     (np.linalg.norm(w0) + 1e-12))       # vs its own init
    log = {
        "arm": arm,
        "args": {"steps": steps, "lr": lr, "seed": seed,
                 "grad_clip": 1.0, "anchor_beta": ANCHOR_BETA,
                 "anchor_cap": ANCHOR_CAP if arm == "reject_full" else None,
                 "cycle": "violation-driven (refresh every 32): violators "
                          "sampled 3:1 nt:text; early stop at convergence",
                 "lr_schedule": f"constant, x0.5 at steps {list(LR_DECAY)}",
                 "anchor_scope": "text = fit router text + clean + control "
                                 "(the ability corpora)",
                 "trainable_rows": [4] if arm == "reject_row"
                 else [0, 1, 2, 3, 4],
                 "biases": "reject bias trainable; script biases frozen "
                           "(parity)",
                 "n_trainable": int(w_rej.numel() + 1) if arm == "reject_row"
                 else int(W5.numel() + 1)},
        "init": {"kind": "LDA prototype (mu_nt - mu_text, midpoint bias)",
                 "w_norm": round(float(np.linalg.norm(w0)), 4),
                 "script_row_norm_median": round(float(np.median(
                     np.linalg.norm(W4, axis=1))), 4)},
        "final": final,
        "rel_frobenius_row4_vs_init": round(rel_row4, 5),
        "rel_frobenius_vs_W4": round(float(np.linalg.norm(dW) /
                                           (np.linalg.norm(W4) + 1e-12)), 5),
        "steps": steps_log, "bench": bench_log, "violations": viol_log,
        "converged_at": converged_at,
        "W5": W5_final, "b5": b5_final,
        "wall_time_s": round(time.time() - t0, 1),
    }
    print(f"[round:{arm}] done in {log['wall_time_s']}s: "
          f"fit_rej={final['fit']['nontext_reject_rate']} "
          f"transfer_rej={final['transfer']['mean_reject_rate']}")
    return log


# ---------------------------------------------------------------------------
# gates + winner
# ---------------------------------------------------------------------------

def gates_for(arm: str, log: dict, base_ability: dict) -> dict:
    f, ab = log["final"], log["final"]["ability"]
    g = {}
    if arm == "reject_row":
        g["G1_script_rows"] = "bit-identical to base (exact by construction)"
        g["G1_script_rows_pass"] = True
    else:
        mse_ok = (f["fit"]["text_logits_mse_vs_base"] or 0) <= ANCHOR_CAP
        route_ok = f["fit"]["text_route_acc"] >= 1.0 - 1e-9
        g["G1_script_rows"] = (f"fit route_acc={f['fit']['text_route_acc']} "
                               f"anchor mse={f['fit']['text_logits_mse_vs_base']}")
        g["G1_script_rows_pass"] = bool(mse_ok and route_ok)
    ab_ok = (ab["clean_route_acc"] >= base_ability["clean"] - GATE_ABILITY_SLACK
             and ab["control_route_acc"] >= base_ability["control"] - GATE_ABILITY_SLACK)
    rejwin_ok = ab["reject_win_rate_ability"] == 0.0
    g["G1_ability"] = (f"clean {ab['clean_route_acc']} (base {base_ability['clean']}), "
                       f"control {ab['control_route_acc']} (base {base_ability['control']}), "
                       f"reject wins {ab['clean_reject_wins'] + ab['control_reject_wins']}")
    g["G1_ability_pass"] = bool(ab_ok and rejwin_ok)
    g["G1_pass"] = bool(g["G1_script_rows_pass"] and g["G1_ability_pass"])
    g["G2_flip"] = "flip head untouched (bitwise, verified post-export)"
    g["G2_pass"] = None                       # decided after export
    g["G3_trunk"] = "all non-head initializers array-identical (post-export)"
    g["G3_pass"] = None
    g["G4_cure"] = (f"fit_reject={f['fit']['nontext_reject_rate']} "
                    f"reject_win_text={f['fit']['reject_win_rate_text']}")
    g["G4_pass"] = bool(f["fit"]["nontext_reject_rate"] >= GATE_CURE and
                        f["fit"]["reject_win_rate_text"] == 0.0)
    g["pass"] = bool(g["G1_pass"] and g["G4_pass"])
    return g


# ---------------------------------------------------------------------------
# export the winner as a deployed ONNX
# ---------------------------------------------------------------------------

def export_onnx(W5: np.ndarray, b5: np.ndarray, out_name: str,
                meta_extra: dict) -> dict:
    import onnx
    from onnx import numpy_helper
    m = onnx.load(str(MODEL_DIR / "router.onnx"))
    init = {i.name: i for i in m.graph.initializer}
    for name, arr in (("group.weight", W5.astype(np.float32)),
                      ("group.bias", b5.astype(np.float32))):
        m.graph.initializer.remove(init[name])
        m.graph.initializer.append(numpy_helper.from_array(arr, name))
    for o in m.graph.output:
        if o.name == "group":
            o.type.tensor_type.shape.dim[1].dim_value = W5.shape[0]
    out_path = MODEL_DIR / out_name
    onnx.save(m, str(out_path))

    # ---- G3: trunk arrays identical ----
    base = onnx.load(str(MODEL_DIR / "router.onnx"))
    base_arr = {i.name: numpy_helper.to_array(i) for i in base.graph.initializer}
    new_arr = {i.name: numpy_helper.to_array(i) for i in
               onnx.load(str(out_path)).graph.initializer}
    trunk_ok = all(np.array_equal(base_arr[k], new_arr[k])
                   for k in base_arr if not k.startswith("group."))
    flip_ok = (np.array_equal(base_arr["flip.weight"], new_arr["flip.weight"])
               and np.array_equal(base_arr["flip.bias"], new_arr["flip.bias"]))

    # ---- smoke parity: ORT base vs ORT extended, same crops ----
    import onnxruntime as ort
    s_base = ort.InferenceSession(str(MODEL_DIR / "router.onnx"),
                                  providers=["CPUExecutionProvider"])
    s_new = ort.InferenceSession(str(out_path),
                                 providers=["CPUExecutionProvider"])
    recs = load_batch("router")[:48]
    crops = crop_arrays(recs)
    x = np.ascontiguousarray(crops[:, None].astype(np.float32))
    gb, fb = s_base.run(None, {"x": x})
    gn, fn = s_new.run(None, {"x": x})
    g4_max = float(np.abs(gn[:, :4] - gb).max())
    flip_bitwise = bool(np.array_equal(fb, fn))
    return {"path": str(out_path.relative_to(KIT_DIR)), "n_outputs": 5,
            "trunk_arrays_identical": bool(trunk_ok),
            "flip_head_bit_identical": bool(flip_ok),
            "group_rows0_3_max_abs_diff_vs_base": round(g4_max, 8),
            "flip_outputs_bitwise_equal_on_smoke": flip_bitwise,
            "sha256": sha256_file(out_path), **meta_extra}


# ---------------------------------------------------------------------------
# full evaluation of the winner (fit + ability + transfer + flip + spectral)
# ---------------------------------------------------------------------------

def evaluate_winner(arm: str, log: dict, data: dict) -> tuple[dict, list]:
    W5, b5 = log["W5"], log["b5"]
    rows = []

    def add_rows(tag, records, feats, wants):
        base = base_group_logits(feats)
        new = feats @ W5.T + b5
        for i, r in enumerate(records):
            rows.append({
                "group": tag, "id": r["id"], "kind": r["kind"],
                "script": r.get("script"), "flipped": bool(r.get("flipped")),
                "want": wants[i],
                "base_argmax": ROUTER_GROUPS[int(base[i].argmax())],
                "new_argmax": ROUTER_GROUPS[int(new[i].argmax())],
                "reject_margin": round(float(new[i, 4] - new[i, :4].max()), 3),
            })

    ftx = data["fit_tx"]["records"]
    add_rows("fit_text", ftx, data["fit_tx"]["feats"],
             [ROUTER_GROUPS[want_index(r)] for r in ftx])
    fnt = data["fit_nt"]["records"]
    add_rows("fit_nontext", fnt, data["fit_nt"]["feats"],
             ["reject"] * len(fnt))
    ffl = data["fit_flip"]["records"]
    add_rows("fit_flipped", ffl, data["fit_flip"]["feats"],
             [None] * len(ffl))
    # ability rows (the anchored ability corpora: clean + control)
    pool = data["text_pool"]
    ab_start = len(data["fit_tx"]["records"])
    clean_recs = _src_records("clean")
    ctrl_recs = _src_records("control")
    add_rows("ability_clean", clean_recs,
             pool["feats"][ab_start:ab_start + len(clean_recs)],
             ["latin"] * len(clean_recs))
    add_rows("ability_control", ctrl_recs,
             pool["feats"][ab_start + len(clean_recs):],
             ["latin"] * len(ctrl_recs))
    for b, d in data["transfer_nt"].items():
        add_rows(f"transfer_{b}", d["records"], d["feats"],
                 ["reject"] * len(d["records"]))

    ev = {"round": "router_reject", "arm": arm,
          "head": {"groups": list(ROUTER_GROUPS),
                   "n_params_added": 193,
                   "trainable_rows": log["args"]["trainable_rows"]},
          "F4_fit": log["final"]["fit"],
          "F4_ability": log["final"]["ability"],
          "F4_transfer": log["final"]["transfer"],
          "F4_base_reference": {
              "nontext_route_rate": 1.0,
              "nontext_routed_to": data["base_histogram"],
              "route_error_rate": data["base_route_error"],
              "ability_route_acc": {"clean": data["base_ability"]["clean"],
                                    "control": data["base_ability"]["control"]},
              "note": "base re-measured on the same features (deployed "
                      "int8 router); matches probe_report.json"},
          "gates": log["gates"],
          "head_spectrum": log["spectrum"],
          "export": log["export"],
          "n_rows": len(rows)}
    return ev, rows


_SRC_CACHE: dict[str, list[dict]] = {}


def _src_records(batch: str) -> list[dict]:
    if batch not in _SRC_CACHE:
        _SRC_CACHE[batch] = load_batch(batch)
    return _SRC_CACHE[batch]


def head_spectrum(dW: np.ndarray, w_init: np.ndarray) -> dict:
    """The edit spectrum of the head delta (the kit's Step-14 language)."""
    sq = dW @ dW.T / max(dW.shape)
    lam = np.clip(np.linalg.eigvalsh(sq), 0, None)[::-1]
    sigma2 = fit_sigma2(lam, trim_hi=0.05)
    if sigma2 <= 0:
        return {"lambda_max": round(float(lam[0]), 6), "lambda_plus": None}
    _, lp = mp_edges(min(dW.shape) / max(dW.shape), sigma2)
    spikes = int((lam > 2.0 * lp).sum())
    U, S, Vh = np.linalg.svd(dW, full_matrices=False)
    v1 = Vh[0] / (np.linalg.norm(Vh[0]) + 1e-12)
    wi = w_init / (np.linalg.norm(w_init) + 1e-12)
    return {"lambda_max": round(float(lam[0]), 6),
            "lambda_plus": round(float(lp), 6),
            "lambda_max_over_lambda_plus": round(float(lam[0] / lp), 2),
            "n_spikes_above_2lambda_plus": spikes,
            "single_spike": bool(spikes == 1),
            "sigma1_share": round(float(S[0] ** 2 / (S ** 2).sum()), 4),
            "cos_v1_edit_w_init": round(float(abs(v1 @ wi)), 4)}


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=STEPS_DEFAULT)
    ap.add_argument("--lr_row", type=float, default=2e-3)
    ap.add_argument("--lr_full", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    seed_all(args.seed)

    print("[round] extracting deployed-graph features (trunk frozen)...")
    sess, feat_name = router_graph_with_features()
    data = build_round_data(sess, feat_name)

    W4, b4 = head_weights()

    # base re-measurement (consistency with the probe + ability reference)
    tx_records = data["fit_tx"]["records"]
    tx_targets = np.array([want_index(r) for r in tx_records])
    base_tx_arg = base_group_logits(data["fit_tx"]["feats"]).argmax(1)
    base_route_error = round(1 - float((base_tx_arg == tx_targets).mean()), 4)
    base_nt_arg = base_group_logits(data["fit_nt"]["feats"]).argmax(1)
    hist = {ROUTER_GROUPS[k]: int((base_nt_arg == k).sum())
            for k in range(N_SCRIPT)}
    base_ability = {}
    for b in ("clean", "control"):
        arg = base_group_logits(data["text_pool"]["feats"][
            ab_slice(data, b)]).argmax(1)
        base_ability[b] = round(float((arg == 0).mean()), 4)
    print(f"[round] base re-measured: route_error={base_route_error} "
          f"nontext histogram={hist} (probe: latin 20 cjk 11 rtl 4 world 5)")
    print(f"[round] base ability route acc: {base_ability}")
    data["base_histogram"] = hist
    data["base_route_error"] = base_route_error
    data["base_ability"] = base_ability

    w0, b0 = reject_row_init(data["fit_nt"]["feats"],
                             data["fit_tx"]["feats"], W4)
    print(f"[round] reject-row LDA init: |w|={np.linalg.norm(w0):.3f} "
          f"(median script-row norm {np.median(np.linalg.norm(W4, axis=1)):.3f})")

    if args.dry_run:
        print("[round] dry run ok - exiting before training")
        return

    logs = {}
    for arm, lr in (("reject_row", args.lr_row), ("reject_full", args.lr_full)):
        logs[arm] = train_arm(arm, data, W4, b4, w0, b0, args.steps, lr,
                              args.seed)
        logs[arm]["gates"] = gates_for(arm, logs[arm], base_ability)

    # winner: gate-passing arm with the higher transfer reject rate
    passing = [a for a, lg in logs.items() if lg["gates"]["pass"]]
    if not passing:
        raise SystemExit("no arm passed the gates - the round fails honestly")
    winner = max(passing, key=lambda a: (logs[a]["final"]["transfer"]
                 ["mean_reject_rate"], a == "reject_row"))
    print(f"[round] winner: {winner} (passing: {passing})")

    # export + post-export gates
    exp = export_onnx(logs[winner]["W5"], logs[winner]["b5"],
                      "router_reject.onnx",
                      {"round": "router_reject", "winner_arm": winner})
    logs[winner]["export"] = exp
    logs[winner]["gates"]["G2_pass"] = bool(exp["flip_head_bit_identical"] and
                                            exp["flip_outputs_bitwise_equal_on_smoke"])
    logs[winner]["gates"]["G3_pass"] = bool(exp["trunk_arrays_identical"])
    logs[winner]["gates"]["pass"] = bool(logs[winner]["gates"]["G1_pass"] and
                                         logs[winner]["gates"]["G2_pass"] and
                                         logs[winner]["gates"]["G3_pass"] and
                                         logs[winner]["gates"]["G4_pass"])

    # spectral section of the winner's head edit
    W4pad = np.vstack([W4, np.zeros((1, W4.shape[1]))])
    logs[winner]["spectrum"] = head_spectrum(logs[winner]["W5"] - W4pad, w0)

    meta = {
        "model": "llamaindex/OCRia-nano-1 router + reject (round router_reject)",
        "parent": "router.onnx (deployed, sha256-pinned in RUN_PASSPORT.json)",
        "group_order": list(ROUTER_GROUPS),
        "reject_definition": "class 4 = textless crop; abstain from routing",
        "winner_arm": winner,
        "parent_sha256": sha256_file(MODEL_DIR / "router.onnx"),
        "sha256": exp["sha256"],
        "gates": logs[winner]["gates"],
    }
    (MODEL_DIR / "router_reject.meta.json").write_text(
        json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")

    ev, rows = evaluate_winner(winner, logs[winner], data)

    # artifacts
    out_log = {"round": "router_reject",
               "model": "llamaindex/OCRia-nano-1 router.onnx (deployed graph)",
               "args": {"steps": args.steps, "lr_row": args.lr_row,
                        "lr_full": args.lr_full, "seed": args.seed,
                        "bench_at": list(BENCH_AT), "anchor_cap": ANCHOR_CAP,
                        "anchor_beta": ANCHOR_BETA, "gate_cure": GATE_CURE,
                        "gate_ability_slack": GATE_ABILITY_SLACK},
               "base_recheck": {"route_error_rate": base_route_error,
                                "nontext_routed_to": hist,
                                "ability_route_acc": base_ability},
               "init": {"kind": "LDA prototype",
                        "w_norm": round(float(np.linalg.norm(w0)), 4)},
               "arms": {a: {k: v for k, v in lg.items()
                            if k not in ("W5", "b5", "steps")}
                        for a, lg in logs.items()},
               "winner": winner,
               "spectral": logs[winner]["spectrum"],
               "export": exp}
    dump_json(out_log, RESULTS_DIR / "router_reject_training_log.json")
    ev["training_log_ref"] = "Results/router_reject_training_log.json"
    dump_json(ev, RESULTS_DIR / "eval_router_reject.json")
    with open(RESULTS_DIR / "eval_router_reject_rows.jsonl", "w",
              encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    np.savez_compressed(RESULTS_DIR / "router_reject_deltas.npz",
                        reject_row_W5_delta=(logs["reject_row"]["W5"] - W4pad),
                        reject_row_b5_delta=(logs["reject_row"]["b5"] -
                                             np.append(b4, 0.0)),
                        reject_full_W5_delta=(logs["reject_full"]["W5"] - W4pad),
                        reject_full_b5_delta=(logs["reject_full"]["b5"] -
                                              np.append(b4, 0.0)),
                        w_rej_init=w0)

    f = logs[winner]["final"]
    print(f"[round] F4 cure: fit reject 0.0 (no class) -> "
          f"{f['fit']['nontext_reject_rate']} | transfer reject "
          f"{f['transfer']['mean_reject_rate']} | fit text route "
          f"{f['fit']['text_route_acc']} (base 1.0) | flip bitwise="
          f"{exp['flip_outputs_bitwise_equal_on_smoke']}")
    print("[round] wrote router_reject_training_log.json / eval_router_reject.json "
          "/ eval_router_reject_rows.jsonl / router_reject_deltas.npz")
    print("[round] wrote Model/router_reject.onnx + Model/router_reject.meta.json")


def ab_slice(data: dict, batch: str) -> slice:
    """Slice of the text-pool feature array that belongs to `batch`."""
    src = data["text_pool"]["src"]
    idx = [i for i, s in enumerate(src) if s == batch]
    return slice(idx[0], idx[-1] + 1)


if __name__ == "__main__":
    main()
