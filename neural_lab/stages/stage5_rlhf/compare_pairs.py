"""
stage5_rlhf.compare_pairs — paired base↔tuned comparison (Stage 5).

For one pair (same family, same tokenizer lineage), both models process
the SAME items:

  1. SPECTRA   — per-layer gap_ratio / effective rank / ncrit_free on a
                 shared open-ended prompt set (greedy generation);
                 paired permutation test per statistic;
  2. N_CRIT    — shared arithmetic task set -> first-error survival fit
                 per model (tau, gamma) — the Stage-3 machinery;
  3. MU_PROXY  — mu_eff measured OUTSIDE the fits: token-mean KL
                 divergence between the two models' next-token
                 distributions on the shared prompt set (both directions;
                 the base→tuned direction is the 'utility pressure' proxy);
  4. VERDICT   — Delta(tau), Delta(gap_ratio), Delta(beta) with CIs;
                 across >= 3 pairs the Stage-3 scaling module aggregates
                 (T_crit ~ mu_eff^(-1/beta)).

Outputs one JSON record per pair under <out>/<pair_id>/ — the aggregate
across pairs is run by stage3's --scaling-in (single source of truth for
the law).
"""

from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_NEURAL_LAB = os.path.dirname(os.path.dirname(_HERE))  # .../neural_lab
if _NEURAL_LAB not in sys.path:
    sys.path.insert(0, _NEURAL_LAB)

from stages.common.generation import generate_with_capture  # noqa: E402
from stages.common.hf_model import load_causal_lm, model_passport  # noqa: E402
from stages.common.passports import build_passport, data_hash  # noqa: E402
from stages.stage2_detector.spectral_features import spectral_summary  # noqa: E402
from stages.stage3_ncrit.error_labels import first_error_position, incremental_token_texts  # noqa: E402
from stages.stage3_ncrit.error_labels import trajectory_table  # noqa: E402
from stages.stage3_ncrit.numeric_tasks import build_task_set, few_shot_header, make_checker  # noqa: E402
from stages.stage3_ncrit.survival_fit import (  # noqa: E402
    bootstrap_indices,
    bootstrap_survival,
    detect_breakpoint,
    fit_survival,
)

DEFAULT_PROMPTS = [
    "The most important thing in life is",
    "In the future, people will",
    "The best way to learn a language is",
    "A good friend always",
    "The history of computing shows that",
    "When the weather is cold, people",
    "The purpose of education is",
    "Scientists recently discovered that",
    "The most beautiful city in the world is",
    "A healthy diet includes",
]


def kl_divergence_proxy(model_a, model_b, tokenizer, prompt: str, max_tokens: int = 8) -> dict:
    """Token-mean KL(p_a || p_b) over `max_tokens` consecutive greedy
    positions. The sequence is built once (greedy by model_a); at each
    position BOTH models are teacher-forced on the SAME prefix and their
    next-token distributions are compared — identical inputs, identical
    positions, so an identical model gives exactly KL = 0.

    The values are normalized by log(V) — a heuristic scale in [0, ~1],
    NOT a calibrated mu_eff; clamp at 0 guards against float round-off
    making a (near-)identical pair come out slightly negative.
    """
    import torch

    device = next(model_a.parameters()).device
    ids = tokenizer.encode(prompt)
    ids = ids[: min(len(ids), 512)]
    positions = []

    with torch.no_grad():
        # position 0: after the prompt
        out = model_a(torch.tensor([ids], device=device))
        logits_a = out.logits[0, -1].float()
        logits_b = model_b(torch.tensor([ids], device=device)).logits[0, -1].float()
        positions.append((logits_a, logits_b))
        cur = ids
        for _ in range(max_tokens - 1):
            cur = cur + [int(logits_a.argmax().item())]
            logits_a = model_a(torch.tensor([cur], device=device)).logits[0, -1].float()
            logits_b = model_b(torch.tensor([cur], device=device)).logits[0, -1].float()
            positions.append((logits_a, logits_b))

    kls_ab, kls_ba = [], []
    V = float(positions[0][0].shape[-1])
    for la, lb in positions:
        kl_ab = torch.nn.functional.kl_div(
            torch.log_softmax(lb, dim=-1), torch.log_softmax(la, dim=-1),
            log_target=True, reduction="sum",
        ).item()
        kl_ba = torch.nn.functional.kl_div(
            torch.log_softmax(la, dim=-1), torch.log_softmax(lb, dim=-1),
            log_target=True, reduction="sum",
        ).item()
        kls_ab.append(max(0.0, kl_ab) / np.log(V))  # log-vocab normalized, [0, ~1]
        kls_ba.append(max(0.0, kl_ba) / np.log(V))
    return {"kl_ab": float(np.mean(kls_ab)), "kl_ba": float(np.mean(kls_ba))}


def assert_same_tokenizer_lineage(tok_a, tok_b, pair_id: str) -> None:
    """Fail loudly if a 'same-lineage' pair actually carries two different
    tokenizers: different vocab => the shared prompts tokenise differently
    and every paired comparison is invalid."""
    if tok_a is tok_b:
        return
    va = tok_a.get_vocab() if hasattr(tok_a, "get_vocab") else None
    vb = tok_b.get_vocab() if hasattr(tok_b, "get_vocab") else None
    if va is not None and vb is not None and va != vb:
        raise ValueError(
            f"pair {pair_id!r}: base and tuned tokenizers have different vocabularies "
            f"({len(va)} vs {len(vb)} entries); same-lineage pairs must share the "
            "tokenizer — check the pair registry or load the tuned model with the base tokenizer"
        )


def spectra_on_prompts(model, tokenizer, prompts, cfg) -> list[dict]:
    out = []
    for p in prompts:
        gen = generate_with_capture(
            model, tokenizer, p, max_new_tokens=cfg["max_new_tokens"],
            temperature=0.0, capture_hidden=True, fast=cfg["fast"],
        )
        out.append(spectral_summary(gen["final_hidden"]))
    return out


def survival_on_tasks(model, tokenizer, tasks, cfg, index_matrix: np.ndarray | None = None) -> dict:
    """First-error survival fit of ONE model on the shared task set.

    index_matrix: shared bootstrap resamples (see bootstrap_indices). When
    base and tuned are fitted with the SAME resample matrix, their c
    samples are aligned row-by-row and the Δτ CI is a properly PAIRED
    bootstrap over tasks. The per-row c samples are returned as
    'tau_boot_samples' (NaN where a resample had no errors) so the caller
    can build that Δτ CI."""
    header = few_shot_header(n_shot=cfg["n_shot"], seed=cfg["seed"])
    tol = cfg.get("tol", 0.05)
    trajs = []
    for task in tasks:
        gen = generate_with_capture(
            model, tokenizer, header + task["prompt"], max_new_tokens=cfg["max_new_tokens"],
            temperature=0.0, capture_hidden=False, fast=cfg["fast"],
        )
        pieces = incremental_token_texts(tokenizer, gen["output_ids"])
        t_err, n_tok = first_error_position(pieces, make_checker(task, tol=tol))
        trajs.append({"task_id": task["task_id"], "first_error": t_err, "n_tokens": n_tok})
    tab = trajectory_table(trajs)
    t = np.asarray([r["t_first_error"] for r in tab])
    d = np.asarray([0.0 if r["censored"] else 1.0 for r in tab])
    if d.sum() == 0:
        return {"note": "no errors on this task set (censored at max tokens)", "n_traj": int(len(t))}
    fit = fit_survival(t, d, mu=None, seed=cfg["seed"])
    warm = np.asarray(fit["params"], dtype=np.float64) if fit.get("params") else None
    boot = bootstrap_survival(
        t, d, mu=None, n_boot=cfg["n_boot"], seed=cfg["seed"],
        index_matrix=index_matrix, warm_start=warm,
    )
    bp = detect_breakpoint(t, d, at_risk=np.asarray([r["n_tokens"] for r in tab]))
    return {"tau": fit["tau"], "gamma": fit["gamma"], "tau_ci95": boot.get("c", {}).get("ci95"),
            "tau_boot_samples": boot.get("c", {}).get("samples"),
            "error_rate": float(d.mean()), "n_traj": int(len(t)),
            "breakpoint": bp}


def paired_permutation(a: list[float], b: list[float], n_perm: int = 2000, seed: int = 0) -> dict:
    """Paired comparison of a vs b: the effect is mean(a - b).

    CI: paired bootstrap over items (resample pairs with replacement) —
    this is a confidence interval of the EFFECT.
    p: exact-style paired permutation (random pair-label swap) with the +1
    correction — the smallest attainable p is 1/(n_perm + 1), never 0.
    This is a test of H0 'a and b are exchangeable'.
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if len(a) != len(b) or len(a) == 0:
        raise ValueError("paired arrays must be same non-zero length")
    d = a - b
    n = len(d)
    rng = np.random.default_rng(seed)
    # paired bootstrap CI of the effect
    boots = np.empty(n_perm)
    for i in range(n_perm):
        idx = rng.integers(0, n, size=n)
        boots[i] = d[idx].mean()
    # permutation p (label swap within pairs)
    null = np.empty(n_perm)
    for i in range(n_perm):
        flip = rng.integers(0, 2, size=n)
        null[i] = np.mean(np.where(flip == 1, -d, d))
    d0 = float(d.mean())
    p = (1.0 + float((np.abs(null) >= abs(d0) - 1e-12).sum())) / (1.0 + n_perm)
    return {"delta_mean": d0,
            "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
            "p_two_sided": float(min(1.0, p))}


def evaluate_pair(pair: dict, cfg) -> dict:
    print(f"[stage5] pair {pair['id']}: base={pair['base']} tuned={pair['tuned']}", flush=True)
    # optional per-side revisions (checkpoint-ablation entries pin them; two
    # runs of the same repo without revisions would silently give Δ = 0)
    model_base, tok_base = load_causal_lm(pair["base"], revision=pair.get("base_revision"),
                                          dtype=cfg["dtype"], device=cfg["device"])
    model_tuned, tok_tuned = load_causal_lm(pair["tuned"], revision=pair.get("tuned_revision"),
                                            dtype=cfg["dtype"], device=cfg["device"])
    if pair.get("base") == pair.get("tuned") and not (pair.get("base_revision") or pair.get("tuned_revision")):
        raise ValueError(f"pair {pair['id']!r}: base == tuned with no revisions pinned — "
                         "the comparison is a no-op; pin base_revision/tuned_revision in the registry")
    assert_same_tokenizer_lineage(tok_base, tok_tuned, pair["id"])
    tok = tok_base  # the registry asserts the same tokenizer lineage; ONE tokenizer
    # for both models keeps tokenization identical by construction (P1.14)

    prompts = cfg.get("prompts") or DEFAULT_PROMPTS
    tasks = build_task_set(n_tasks=cfg["n_tasks"], seed=cfg["seed"])
    passport = build_passport(
        experiment_id="stage5_pair_v1",
        hypothesis_id="S5",
        model_fields={
            "model_name": f"{pair['base']} vs {pair['tuned']}",
            "config_hash": data_hash([model_passport(model_base, tok), model_passport(model_tuned, tok_tuned)]),
            "weight_sha256": None,
        },
        config={"pair": pair, "n_tasks": cfg["n_tasks"], "seed": cfg["seed"],
                "data_hash": data_hash([p for p in prompts])},
    )

    spec_base = spectra_on_prompts(model_base, tok, prompts, cfg)
    spec_tuned = spectra_on_prompts(model_tuned, tok, prompts, cfg)

    def col(specs, key):
        return [s[key] for s in specs]

    deltas = {}
    for key in ("gap_ratio_last", "effrank_last", "ncrit_free_last", "gap_ratio_mean"):
        deltas[key] = paired_permutation(col(spec_base, key), col(spec_tuned, key), seed=cfg["seed"])

    # PAIRED survival bootstrap: one shared resample matrix over the shared
    # task set, so Δτ = tau_tuned − tau_base has a real paired CI (P3.14)
    n_traj_tasks = len(tasks)
    shared_idx = bootstrap_indices(n_traj_tasks, cfg["n_boot"], seed=cfg["seed"])
    surv_base = survival_on_tasks(model_base, tok, tasks, cfg, index_matrix=shared_idx)
    surv_tuned = survival_on_tasks(model_tuned, tok, tasks, cfg, index_matrix=shared_idx)

    kl_rows = [kl_divergence_proxy(model_base, model_tuned, tok, p, max_tokens=cfg["kl_positions"])
               for p in prompts]
    mu_proxy = float(np.mean([r["kl_ab"] for r in kl_rows]))
    mu_proxy_rev = float(np.mean([r["kl_ba"] for r in kl_rows]))

    delta_tau = None
    if surv_base.get("tau") is not None and surv_tuned.get("tau") is not None:
        sb = np.asarray(surv_base.get("tau_boot_samples") or [], dtype=np.float64)
        st = np.asarray(surv_tuned.get("tau_boot_samples") or [], dtype=np.float64)
        point = float(surv_tuned["tau"] - surv_base["tau"])
        if sb.size and sb.size == st.size:
            diffs = st - sb
            ok = diffs[np.isfinite(diffs)]
            if ok.size:
                delta_tau = {
                    "delta_tau": point,
                    "ci95": [float(np.quantile(ok, 0.025)), float(np.quantile(ok, 0.975))],
                    "n_paired_boot": int(ok.size),
                    "note": "paired bootstrap over the shared task set (same resamples for both models)",
                }
        else:
            delta_tau = {"delta_tau": point, "ci95": None,
                         "note": "point difference only; aligned bootstrap samples unavailable"}

    record = {
        "pair_id": pair["id"],
        "passport": passport,
        "spectral_deltas": deltas,
        "survival_base": surv_base,
        "survival_tuned": surv_tuned,
        "delta_tau_tuned_minus_base": delta_tau,
        "mu_eff_proxy_kl_ab": mu_proxy,
        "mu_eff_proxy_kl_ba": mu_proxy_rev,
        "spectra_per_prompt": {"base": spec_base, "tuned": spec_tuned},
    }
    return record
