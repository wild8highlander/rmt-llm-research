"""
experiments.run_h1 — Experiment No. 1: predicting LLM errors from internal
features (H1), a reproducible pipeline.

HYPOTHESIS H1: the spectral characteristics of internal activations contain
information about the model's errors BEYOND what standard confidence
signals provide.

The protocol was fixed BEFORE the run (EXPERIMENT_PROTOCOL.md, version 1.0):

  1. LOAD/SETUP    — model, tokenizer, sha256 of weights and corpus;
  2. TASKS         — three difficulty bands (indist/near/far), the truth is
                     recomputed by the library independently;
  3. PASSPORT      — machine-readable run passport (protocol freeze);
  4. OBSERVE       — 4 sampled (T=0.8) + 1 greedy generation per task;
                     raw observations WITHOUT labels;
  5. LABEL         — independent labeling on the tolerance grid;
  6. SPLIT         — 5 task-grouped folds (family x band stratification) +
                     leave-one-family-out; formal leakage checks;
  7. FIT/PREDICT   — A, A2, B, C, D, E with out-of-fold predictions and
                     honest Platt calibration on an internal held-out
                     subset of train;
  8. UNBLIND       — metrics with clustered CIs, paired differences, clustered
                     permutation tests, label permutation;
  9. REPORT        — report.json + REPORT.md, generated from the data.

Run:
  python3 neural_lab/science/experiments/run_h1.py \
      [--base DIR] [--out DIR] [--n-boot 2000] [--n-perm 2000]

Quick smoke run:  SMOKE=1 python3 ... (reduced volumes).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import zlib

import numpy as np


_SCIENCE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SCIENCE_DIR not in sys.path:
    sys.path.insert(0, os.path.dirname(_SCIENCE_DIR))  # neural_lab/

from science import PROTOCOL_VERSION  # noqa: E402
from science.data import (  # noqa: E402
    build_task_set,
    grouped_kfold,
    label_observation,
    leave_one_family_out,
)
from science.evaluation import (  # noqa: E402
    assert_no_task_overlap,
    average_precision,
    bootstrap_metric_ci,
    brier,
    cluster_bootstrap_delta,
    cluster_permutation_paired,
    duplicate_prompt_check,
    label_permutation_auc_p,
    log_loss,
    roc_auc,
)
from science.features import (  # noqa: E402
    as_matrix,
    build_feature_vector,
    feature_groups,
)
from science.instrumentation import capture_observation, load_model  # noqa: E402
from science.predictors import (  # noqa: E402
    ConstantRateBaseline,
    FamilyRateBaseline,
    LogisticRegression,
    ece,
    platt_apply,
    platt_fit,
)
from science.storage import (  # noqa: E402
    data_hash,
    environment_fingerprint,
    load_jsonl,
    save_json,
    save_jsonl,
    sha256_file,
)


TOLERANCE_GRID = [0.15, 0.25, 0.40]  # primary tolerance: 0.25
PRIMARY_TOL_KEY = "numeric_ok_025"
PRIMARY_METRIC = "roc_auc"
N_FOLDS = 5
TEMPERATURE = 0.8
N_TEMPERED = 4
N_GREEDY = 1
MAX_NEW_TOKENS = 28
L2 = 1.0
SEED = 42
N_INDIST = 96  # the pilot (SMOKE run) showed rare positives — expanding
N_NEAR = 16  # the choice was fixed in the protocol BEFORE the full run


def build_models(n_layers: int) -> dict[str, list[str] | None]:
    """Run predictors: name -> feature list (None — frequency baselines).

    Protocol v1.1: feature groups are built for the model's actual depth
    (for 4 layers identical to the frozen v1.0 scheme).
    """
    groups = feature_groups(n_layers)
    return {
        "A_constant_rate": None,  # base rate (the simplest baseline)
        "A2_family_rate": None,  # per-family base rate
        "B_confidence": groups["confidence"],
        "C_internal": groups["internal"],
        "D_combined": groups["combined"],
        "E_combined_minus_spectral": [n for n in groups["combined"] if n not in groups["spectral"]],
    }


def _task_seed(task_id: str, sample_idx: int) -> int:
    return (zlib.crc32(task_id.encode()) * 100 + sample_idx) % (2**31)


# ---------------------------------------------------------------------------
# Phases 1-3: loading, tasks, passport
# ---------------------------------------------------------------------------
def setup(args) -> dict:
    ctx = load_model(args.base)  # puts src/rmt_llm and lab_en on sys.path
    base = ctx["base"]
    corpus = os.path.join(base, "data", "formula_corpus_train.txt")
    if not os.path.exists(corpus):
        raise FileNotFoundError(corpus)
    tok = ctx["tokenizer"]
    n_layers = int(ctx["model"].config.n_layers)
    models = build_models(n_layers)  # per-run groups for the model depth

    tasks, exclusions, stats = build_task_set(
        corpus_path=corpus,
        max_prompt_tokens=ctx["model"].config.max_seq_len // 2,
        encode_fn=tok.encode,
        n_indist=8 if args.smoke else N_INDIST,
        n_near=4 if args.smoke else N_NEAR,
    )
    if len(tasks) < 10:
        raise RuntimeError(f"too few tasks: {len(tasks)}")
    dataset_hash = data_hash([t.to_dict() for t in tasks])
    passport = {
        "experiment_id": "error_prediction_rmt_v1",
        "hypothesis_id": "H1",
        "protocol_version": PROTOCOL_VERSION,
        "seed": SEED,
        "model_revision": ctx["weights_sha256"],
        "tokenizer_revision": ctx["bpe_sha256"],
        "model_config": ctx["config"],
        "dataset_hash": dataset_hash,
        "corpus_train_sha256": sha256_file(corpus),
        "split_scheme": "grouped_kfold_v1",
        "split_params": {"k": N_FOLDS, "stratify": "family_band", "seed": SEED},
        "feature_groups": feature_groups(n_layers),
        "predictors": [
            {
                "name": name,
                "features": feats,
                "l2": L2,
                "class": "ConstantRateBaseline" if feats is None else "LogisticRegression",
            }
            for name, feats in models.items()
        ],
        "primary_metric": PRIMARY_METRIC,
        "tolerance_grid": TOLERANCE_GRID,
        "generation": {
            "temperature": TEMPERATURE,
            "n_tempered": N_TEMPERED,
            "n_greedy": N_GREEDY,
            "max_new_tokens": MAX_NEW_TOKENS,
        },
        "environment": environment_fingerprint(),
        "exclusions": exclusions,
    }
    print(
        f"[setup] model {ctx['weights_sha256'][:12]}…  tasks={len(tasks)}  "
        f"dataset_hash={dataset_hash[:12]}…"
    )
    print(f"[setup] bands: {stats['by_band']}  families: {stats['by_family']}")
    return {
        "ctx": ctx,
        "tasks": tasks,
        "stats": stats,
        "passport": passport,
        "corpus": corpus,
        "base": base,
        "models": models,
        "n_layers": n_layers,
    }


# ---------------------------------------------------------------------------
# Phases 4-5: observation and labeling
# ---------------------------------------------------------------------------
def observe_and_label(ctx, tasks, _out_dir, smoke: bool) -> tuple[list[dict], list[dict]]:
    model, tok = ctx["model"], ctx["tokenizer"]
    observations, labels = [], []
    sample_plan = (
        [{"temperature": 0.0, "idx": 0}]
        if smoke
        else [{"temperature": TEMPERATURE, "idx": i} for i in range(N_TEMPERED)]
        + [{"temperature": 0.0, "idx": N_TEMPERED}]
    )
    t0 = time.time()
    n_planned = len(tasks) * len(sample_plan)
    done = 0
    # --- resume: incremental observation checkpoints (RMT_H1_OBS_RESUME=1).
    # Computationally identical to a single-shot run: tasks/order/seeds
    # are deterministic; observations (task_id, sample_id) are reproducible.
    obs_path = os.path.join(_out_dir, "observations.jsonl") if _out_dir else None
    if obs_path and os.environ.get("RMT_H1_OBS_RESUME") == "1" and os.path.exists(obs_path):
        from science.storage import load_jsonl

        prev = load_jsonl(obs_path)
        valid_prev_tasks = {t.task_id for t in tasks}
        observations = [o for o in prev if o.get("task_id") in valid_prev_tasks]
        done_pairs = {(o["task_id"], o["sample_id"]) for o in observations}
        done = len(done_pairs)
        print(f"[observe] resume: {done}/{n_planned} done, continuing", flush=True)
    else:
        done_pairs = set()

    def _checkpoint() -> None:
        if obs_path:
            from science.storage import save_jsonl

            save_jsonl(observations, obs_path)

    for task in tasks:
        for plan in sample_plan:
            seed = None if plan["temperature"] == 0.0 else _task_seed(task.task_id, plan["idx"])
            sid = f"s{plan['idx']}_{('greedy' if seed is None else 't08')}"
            if (task.task_id, sid) in done_pairs:
                continue
            try:
                obs = capture_observation(
                    model,
                    tok,
                    task.prompt,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=plan["temperature"],
                    seed=seed,
                )
                feats = build_feature_vector(obs)
            except ValueError as e:
                observations.append(
                    {
                        "task_id": task.task_id,
                        "sample_id": sid,
                        "excluded": True,
                        "reason": str(e)[:200],
                    }
                )
                done += 1
                continue
            observations.append(
                {
                    "task_id": task.task_id,
                    "sample_id": sid,
                    "band": task.band,
                    "family": task.family,
                    "temperature": plan["temperature"],
                    "seed": seed,
                    "generated_text": obs["generated_text"],
                    "n_new_tokens": obs["n_new_tokens"],
                    "elapsed_ms": obs["elapsed_ms"],
                    "features": feats,
                    "excluded": False,
                }
            )
            done += 1
            if done % 50 == 0:
                print(f"[observe] {done}/{n_planned} ({time.time() - t0:.0f}s)", flush=True)
                if done % 100 == 0:
                    _checkpoint()
    valid = [o for o in observations if not o.get("excluded")]
    for o in valid:
        task = next(t for t in tasks if t.task_id == o["task_id"])
        labels.append(
            {
                "task_id": o["task_id"],
                "sample_id": o["sample_id"],
                **label_observation(o["generated_text"], task.ground_truth, TOLERANCE_GRID),
            }
        )
    print(
        f"[observe] done {len(valid)}/{n_planned} observations "
        f"({len(observations) - len(valid)} excluded) in {time.time() - t0:.0f}s"
    )
    _checkpoint()  # final dump (also for resume)
    return observations, labels


# ---------------------------------------------------------------------------
# Phases 6-7: splits, fitting, out-of-fold predictions
# ---------------------------------------------------------------------------
def _fit_predict_fold(model_name, X_tr, y_tr, X_te, tasks_tr, _tasks_te, fam_tr=None, fam_te=None):
    """Returns (p_raw, p_cal) for the fold's test part.

    Platt calibration is honest: within a train fold the tasks are split into fit-
    and calibration parts (an 80/20 task-grouped share). For
    A2_family_rate, separate family labels fam_* are passed.
    """
    if model_name == "A_constant_rate":
        m = ConstantRateBaseline().fit(X_tr, y_tr)
        p = m.predict_proba(X_te)
        return p, p
    if model_name == "A2_family_rate":
        m = FamilyRateBaseline().fit(None, y_tr, families=fam_tr)
        p = m.predict_proba(X_te, families=fam_te)
        return p, p

    rng = np.random.default_rng(SEED + 1)
    uniq = np.unique(tasks_tr)
    perm = rng.permutation(len(uniq))
    n_cal = max(2, int(0.2 * len(uniq)))
    cal_tasks = set(uniq[perm[:n_cal]].tolist())
    is_cal = np.array([t in cal_tasks for t in tasks_tr])
    fit_idx, cal_idx = np.where(~is_cal)[0], np.where(is_cal)[0]

    # degenerate train (one class): logistic regression is undefined — honest
    # fallback with a constant base-rate prediction
    if len(np.unique(y_tr[fit_idx])) < 2:
        const = float(y_tr.mean())
        return (np.full(len(X_te), const), np.full(len(X_te), const))

    lr = LogisticRegression(l2=L2).fit(X_tr[fit_idx], y_tr[fit_idx])
    p_cal_set = lr.predict_proba(X_tr[cal_idx])
    a, b = platt_fit(p_cal_set, y_tr[cal_idx])
    p_raw = lr.predict_proba(X_te)
    return p_raw, platt_apply(p_raw, a, b)


def fit_predict(observations, labels, tasks, _out_dir, models) -> tuple[list[dict], dict]:
    valid = [o for o in observations if not o.get("excluded")]
    lab_map = {(l["task_id"], l["sample_id"]): l for l in labels}
    task_map = {t.task_id: t for t in tasks}

    task_ids = sorted({o["task_id"] for o in valid})
    keys = [(task_map[t].family, task_map[t].band) for t in task_ids]
    folds = grouped_kfold(task_ids, keys, k=N_FOLDS, seed=SEED)

    task_family = {t: task_map[t].family for t in task_ids}

    # formal leakage checks over the whole split scheme
    for f, test_ids in folds.items():
        train_ids = [t for t in task_ids if t not in set(test_ids)]
        assert_no_task_overlap(train_ids, test_ids, tag=f"fold{f}")
        duplicate_prompt_check(
            [task_map[t].prompt for t in train_ids],
            [task_map[t].prompt for t in test_ids],
            tag=f"fold{f}",
        )

    y = np.array(
        [lab_map[(o["task_id"], o["sample_id"])][PRIMARY_TOL_KEY] for o in valid], dtype=np.float64
    )
    obs_task = np.array([o["task_id"] for o in valid])
    obs_family = np.array([task_family[t] for t in obs_task])
    obs_fold = np.empty(len(valid), dtype=int)
    fold_of_task = {t: f for f, ids in folds.items() for t in ids}
    for i, t in enumerate(obs_task):
        obs_fold[i] = fold_of_task[t]

    # Fit targets: the primary (numeric_ok_025) + the pre-declared
    # secondary (format_ok) and the sensitivity grid (015/040). Each
    # target is fitted SEPARATELY — reusing one target's predictions
    # for another target is forbidden.
    def _target(key: str) -> np.ndarray:
        return np.array(
            [lab_map[(o["task_id"], o["sample_id"])][key] for o in valid], dtype=np.float64
        )

    targets = {
        "": y,  # primary target
        "_fmt": _target("format_ok"),  # secondary target
        "_015": _target("numeric_ok_015"),  # sensitivity
        "_040": _target("numeric_ok_040"),  # sensitivity
    }

    # prediction record index (task_id, sample_id) -> dict
    rec_index: dict[tuple[str, str], dict] = {}
    for pos, o in enumerate(valid):
        rec_index[(o["task_id"], o["sample_id"])] = {
            "task_id": o["task_id"],
            "sample_id": o["sample_id"],
            "band": o["band"],
            "family": o["family"],
            "fold": -1,
            "y_true": int(y[pos]),  # primary target
        }

    X_cache = {
        name: (as_matrix([o["features"] for o in valid], feats) if feats else None)
        for name, feats in models.items()
    }
    for suffix, yy in targets.items():
        for f in sorted(folds):
            test_mask = obs_fold == f
            train_mask = ~test_mask
            for name in models:
                X_all = X_cache[name]
                X_tr = (
                    X_all[train_mask] if X_all is not None else np.zeros((int(train_mask.sum()), 1))
                )
                X_te = (
                    X_all[test_mask] if X_all is not None else np.zeros((int(test_mask.sum()), 1))
                )
                p_raw, p_cal = _fit_predict_fold(
                    name,
                    X_tr,
                    yy[train_mask],
                    X_te,
                    obs_task[train_mask],
                    obs_task[test_mask],
                    fam_tr=obs_family[train_mask],
                    fam_te=obs_family[test_mask],
                )
                tidx = np.where(test_mask)[0]
                for j, oidx in enumerate(tidx):
                    o = valid[oidx]
                    rec = rec_index[(o["task_id"], o["sample_id"])]
                    rec["fold"] = int(f)
                    rec[name + suffix] = float(p_raw[j])
                    rec[name + suffix + "_cal"] = float(p_cal[j])
            if suffix == "":
                print(f"[fit] fold {f}: train={int(train_mask.sum())} test={int(test_mask.sum())}")

    from science.data import family_overlap_report

    fold_report = {
        "folds": {str(f): len(v) for f, v in folds.items()},
        "task_family": task_family,
        "family_overlap": family_overlap_report(folds, task_family),
    }
    records = [rec_index[(o["task_id"], o["sample_id"])] for o in valid]
    return records, fold_report


def lofo_predictions(observations, labels, tasks, models) -> dict:
    """leave-one-family-out: point AUCs of each model on the held-out
    family (a strict generalization check; small n — points only)."""
    valid = [o for o in observations if not o.get("excluded")]
    lab_map = {(l["task_id"], l["sample_id"]): l for l in labels}
    task_map = {t.task_id: t for t in tasks}
    y = np.array([lab_map[(o["task_id"], o["sample_id"])][PRIMARY_TOL_KEY] for o in valid])
    obs_task = np.array([o["task_id"] for o in valid])
    fams = {t: task_map[t].family for t in np.unique(obs_task)}
    obs_family = np.array([fams[t] for t in obs_task])
    splits = leave_one_family_out(list(fams), [fams[t] for t in fams])
    X_cache = {
        name: (as_matrix([o["features"] for o in valid], feats) if feats else None)
        for name, feats in models.items()
    }
    out = {}
    for fam, sp in splits.items():
        test_mask = np.isin(obs_task, sp["test"])
        train_mask = ~test_mask
        row = {"n_test_obs": int(test_mask.sum()), "n_pos": float(y[test_mask].sum())}
        for name in models:
            X_all = X_cache[name]
            X_tr = X_all[train_mask] if X_all is not None else np.zeros((int(train_mask.sum()), 1))
            X_te = X_all[test_mask] if X_all is not None else np.zeros((int(test_mask.sum()), 1))
            try:
                p_raw, _ = _fit_predict_fold(
                    name,
                    X_tr,
                    y[train_mask],
                    X_te,
                    obs_task[train_mask],
                    obs_task[test_mask],
                    fam_tr=obs_family[train_mask],
                    fam_te=obs_family[test_mask],
                )
                row[name] = round(float(roc_auc(y[test_mask], p_raw)), 4)
            except (ValueError, RuntimeError):
                row[name] = None
        out[fam] = row
    return out


# ---------------------------------------------------------------------------
# Phase 8: unblinding — metrics and tests
# ---------------------------------------------------------------------------
def unblind(records, fold_report, lofo, out_dir, n_boot, n_perm, models) -> dict:
    y = np.array([r["y_true"] for r in records], dtype=np.float64)
    task_ids = np.array([r["task_id"] for r in records])
    strata = np.array([f"{r['family']}|{r['band']}" for r in records])
    base_rate = float(y.mean())
    n_pos = int(y.sum())
    print(f"[unblind] n={len(records)} positives={n_pos} base_rate={base_rate:.3f}")

    model_names = [m for m in models if m.startswith(("A", "B", "C", "D", "E"))]
    metrics: dict[str, dict] = {}
    for name in model_names:
        p = np.array([r[name] for r in records])
        key = name + "_cal"
        p_cal = np.array([r[key] for r in records])
        metrics[name] = {
            "roc_auc": bootstrap_metric_ci(y, p, task_ids, roc_auc, n_boot=n_boot),
            "average_precision": bootstrap_metric_ci(
                y, p, task_ids, average_precision, n_boot=n_boot
            ),
            "brier": brier(y, p),
            "log_loss": log_loss(y, p),
            "ece_raw": ece(y, p),
            "ece_platt": ece(y, p_cal),
        }
        if n_pos > 0 and (1 - base_rate) > 0:
            print(
                f"  {name:28s} AUC={metrics[name]['roc_auc']['point']:.3f} "
                f"[{metrics[name]['roc_auc']['ci_low']:.3f}, "
                f"{metrics[name]['roc_auc']['ci_high']:.3f}]"
            )

    comparisons = {}
    if n_pos > 0 and n_pos < len(y):
        for a, bmn in (
            ("D_combined", "B_confidence"),
            ("D_combined", "C_internal"),
            ("C_internal", "B_confidence"),
        ):
            pa = np.array([r[a] for r in records])
            pb = np.array([r[bmn] for r in records])
            comparisons[f"{a}_minus_{bmn}"] = {
                "delta_auc": cluster_bootstrap_delta(y, pa, pb, task_ids, n_boot=n_boot),
                "permutation": cluster_permutation_paired(y, pa, pb, task_ids, n_perm=n_perm),
            }
            d = comparisons[f"{a}_minus_{bmn}"]
            print(
                f"  ΔAUC {a} − {bmn}: {d['delta_auc']['point']:+.3f} "
                f"[{d['delta_auc']['ci_low']:+.3f}, {d['delta_auc']['ci_high']:+.3f}] "
                f"perm p={d['permutation']['p_value']:.3f}"
            )
        pd_ = np.array([r["D_combined"] for r in records])
        pb_ = np.array([r["B_confidence"] for r in records])
        label_perm = {
            "D_combined": label_permutation_auc_p(
                y, pd_, task_ids, strata, n_perm=min(n_perm, 500)
            ),
            "B_confidence": label_permutation_auc_p(
                y, pb_, task_ids, strata, n_perm=min(n_perm, 500)
            ),
        }
    else:
        label_perm = {}

    # secondary endpoint: format_ok — a separate fit (the _fmt suffix)
    lab_map = {
        (l["task_id"], l["sample_id"]): l for l in load_jsonl(os.path.join(out_dir, "labels.jsonl"))
    }
    y_fmt = np.array(
        [lab_map[(r["task_id"], r["sample_id"])]["format_ok"] for r in records], dtype=np.float64
    )
    secondary = {}
    if 0 < y_fmt.sum() < len(y_fmt):
        for name in (
            "A_constant_rate",
            "A2_family_rate",
            "B_confidence",
            "C_internal",
            "D_combined",
        ):
            p = np.array([r[name + "_fmt"] for r in records])
            p_cal = np.array([r[name + "_fmt_cal"] for r in records])
            secondary[name] = {
                "roc_auc": bootstrap_metric_ci(y_fmt, p, task_ids, roc_auc, n_boot=n_boot),
                "brier": round(brier(y_fmt, p), 4),
                "ece_raw": round(ece(y_fmt, p), 4),
                "ece_platt": round(ece(y_fmt, p_cal), 4),
            }
            print(
                f"  [fmt] {name:22s} AUC={secondary[name]['roc_auc']['point']:.3f} "
                f"[{secondary[name]['roc_auc']['ci_low']:.3f}, "
                f"{secondary[name]['roc_auc']['ci_high']:.3f}]"
            )
        d_fb = cluster_bootstrap_delta(
            y_fmt,
            np.array([r["D_combined_fmt"] for r in records]),
            np.array([r["B_confidence_fmt"] for r in records]),
            task_ids,
            n_boot=n_boot,
        )
        pm_fb = cluster_permutation_paired(
            y_fmt,
            np.array([r["D_combined_fmt"] for r in records]),
            np.array([r["B_confidence_fmt"] for r in records]),
            task_ids,
            n_perm=n_perm,
        )
        secondary["delta_D_minus_B"] = {"delta_auc": d_fb, "permutation": pm_fb}

    # tolerance sensitivity — predictions fitted FOR EACH target
    sensitivity = {}
    for tol in TOLERANCE_GRID:
        key = f"numeric_ok_{round(tol * 100):03d}"
        yt = np.array(
            [lab_map[(r["task_id"], r["sample_id"])][key] for r in records], dtype=np.float64
        )
        suffix = "" if key == PRIMARY_TOL_KEY else "_" + key.split("_")[-1]
        if yt.sum() == 0 or yt.sum() == len(yt):
            sensitivity[key] = {
                "base_rate": float(yt.mean()),
                "auc": None,
                "note": "degenerate: single class",
            }
            continue
        row = {"base_rate": round(float(yt.mean()), 4), "n_pos": int(yt.sum())}
        for name in ("B_confidence", "C_internal", "D_combined"):
            p = np.array([r[name + suffix] for r in records])
            row[name] = round(float(roc_auc(yt, p)), 4)
        sensitivity[key] = row

    return {
        "n_observations": len(records),
        "n_tasks": len(np.unique(task_ids)),
        "base_rate": round(base_rate, 4),
        "n_positives": n_pos,
        "metrics": metrics,
        "comparisons": comparisons,
        "label_permutation": label_perm,
        "secondary_format_ok": secondary,
        "tolerance_sensitivity": sensitivity,
        "lofo": lofo,
        "fold_report": fold_report,
    }


# ---------------------------------------------------------------------------
# Phase 9: the report (generated FROM the data)
# ---------------------------------------------------------------------------
def _fmt3(v) -> str:
    return "—" if v is None else f"{v:.3f}"


def render_report_md(report: dict, passport: dict) -> str:  # noqa: C901
    m = report["metrics"]
    lines = [
        "# H1 — run result (error_prediction_rmt_v1)",
        "",
        f"Protocol v{passport['protocol_version']} · model `{passport['model_revision'][:12]}…` · "
        f"dataset `{passport['dataset_hash'][:12]}…` · seed {passport['seed']}",
        "",
        f"Observations: **{report['n_observations']}** (tasks {report['n_tasks']}), "
        f"positives: **{report['n_positives']}** (base rate "
        f"{report['base_rate']:.1%}). Primary endpoint: `{PRIMARY_TOL_KEY}`, "
        f"metric: `{PRIMARY_METRIC}`, CI — task-clustered bootstrap.",
        "",
        "## Main table (out-of-fold)",
        "",
        "| Model | ROC-AUC [95% CI] | PR-AUC | Brier | ECE raw→Platt |",
        "|---|---|---|---|---|",
    ]
    for name, mm in m.items():
        r = mm["roc_auc"]
        lines.append(
            f"| {name} | {_fmt3(r['point'])} [{_fmt3(r['ci_low'])}, {_fmt3(r['ci_high'])}] "
            f"| {_fmt3(mm['average_precision']['point'])} | {mm['brier']:.3f} "
            f"| {mm['ece_raw']:.3f} → {mm['ece_platt']:.3f} |"
        )
    lines += ["", "## Key comparisons (ΔAUC, clustered tests)", ""]
    for key, c in report["comparisons"].items():
        d, pm = c["delta_auc"], c["permutation"]
        lines.append(
            f"- **{key}**: {d['point']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}], "
            f"permutation p = {pm['p_value']:.4f}"
        )
    if report.get("label_permutation"):
        lines += ["", "## Label permutation (null hypothesis)", ""]
        for name, lp in report["label_permutation"].items():
            lines.append(
                f"- {name}: observed AUC {lp['observed_auc']:.3f}, "
                f"p = {lp['p_value']:.4f} "
                f"(permutable strata {lp.get('n_strata_permutable', '?')}/"
                f"{lp.get('n_strata_total', '?')})"
            )
            if lp.get("note") != "ok":
                lines.append(f"  - warning: {lp['note']}")
    if report.get("secondary_format_ok"):
        lines += [
            "",
            "## Secondary endpoint: format_ok — the “Utility Trap” (separate fit, out-of-fold)",
            "",
            "| Model | ROC-AUC [95% CI] | Brier | ECE raw→Platt |",
            "|---|---|---|---|",
        ]
        for name, v in report["secondary_format_ok"].items():
            if name.startswith("delta"):
                d, pm = v["delta_auc"], v["permutation"]
                lines.append("")
                lines.append(
                    f"**{name}**: {d['point']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}], "
                    f"permutation p = {pm['p_value']:.4f}"
                )
            else:
                r = v["roc_auc"]
                lines.append(
                    f"| {name} | {r['point']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}] "
                    f"| {v['brier']:.3f} | {v['ece_raw']:.3f} → {v['ece_platt']:.3f} |"
                )
    lines += [
        "",
        "## Tolerance sensitivity (each target fitted separately)",
        "",
        "| Tolerance | base rate | B | C | D |",
        "|---|---|---|---|---|",
    ]
    for key, row in report["tolerance_sensitivity"].items():
        if "B_confidence" not in row:
            lines.append(f"| {key} | {row['base_rate']:.3f} | — degenerate class — | | |")
        else:
            lines.append(
                f"| {key} | {row['base_rate']:.3f} ({row.get('n_pos', '?')}+) "
                f"| {row.get('B_confidence', '—')} | {row.get('C_internal', '—')} "
                f"| {row.get('D_combined', '—')} |"
            )
    lines += [
        "",
        "## Leave-one-family-out (generalization to an unseen family)",
        "",
        "| Family | n_test | n_pos | A | B | C | D |",
        "|---|---|---|---|---|---|---|",
    ]
    for fam, row in report["lofo"].items():
        fmt = lambda v: "—" if v is None else f"{v:.3f}"
        lines.append(
            f"| {fam} | {row['n_test_obs']} | {row['n_pos']:.0f} "
            f"| {fmt(row.get('A_constant_rate'))} | {fmt(row.get('B_confidence'))} "
            f"| {fmt(row.get('C_internal'))} | {fmt(row.get('D_combined'))} |"
        )
    lines += ["", "## Verdict on the success criterion (automatically from the protocol)", ""]
    n_pos = report["n_positives"]
    if n_pos < 20:
        lines.append(
            f"Primary endpoint UNDERPOWERED: {n_pos} positives (< 20) — "
            "any p-values on the primary endpoint should be read with caution."
        )
    comp = report["comparisons"].get("D_combined_minus_B_confidence")
    if comp:
        d, pm = comp["delta_auc"], comp["permutation"]
        ci_excl0 = d["ci_low"] > 0
        psig = pm["p_value"] < 0.05
        if ci_excl0 and psig:
            verdict = "H1 SUPPORTED by the protocol's criterion (CI > 0 and p < 0.05)."
        elif ci_excl0 and not psig:
            verdict = (
                "H1 NOT STRICTLY SUPPORTED (the CI excludes 0, but "
                f"the permutation p = {pm['p_value']:.3f} >= 0.05): "
                "the direction is positive, power is insufficient — "
                "more data / a larger model is required."
            )
        else:
            verdict = "H1 NOT SUPPORTED: the CI includes 0."
        lines.append(
            f"ΔAUC(D−B) = {d['point']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}], "
            f"p = {pm['p_value']:.4f} → **{verdict}**"
        )
    lines.append("")

    lines += [
        "",
        "## Honest limitations",
        "",
        "- a single run of a single seeded training; the weights are not committed "
        "(see reproduction in EXPERIMENT_PROTOCOL.md);",
        "- repeated generations of one task are correlated — all CIs and tests "
        "are clustered by task;",
        "- A_constant_rate below 0.5 is an artifact of between-fold variation of the base "
        "rate under a constant prediction; read jointly with A2;",
        "- the small number of positive labels limits power; a negative "
        "or inconclusive outcome is still a result and is recorded as it is.",
        "",
        "Artifacts: passport.json · tasks.json · observations.jsonl · "
        "labels.jsonl · predictions.jsonl · report.json · REPORT.md",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="H1: error prediction from internal features")
    ap.add_argument("--base", default=None, help="research/tinygpt_formula base dir")
    ap.add_argument("--out", default=os.path.join(_SCIENCE_DIR, "results", "h1_v1"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    args.smoke = args.smoke or os.environ.get("SMOKE") == "1"
    if args.smoke:
        args.n_boot, args.n_perm = 200, 200

    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)
    setup_ctx = setup(args)
    ctx, tasks = setup_ctx["ctx"], setup_ctx["tasks"]
    passport = setup_ctx["passport"]
    passport["passport_sha256"] = save_json(passport, os.path.join(args.out, "passport.json"))
    save_json(
        {
            "tasks": [t.to_dict() for t in tasks],
            "stats": setup_ctx["stats"],
            "exclusions": passport["exclusions"],
        },
        os.path.join(args.out, "tasks.json"),
    )

    observations, labels = observe_and_label(ctx, tasks, args.out, args.smoke)
    n_valid = sum(1 for o in observations if not o.get("excluded"))
    if n_valid == 0:
        from collections import Counter

        reasons = Counter(o.get("reason", "?")[:120] for o in observations)
        raise RuntimeError(
            "0 valid observations — check the instrumentation's compatibility with "
            f"the model configuration. Top exclusion reasons: {dict(reasons.most_common(3))}"
        )
    if n_valid < 0.5 * len(observations):
        from collections import Counter

        reasons = Counter(o.get("reason", "?")[:80] for o in observations if o.get("excluded"))
        print(
            f"[warn] excluded {len(observations) - n_valid}/{len(observations)} observations; "
            f"top reasons: {dict(reasons.most_common(3))}"
        )
    save_jsonl(observations, os.path.join(args.out, "observations.jsonl"))
    save_jsonl(labels, os.path.join(args.out, "labels.jsonl"))

    records, fold_report = fit_predict(observations, labels, tasks, args.out, setup_ctx["models"])
    save_jsonl(records, os.path.join(args.out, "predictions.jsonl"))

    lofo = lofo_predictions(observations, labels, tasks, setup_ctx["models"])
    report = unblind(
        records, fold_report, lofo, args.out, args.n_boot, args.n_perm, setup_ctx["models"]
    )
    report["wall_minutes"] = round((time.time() - t0) / 60, 1)
    save_json(report, os.path.join(args.out, "report.json"))
    md = render_report_md(report, passport)
    with open(os.path.join(args.out, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[done] results in {args.out} ({report['wall_minutes']} min)")
    print(md)


if __name__ == "__main__":
    main()
