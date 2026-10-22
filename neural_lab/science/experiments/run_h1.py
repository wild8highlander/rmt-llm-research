"""
experiments.run_h1 — Эксперимент № 1: предсказание ошибки LLM по внутренним
признакам (H1), воспроизводимый конвейер.

ГИПОТЕЗА H1: спектральные характеристики внутренних активаций содержат
информацию об ошибках модели СВЕРХ той, которую дают стандартные
показатели уверенности.

Протокол зафиксирован ДО запуска (EXPERIMENT_PROTOCOL.md, версия 1.0):

  1. LOAD/SETUP    — модель, токенизатор, sha256 весов и корпуса;
  2. TASKS         — три полосы сложности (indist/near/far), истина
                     пересчитывается библиотекой независимо;
  3. PASSPORT      — машиночитаемый паспорт запуска (заморозка протокола);
  4. OBSERVE       — 4 сэмплированных (T=0.8) + 1 жадная генерация на
                     задачу; сырые наблюдения БЕЗ меток;
  5. LABEL         — независимая разметка по сетке допусков;
  6. SPLIT         — 5 фолдов по задачам (стратификация family x band) +
                     leave-one-family-out; формальные проверки утечек;
  7. FIT/PREDICT   — A, A2, B, C, D, E с out-of-fold предсказаниями и
                     честной Платт-калибровкой на внутреннем отложенном
                     подмножестве train;
  8. UNBLIND       — метрики с кластерными CI, парные разницы, кластерные
                     перестановочные тесты, перестановка меток;
  9. REPORT        — report.json + REPORT.md, сгенерированный из данных.

Запуск:
  python3 neural_lab/science/experiments/run_h1.py \
      [--base DIR] [--out DIR] [--n-boot 2000] [--n-perm 2000]

Быстрый smoke-прогон:  SMOKE=1 python3 ... (уменьшенные объёмы).
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


TOLERANCE_GRID = [0.15, 0.25, 0.40]  # первичный допуск: 0.25
PRIMARY_TOL_KEY = "numeric_ok_025"
PRIMARY_METRIC = "roc_auc"
N_FOLDS = 5
TEMPERATURE = 0.8
N_TEMPERED = 4
N_GREEDY = 1
MAX_NEW_TOKENS = 28
L2 = 1.0
SEED = 42
N_INDIST = 96  # пилот (SMOKE-прогон) показал редкие позитивы — расширяем
N_NEAR = 16  # выбор зафиксирован в протоколе ДО полного прогона

def build_models(n_layers: int) -> dict[str, list[str] | None]:
    """Предикторы прогона: имя -> список признаков (None — частотные baseline).

    Протокол v1.1: группы признаков строятся под фактическую глубину модели
    (для 4 слоёв идентично замороженной схеме v1.0).
    """
    groups = feature_groups(n_layers)
    return {
        "A_constant_rate": None,  # базовая частота (простейший baseline)
        "A2_family_rate": None,  # базовая частота по семейству
        "B_confidence": groups["confidence"],
        "C_internal": groups["internal"],
        "D_combined": groups["combined"],
        "E_combined_minus_spectral": [
            n for n in groups["combined"] if n not in groups["spectral"]
        ],
    }


def _task_seed(task_id: str, sample_idx: int) -> int:
    return (zlib.crc32(task_id.encode()) * 100 + sample_idx) % (2**31)


# ---------------------------------------------------------------------------
# Фазы 1-3: загрузка, задачи, паспорт
# ---------------------------------------------------------------------------
def setup(args) -> dict:
    ctx = load_model(args.base)  # подключает src/rmt_llm и lab_en в sys.path
    base = ctx["base"]
    corpus = os.path.join(base, "data", "formula_corpus_train.txt")
    if not os.path.exists(corpus):
        raise FileNotFoundError(corpus)
    tok = ctx["tokenizer"]
    n_layers = int(ctx["model"].config.n_layers)
    models = build_models(n_layers)  # per-run группы под глубину модели

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
# Фазы 4-5: наблюдения и разметка
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
    # --- resume: инкрементальный чекпоинт наблюдений (RMT_H1_OBS_RESUME=1).
    # Вычислительно идентично однократному прогону: задачи/порядок/сиды
    # детерминированы, наблюдения (task_id, sample_id) воспроизводимы.
    obs_path = os.path.join(_out_dir, "observations.jsonl") if _out_dir else None
    if obs_path and os.environ.get("RMT_H1_OBS_RESUME") == "1" and os.path.exists(obs_path):
        from science.storage import load_jsonl

        prev = load_jsonl(obs_path)
        valid_prev_tasks = {t.task_id for t in tasks}
        observations = [o for o in prev if o.get("task_id") in valid_prev_tasks]
        done_pairs = {(o["task_id"], o["sample_id"]) for o in observations}
        done = len(done_pairs)
        print(f"[observe] resume: {done}/{n_planned} готово, продолжаю", flush=True)
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
    _checkpoint()  # финальный дамп (в т.ч. для resume)
    return observations, labels


# ---------------------------------------------------------------------------
# Фазы 6-7: разбиения, фит, out-of-fold предсказания
# ---------------------------------------------------------------------------
def _fit_predict_fold(model_name, X_tr, y_tr, X_te, tasks_tr, _tasks_te, fam_tr=None, fam_te=None):
    """Возвращает (p_raw, p_cal) для тестовой части фолда.

    Платт-калибровка честная: внутри train-фолда задачи делятся на фит-
    и калибровочную части (групповая доля 80/20 по задачам). Для
    A2_family_rate передаются отдельные метки семейств fam_*.
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

    # вырожденный train (один класс): логрегрессия неопределима — честный
    # fallback с константным предсказанием базовой частоты
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

    # формальные проверки утечек на всей схеме разбиения
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

    # Цели фита: первичная (numeric_ok_025) + заранее объявленные
    # вторичная (format_ok) и сетка чувствительности (015/040). Каждая
    # цель фитируется ОТДЕЛЬНО — предсказания одной цели запрещено
    # переиспользовать для другой.
    def _target(key: str) -> np.ndarray:
        return np.array(
            [lab_map[(o["task_id"], o["sample_id"])][key] for o in valid], dtype=np.float64
        )

    targets = {
        "": y,  # первичная цель
        "_fmt": _target("format_ok"),  # вторичная цель
        "_015": _target("numeric_ok_015"),  # sensitivity
        "_040": _target("numeric_ok_040"),  # sensitivity
    }

    # индекс записей предсказаний (task_id, sample_id) -> dict
    rec_index: dict[tuple[str, str], dict] = {}
    for pos, o in enumerate(valid):
        rec_index[(o["task_id"], o["sample_id"])] = {
            "task_id": o["task_id"],
            "sample_id": o["sample_id"],
            "band": o["band"],
            "family": o["family"],
            "fold": -1,
            "y_true": int(y[pos]),  # первичная цель
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
    """leave-one-family-out: точечные AUC каждой модели на отложенном
    семействе (жёсткая проверка обобщения; малые n — только точки)."""
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
# Фаза 8: раскрытие — метрики и тесты
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

    # secondary endpoint: format_ok — отдельный фит (суффикс _fmt)
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

    # sensitivity по допускам — предсказания, фитированные НА КАЖДУЮ цель
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
# Фаза 9: отчёт (генерируется ИЗ данных)
# ---------------------------------------------------------------------------
def _fmt3(v) -> str:
    return "—" if v is None else f"{v:.3f}"


def render_report_md(report: dict, passport: dict) -> str:  # noqa: C901
    m = report["metrics"]
    lines = [
        "# H1 — результат запуска (error_prediction_rmt_v1)",
        "",
        f"Протокол v{passport['protocol_version']} · модель `{passport['model_revision'][:12]}…` · "
        f"датасет `{passport['dataset_hash'][:12]}…` · seed {passport['seed']}",
        "",
        f"Наблюдений: **{report['n_observations']}** (задач {report['n_tasks']}), "
        f"положительных: **{report['n_positives']}** (базовая частота "
        f"{report['base_rate']:.1%}). Первичная цель: `{PRIMARY_TOL_KEY}`, "
        f"метрика: `{PRIMARY_METRIC}`, CI — кластерный бутстреп по задачам.",
        "",
        "## Основная таблица (out-of-fold)",
        "",
        "| Модель | ROC-AUC [95% CI] | PR-AUC | Brier | ECE raw→Platt |",
        "|---|---|---|---|---|",
    ]
    for name, mm in m.items():
        r = mm["roc_auc"]
        lines.append(
            f"| {name} | {_fmt3(r['point'])} [{_fmt3(r['ci_low'])}, {_fmt3(r['ci_high'])}] "
            f"| {_fmt3(mm['average_precision']['point'])} | {mm['brier']:.3f} "
            f"| {mm['ece_raw']:.3f} → {mm['ece_platt']:.3f} |"
        )
    lines += ["", "## Ключевые сравнения (ΔAUC, кластерные тесты)", ""]
    for key, c in report["comparisons"].items():
        d, pm = c["delta_auc"], c["permutation"]
        lines.append(
            f"- **{key}**: {d['point']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}], "
            f"перестановочный p = {pm['p_value']:.4f}"
        )
    if report.get("label_permutation"):
        lines += ["", "## Перестановка меток (нулевая гипотеза)", ""]
        for name, lp in report["label_permutation"].items():
            lines.append(
                f"- {name}: наблюдаемый AUC {lp['observed_auc']:.3f}, "
                f"p = {lp['p_value']:.4f} "
                f"(переставляемых страт {lp.get('n_strata_permutable', '?')}/"
                f"{lp.get('n_strata_total', '?')})"
            )
            if lp.get("note") != "ok":
                lines.append(f"  - внимание: {lp['note']}")
    if report.get("secondary_format_ok"):
        lines += [
            "",
            "## Вторичная цель: format_ok — «Ловушка Полезности» (отдельный фит, out-of-fold)",
            "",
            "| Модель | ROC-AUC [95% CI] | Brier | ECE raw→Platt |",
            "|---|---|---|---|",
        ]
        for name, v in report["secondary_format_ok"].items():
            if name.startswith("delta"):
                d, pm = v["delta_auc"], v["permutation"]
                lines.append("")
                lines.append(
                    f"**{name}**: {d['point']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}], "
                    f"перестановочный p = {pm['p_value']:.4f}"
                )
            else:
                r = v["roc_auc"]
                lines.append(
                    f"| {name} | {r['point']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}] "
                    f"| {v['brier']:.3f} | {v['ece_raw']:.3f} → {v['ece_platt']:.3f} |"
                )
    lines += [
        "",
        "## Чувствительность к допуску (каждая цель фитована отдельно)",
        "",
        "| Допуск | base rate | B | C | D |",
        "|---|---|---|---|---|",
    ]
    for key, row in report["tolerance_sensitivity"].items():
        if "B_confidence" not in row:
            lines.append(f"| {key} | {row['base_rate']:.3f} | — вырожденный класс — | | |")
        else:
            lines.append(
                f"| {key} | {row['base_rate']:.3f} ({row.get('n_pos', '?')}+) "
                f"| {row.get('B_confidence', '—')} | {row.get('C_internal', '—')} "
                f"| {row.get('D_combined', '—')} |"
            )
    lines += [
        "",
        "## Leave-one-family-out (обобщение на незнакомое семейство)",
        "",
        "| Семейство | n_test | n_pos | A | B | C | D |",
        "|---|---|---|---|---|---|---|",
    ]
    for fam, row in report["lofo"].items():
        fmt = lambda v: "—" if v is None else f"{v:.3f}"
        lines.append(
            f"| {fam} | {row['n_test_obs']} | {row['n_pos']:.0f} "
            f"| {fmt(row.get('A_constant_rate'))} | {fmt(row.get('B_confidence'))} "
            f"| {fmt(row.get('C_internal'))} | {fmt(row.get('D_combined'))} |"
        )
    lines += ["", "## Вердикт по критерию успеха (автоматически из протокола)", ""]
    n_pos = report["n_positives"]
    if n_pos < 20:
        lines.append(
            f"Первичная цель UNDERPOWERED: позитивов {n_pos} (< 20) — "
            "любые p-value по первичной цели следует читать с осторожностью."
        )
    comp = report["comparisons"].get("D_combined_minus_B_confidence")
    if comp:
        d, pm = comp["delta_auc"], comp["permutation"]
        ci_excl0 = d["ci_low"] > 0
        psig = pm["p_value"] < 0.05
        if ci_excl0 and psig:
            verdict = "H1 ПОДДЕРЖАНА по критерию протокола (CI > 0 и p < 0.05)."
        elif ci_excl0 and not psig:
            verdict = (
                "H1 НЕ ПОДТВЕРЖДЕНА строго (CI не накрывает 0, но "
                f"перестановочный p = {pm['p_value']:.3f} >= 0.05): "
                "направление положительное, мощность недостаточна — "
                "требуются данные большего объёма / модель большего размера."
            )
        else:
            verdict = "H1 НЕ ПОДДЕРЖАНА: CI накрывает 0."
        lines.append(
            f"ΔAUC(D−B) = {d['point']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}], "
            f"p = {pm['p_value']:.4f} → **{verdict}**"
        )
    lines.append("")

    lines += [
        "",
        "## Честные ограничения",
        "",
        "- один запуск одного seeded-прогона обучения; веса не коммитятся "
        "(см. воспроизведение в EXPERIMENT_PROTOCOL.md);",
        "- повторные генерации одной задачи коррелированы — все CI и тесты "
        "кластеризованы по задачам;",
        "- A_constant_rate ниже 0.5 — артефакт межфолдовой вариации базовой "
        "частоты при константном предсказании; смотреть совместно с A2;",
        "- малое число положительных меток ограничивает мощность; отрицательный "
        "или неубедительный результат — тоже результат и фиксируется как есть.",
        "",
        "Артефакты: passport.json · tasks.json · observations.jsonl · "
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
            "0 валидных наблюдений — проверьте совместимость instrumentation с "
            f"конфигурацией модели. Топ причин исключений: {dict(reasons.most_common(3))}"
        )
    if n_valid < 0.5 * len(observations):
        from collections import Counter

        reasons = Counter(o.get("reason", "?")[:80] for o in observations if o.get("excluded"))
        print(f"[warn] исключено {len(observations) - n_valid}/{len(observations)} наблюдений; "
              f"топ причин: {dict(reasons.most_common(3))}")
    save_jsonl(observations, os.path.join(args.out, "observations.jsonl"))
    save_jsonl(labels, os.path.join(args.out, "labels.jsonl"))

    records, fold_report = fit_predict(
        observations, labels, tasks, args.out, setup_ctx["models"]
    )
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
