# -*- coding: utf-8 -*-
"""
scaling_experiment.py — лестница масштабирования TinyGPT v3 + A/B-тест
«влияние формул репозитория на галлюцинации».

Исследовательские вопросы:
  1. Можно ли снизить галлюцинации, обучая модель на ПРАВИЛЬНЫХ формулах
     репозитория (vs ложной арифметике той же формы)?
  2. Как этот эффект меняется с ростом числа параметров?

Дизайн (для каждого размера S/M/L/XL):
  arm formula  — обучение на истинном корпусе (data/formula_corpus_*.txt)
  arm control  — обучение на контрольном корпусе (та же форма, ложная
                 арифметика; build_control_corpus.py)
  arm untrained— свежая модель того же размера без обучения (точка отсчёта,
                 оценивается автоматически, без обучения)

ДВА бенчмарка (методология scripts/evaluate_model.py, ±15% допуск):
  HELD-OUT (37 свежих задач, параметров не было в корпусе):
      numeric_score                 — доля ответов с верными числами
      confident_hallucination_rate  — доля ОФОРМЛЕННЫХ ответов с НЕВЕРНЫМИ
                                      числами («Ловушка Полезности» монографии)
  IN-DISTRIBUTION (до 60 пар из самого корпуса; правая часть корпуса formula —
  истинные значения, сгенерированные библиотекой):
      Здесь у модели ЕСТЬ знание (пары видены при обучении). Ожидание:
      formula-рука воспроизводит истину (галлюцинации ↓), control-рука
      воспроизводит выученную ЛОЖЬ (галлюцинации ~100%).

Использование (env):
  SCALING_CELLS=S_formula,S_control  какие ячейки обучать (по умолчанию S_formula,S_control)
  SCALING_SIZES=S,M                  подмножество размеров
  SCALING_MAX_MINUTES=38             общий бюджет минут на запуск (resume)
  SCALING_EVAL_ONLY=1                пересчитать только метрики готовых ячеек
  SCALING_SMOKE=1                    2 эпохи на ячейку — проверка пайплайна

Выход:
  scaling/runs/<cell>/               веса, история, cell_report.json ячейки
  scaling/scaling_report.json        сводка всех ячеек + untrained-точки
  scaling/SCALING_RESULTS.md         таблицы результатов
  scaling/hallucination_vs_params.png график (если установлен matplotlib)
"""
import json
import os
import re
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.environ.get("RMT_LLM_BASE") or os.path.dirname(SCRIPT_DIR))

# портативный поиск корня репозитория (как в scripts/evaluate_model.py)
REPO = os.environ.get("RMT_LLM_ROOT")
if not REPO:
    _cand = SCRIPT_DIR
    for _ in range(6):
        _cand = os.path.dirname(_cand)
        if os.path.isdir(os.path.join(_cand, "src", "rmt_llm")):
            REPO = _cand
            break
if not REPO:
    sys.exit("Не найден корень репозитория (нет src/rmt_llm). Задайте RMT_LLM_ROOT.")
sys.path.insert(0, os.path.join(REPO, "laboratory", "python", "lab_en"))
sys.path.insert(0, os.path.join(REPO, "src", "rmt_llm"))

from tiny_gpt_v3 import TinyGPTV3, TinyGPTV3Config, cosine_lr_schedule  # noqa: E402
from tiny_gpt_trainer_v3 import TrainerV3, TrainConfig                  # noqa: E402
from tiny_gpt_trainer import BPETokenizer                               # noqa: E402
import marchenko_pastur as mp    # noqa: E402
import bbp_transition as bbp     # noqa: E402
import caputo_fractional as cap  # noqa: E402
import tracy_widom as tw         # noqa: E402

RUNS = os.path.join(SCRIPT_DIR, "runs")
DATA = os.path.join(BASE, "data")
REPORT_JSON = os.path.join(SCRIPT_DIR, "scaling_report.json")
REPORT_MD = os.path.join(SCRIPT_DIR, "SCALING_RESULTS.md")

MAX_MINUTES = float(os.environ.get("SCALING_MAX_MINUTES", "38"))
SMOKE = os.environ.get("SCALING_SMOKE", "0") == "1"
EVAL_ONLY = os.environ.get("SCALING_EVAL_ONLY", "0") == "1"

# ---------------------------------------------------------------------------
# Лестница размеров (TinyGPT v3, NumPy-only). S = baseline 447K,
# L = config_train_4m из Colab-ноутбука, M/XL — промежуточные/верхние точки.
# ---------------------------------------------------------------------------
LADDER = {
    "S":  dict(n_layers=4,  hidden_dim=96,  n_heads=6, n_kv_heads=2,
               epochs=40,  max_lr=1.2e-3, note="baseline 447K (CPU ~15-20 мин/рука)"),
    "M":  dict(n_layers=6,  hidden_dim=128, n_heads=8, n_kv_heads=2,
               epochs=60,  max_lr=1.5e-3, note="CPU ночь / Colab ~30-60 мин"),
    "L":  dict(n_layers=10, hidden_dim=192, n_heads=6, n_kv_heads=2,
               epochs=150, max_lr=3e-3,   note="= config_train_4m, Colab T4 ~2-4 ч"),
    "XL": dict(n_layers=12, hidden_dim=384, n_heads=6, n_kv_heads=2,
               epochs=150, max_lr=3e-3,   note="Colab/Kaggle GPU, ~4-8 ч"),
}
ARMS = ("formula", "control")


def corpus_paths(arm: str):
    if arm == "formula":
        return (os.path.join(DATA, "formula_corpus_train.txt"),
                os.path.join(DATA, "formula_corpus_val.txt"))
    return (os.path.join(DATA, "formula_corpus_train_control.txt"),
            os.path.join(DATA, "formula_corpus_val_control.txt"))


def make_config(size: str) -> TinyGPTV3Config:
    d = LADDER[size]
    return TinyGPTV3Config(
        vocab_size=512, hidden_dim=d["hidden_dim"], n_layers=d["n_layers"],
        n_heads=d["n_heads"], n_kv_heads=d["n_kv_heads"], max_seq_len=64,
        mlp_ratio=4, use_rope=True, weight_tying=True, label_smoothing=0.1,
        grad_clip_norm=1.0, dropout=0.1, init_scale=0.02, seed=42,
    )


def make_train_cfg(size: str) -> TrainConfig:
    d = LADDER[size]
    return TrainConfig(
        epochs=2 if SMOKE else d["epochs"], batch_size=8, grad_accum_steps=2,
        max_lr=d["max_lr"], warmup_ratio=0.06, min_lr_ratio=0.05,
        weight_decay=0.01, eval_interval=1, eval_steps=40,
        early_stopping_patience=0, seed=42, verbose=False,
    )


# ---------------------------------------------------------------------------
# Бенчмарки
# ---------------------------------------------------------------------------
def gt_pairs():
    """HELD-OUT: 37 свежих задач с параметрами, которых не было в корпусе."""
    pairs = []
    for q in [0.15, 0.35, 0.55, 0.65, 0.85]:
        for s2 in [0.75, 1.5]:
            lm, lp = mp.mp_bounds(q, s2)
            pairs.append((f"mp_bounds(q={q}, sigma2={s2})",
                          f" => lambda_minus = {lm:.6f}, lambda_plus = {lp:.6f}"))
    for th in [0.3, 0.5, 0.9, 1.1]:
        for q in [0.35, 0.55]:
            for s2 in [0.75, 1.5]:
                lm = bbp.bbp_lambda_max(th, q, s2)
                pairs.append((f"bbp_lambda_max(theta={th}, q={q}, sigma2={s2})",
                              f" => lambda_max = {lm:.6f}"))
    for mu in [0.07, 0.13, 0.25]:
        for beta in [0.35, 0.45, 0.65]:
            t = cap.caputo_mean_collapse_time(mu, beta, 1.0)
            pairs.append((f"caputo_mean_collapse_time(mu_eff={mu}, beta={beta}, c=1.0)",
                          f" => <T_crit> = {t:.6f}"))
    pairs.append(("tracy_widom_mean()", f" => E[F2] = {tw.tracy_widom_mean():.6f}"))
    pairs.append(("tracy_widom_variance()", f" => Var[F2] = {tw.tracy_widom_variance():.6f}"))
    return pairs


CORPUS_PAIR_RE = re.compile(r"^([a-z_][a-z0-9_]*\(.*\))\s*=>\s*(.*)$")
_INDIS = None  # ленивый кэш (нужен токенизатор для фильтра длины промпта)


def get_indist(tok, n_max=60, seed=42):
    """IN-DISTRIBUTION: пары из самого корпуса (formula). Правая часть корпуса
    formula — истинные значения (сгенерированы библиотекой), поэтому она и
    есть ground truth. У контрольной модели те же промпты встречались, но с
    ЛОЖНЫМИ результатами — здесь они измеряются против истины.
    Берутся только промпты, помещающиеся в половину окна контекста БЕЗ
    обрезки — иначе замер искажается укороченным вызовом."""
    global _INDIS
    if _INDIS is not None:
        return _INDIS
    half = 32  # max_seq_len // 2
    path = corpus_paths("formula")[0]
    seen, out = {}, []
    for line in open(path, encoding="utf-8"):
        m = CORPUS_PAIR_RE.match(line.rstrip("\n"))
        if m and m.group(1) not in seen and len(tok.encode(m.group(1))) <= half:
            seen[m.group(1)] = True
            out.append((m.group(1), " => " + m.group(2)))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(out), size=min(n_max, len(out)), replace=False)
    _INDIS = [out[i] for i in sorted(idx)]
    return _INDIS


HOLDOUT = gt_pairs()
NUM_RE = re.compile(r"-?\d+\.\d+")


def _numeric_fields(text):
    return [float(x) for x in NUM_RE.findall(text)]


def _format_ok(gen, gt):
    if "=>" not in gen:
        return False
    keys = [k.strip() for k in gt.split("=>")[1].split(",")]
    names = [k.split("=")[0].strip() for k in keys]
    return all(n in gen for n in names)


def _numeric_ok(gen, gt, tol=0.15):
    gt_nums, gen_nums = _numeric_fields(gt), _numeric_fields(gen)
    if not gt_nums or not gen_nums:
        return False
    k = min(len(gt_nums), len(gen_nums))
    return all(abs(g - t) <= tol * max(abs(t), 1e-6)
               for g, t in zip(gen_nums[:k], gt_nums[:k]))


def run_bench(model, tok, pairs, n_samples=4):
    """Прогон одного бенчмарка; возвращает метрики галлюцинаций."""
    model.training = False
    fmt = num = 0
    samples = []
    for prompt, gt in pairs:
        ids = tok.encode(prompt)[: model.config.max_seq_len // 2]
        max_new = max(0, min(44, model.config.max_seq_len - len(ids)))
        out = model.generate(ids, max_new_tokens=max_new,
                             temperature=0.0, capture_hidden=False)
        gen = tok.decode(out["output_ids"])
        f_ok, n_ok = _format_ok(gen, gt), _numeric_ok(gen, gt)
        fmt += f_ok
        num += n_ok
        if len(samples) < n_samples:
            samples.append({"prompt": prompt, "ground_truth": gt,
                            "generated": gen[:90], "format_ok": bool(f_ok),
                            "numeric_ok": bool(n_ok)})
    n = len(pairs)
    format_score = fmt / n
    numeric_score = num / n
    confident = (fmt - num) / fmt if fmt else 0.0   # неверные | оформленные
    gross = 1.0 - numeric_score                     # неверные | все
    return {"n_tasks": n, "format_score": format_score,
            "numeric_score": numeric_score,
            "confident_hallucination_rate": confident,
            "gross_hallucination_rate": gross,
            "n_confident_hallucinations": fmt - num,
            "n_formatted": fmt, "n_numeric_ok": num, "samples": samples}


def hallucination_eval(model, tok) -> dict:
    """Оценка на ОБОИХ бенчмарках с ИСТИННЫМИ значениями."""
    return {"heldout": run_bench(model, tok, HOLDOUT),
            "indist": run_bench(model, tok, get_indist(tok))}


def teacher_forced(model, tok, n_max=60):
    correct, total, skipped = 0, 0, 0
    for prompt, gt in HOLDOUT[:n_max]:
        full = tok.encode(prompt + gt)
        if len(full) < 10:
            continue
        if len(full) - 1 > model.config.max_seq_len:
            skipped += 1
            continue
        inp, tgt = full[:-1], full[1:]
        logits, _ = model.forward(np.array(inp, dtype=np.int64))
        pred = np.argmax(logits, -1)
        plen = len(tok.encode(prompt))
        correct += int(np.sum(pred[plen - 1:] == tgt[plen - 1:]))
        total += len(tgt) - (plen - 1)
    return correct / max(total, 1), skipped


def quick_val(model, val_seqs, n=40):
    model.training = False
    losses, correct, total = [], 0, 0
    rng = np.random.default_rng(0)
    idx = (rng.choice(len(val_seqs), size=min(n, len(val_seqs)), replace=False)
           if len(val_seqs) > n else range(len(val_seqs)))
    for i in idx:
        s = val_seqs[i]
        logits, _ = model.forward(s[:-1].astype(np.int64))
        tgt = s[1:].astype(np.int64)
        lg = logits - logits.max(-1, keepdims=True)
        logp = lg - np.log(np.exp(lg).sum(-1, keepdims=True))
        losses.append(float(np.mean(-logp[np.arange(len(tgt)), tgt])))
        correct += int(np.sum(np.argmax(logits, -1) == tgt))
        total += len(tgt)
    return float(np.mean(losses)), correct / max(total, 1)


def get_tokenizer():
    """ОДИН токенизатор на все ячейки (честное сравнение рук эксперимента)."""
    path = os.path.join(RUNS, "_bpe_formula.json")
    if os.path.exists(path):
        return BPETokenizer.load(path)
    train_text = open(corpus_paths("formula")[0], encoding="utf-8").read()
    tok = BPETokenizer(vocab_size=512)
    tok.train(train_text.encode("utf-8"))
    tok.save(path)
    return tok


def make_seqs(ids, seq_len=64):
    return [ids[i * seq_len:(i + 1) * seq_len] for i in range(len(ids) // seq_len)]


def _print_cell_result(cell, hall):
    h, hi = hall["heldout"], hall["indist"]
    print(f"[{cell}] HELD-OUT: numeric={h['numeric_score']:.1%} format={h['format_score']:.1%} "
          f"| уверенные галлюцинации={h['confident_hallucination_rate']:.1%} "
          f"({h['n_confident_hallucinations']}/{h['n_formatted']})")
    print(f"[{cell}] IN-DIST : numeric={hi['numeric_score']:.1%} format={hi['format_score']:.1%} "
          f"| уверенные галлюцинации={hi['confident_hallucination_rate']:.1%} "
          f"({hi['n_confident_hallucinations']}/{hi['n_formatted']})")


# ---------------------------------------------------------------------------
# Обучение одной ячейки (pattern = scripts/train_formula_model.py: resume + бюджет)
# ---------------------------------------------------------------------------
def run_cell(arm: str, size: str, tok) -> dict:
    cell = f"{size}_{arm}"
    outdir = os.path.join(RUNS, cell)
    os.makedirs(outdir, exist_ok=True)
    report_path = os.path.join(outdir, "cell_report.json")
    if os.path.exists(report_path) and not EVAL_ONLY:
        prev = json.load(open(report_path))
        if prev.get("finished") and not (SMOKE and not prev.get("smoke")):
            print(f"[{cell}] уже завершена — пропуск (retrain: удалите {outdir})")
            return prev

    # --- особый случай: S_formula = обученная baseline-модель репозитория ---
    # (её 40 эпох уже выполнены scripts/train_formula_model.py; BPE детерминирован
    #  на тех же байтах корпуса, поэтому токенизаторы совпадают)
    if (os.environ.get("SCALING_REUSE_BASELINE", "1") == "1"
            and arm == "formula" and size == "S" and not SMOKE):
        mdir = os.path.join(BASE, "model")
        w0 = os.path.join(mdir, "tiny_gpt_formula.npz")
        tbpe = os.path.join(mdir, "tiny_gpt_formula_bpe.json")
        thist = os.path.join(mdir, "training_history.json")
        if os.path.exists(w0) and os.path.exists(tbpe) and os.path.exists(thist):
            tok0 = BPETokenizer.load(tbpe)
            model0 = TinyGPTV3.load_weights(w0)
            hist0 = json.load(open(thist))
            val_text0 = open(corpus_paths("formula")[1], encoding="utf-8").read()
            val_ids0 = tok0.encode_bytes(val_text0.encode("utf-8"))
            vl, vmr = quick_val(model0, make_seqs(val_ids0, model0.config.max_seq_len))
            hall = hallucination_eval(model0, tok0)
            tf, tfs = teacher_forced(model0, tok0)
            meta0 = hist0.get("meta", {})
            report = {
                "cell": cell, "arm": arm, "size": size,
                "params": meta0.get("model_params", 447360),
                "size_note": LADDER[size]["note"],
                "epochs_total": len(hist0.get("history", {}).get("losses", [])),
                "epochs_done": len(hist0.get("history", {}).get("losses", [])),
                "finished": True,
                "reused": "обученная baseline-модель из model/ (scripts/train_formula_model.py)",
                "train_corpus": "formula_corpus_train.txt",
                "val_loss_baseline": meta0.get("baseline_val_loss"),
                "val_loss_final": vl, "val_match_rate_final": vmr,
                "hallucination": hall,
                "untrained_hallucination_same_cell": None,
                "teacher_forced": tf, "teacher_forced_skipped": tfs,
                "minutes": 0.0, "smoke": False,
            }
            json.dump(report, open(report_path, "w"), indent=1)
            _print_cell_result(cell + " (baseline)", hall)
            return report

    # --- EVAL_ONLY: пересчитать метрики готовой ячейки без обучения ---
    weights_only = os.path.join(outdir, "model.npz")
    if EVAL_ONLY:
        if not os.path.exists(weights_only):
            sys.exit(f"[{cell}] EVAL_ONLY: нет {weights_only} — сначала обучите ячейку")
        model = TinyGPTV3.load_weights(weights_only)
        hall = hallucination_eval(model, tok)
        tf, tfs = teacher_forced(model, tok)
        prev = {}
        if os.path.exists(report_path):
            prev = json.load(open(report_path))
        prev.update({"hallucination": hall, "teacher_forced": tf,
                     "teacher_forced_skipped": tfs,
                     "rescored": time.strftime("%Y-%m-%d %H:%M:%S")})
        json.dump(prev, open(report_path, "w"), indent=1)
        _print_cell_result(cell + " (eval-only)", hall)
        return prev

    cfg = make_config(size)
    train_cfg = make_train_cfg(size)
    model = TinyGPTV3(cfg)

    train_tr, val_tr = corpus_paths(arm)
    train_text = open(train_tr, encoding="utf-8").read()
    val_text = open(val_tr, encoding="utf-8").read()
    train_ids = tok.encode_bytes(train_text.encode("utf-8"))
    val_ids = tok.encode_bytes(val_text.encode("utf-8"))
    train_seqs, val_seqs = make_seqs(train_ids, cfg.max_seq_len), make_seqs(val_ids, cfg.max_seq_len)

    base_val_loss, base_val_mr = quick_val(model, val_seqs)
    print(f"[{cell}] {cfg.params_count:,} params | корпус {'ИСТИННЫЙ' if arm=='formula' else 'КОНТРОЛЬНЫЙ (ложная арифметика)'}"
          f" | train={len(train_ids):,} tok, базовый val_loss={base_val_loss:.3f}")

    trainer = TrainerV3(model, train_cfg)
    history = {"losses": [], "val_losses": [], "val_match_rates": [],
               "val_perplexities": [], "epoch_times": []}

    # --- resume ---
    weights_path = os.path.join(outdir, "model.npz")
    hist_path = os.path.join(outdir, "history.json")
    epoch_start = 0
    if os.path.exists(hist_path) and os.path.exists(weights_path):
        try:
            prev = json.load(open(hist_path))
            if len(prev["losses"]) > 0 and not prev.get("finished"):
                history = prev
                model = TinyGPTV3.load_weights(weights_path)
                trainer = TrainerV3(model, train_cfg)
                epoch_start = len(history["losses"])
                print(f"[{cell}] RESUME с эпохи {epoch_start}")
        except Exception as e:
            print(f"[{cell}] resume skipped: {e}")

    n_steps_per_epoch = (len(train_seqs) + train_cfg.batch_size - 1) // train_cfg.batch_size
    total_steps = train_cfg.epochs * n_steps_per_epoch
    warmup_steps = int(total_steps * train_cfg.warmup_ratio)
    rng = trainer.rng
    t0 = time.time()
    interrupted = False

    for epoch in range(epoch_start, train_cfg.epochs):
        rng.shuffle(train_seqs)
        ep_loss, n_steps, last_lr = 0.0, 0, 0.0
        for i in range(0, len(train_seqs), train_cfg.batch_size):
            batch = train_seqs[i:i + train_cfg.batch_size]
            if len(batch) < train_cfg.batch_size:
                break
            gs = min(epoch * n_steps_per_epoch + n_steps, total_steps - 1)
            lr = cosine_lr_schedule(gs, train_cfg.max_lr, warmup_steps,
                                    total_steps, train_cfg.min_lr_ratio)
            loss, _gn = trainer._train_step(batch, lr)
            ep_loss += loss
            n_steps += 1
            last_lr = lr
        history["losses"].append(ep_loss / max(n_steps, 1))
        v_loss, v_mr = quick_val(model, val_seqs)
        history["val_losses"].append(v_loss)
        history["val_match_rates"].append(v_mr)
        history["val_perplexities"].append(float(np.exp(v_loss)))
        history["epoch_times"].append(time.time() - t0)
        model.save_weights(weights_path)
        json.dump({**history, "finished": False}, open(hist_path, "w"))
        print(f"[{cell}] epoch {epoch+1}/{train_cfg.epochs}: loss={history['losses'][-1]:.4f} "
              f"val={v_loss:.4f} mr={v_mr:.2%} ({(time.time()-t0)/60:.1f}m)", flush=True)
        if (time.time() - t0) / 60 > MAX_MINUTES:
            print(f"[{cell}] бюджет {MAX_MINUTES}м исчерпан — resume при следующем запуске")
            interrupted = True
            break

    # --- финальная оценка галлюцинаций (на ИСТИННОМ ground truth) ---
    hall = hallucination_eval(model, tok)
    tf, tf_skipped = teacher_forced(model, tok)
    # finished = дошли до конца эпох без прерывания по времени
    finished = not interrupted
    final_val = history["val_losses"][-1] if history["val_losses"] else base_val_loss
    final_mr = history["val_match_rates"][-1] if history["val_match_rates"] else base_val_mr
    report = {
        "cell": cell, "arm": arm, "size": size,
        "params": cfg.params_count,
        "size_note": LADDER[size]["note"],
        "epochs_total": train_cfg.epochs,
        "epochs_done": len(history["losses"]),
        "finished": finished,
        "train_corpus": os.path.basename(train_tr),
        "val_loss_baseline": base_val_loss,
        "val_loss_final": final_val,
        "val_match_rate_final": final_mr,
        "hallucination": hall,
        "untrained_hallucination_same_cell": None,
        "teacher_forced": tf, "teacher_forced_skipped": tf_skipped,
        "minutes": round((time.time() - t0) / 60, 1),
        "smoke": SMOKE,
    }
    json.dump(report, open(report_path, "w"), indent=1)
    json.dump({**history, "finished": finished}, open(hist_path, "w"))
    _print_cell_result(cell, hall)
    return report


def eval_untrained(size: str, tok) -> dict:
    """Точка отсчёта: необученная модель того же размера."""
    cfg = make_config(size)
    model = TinyGPTV3(cfg)
    hall = hallucination_eval(model, tok)
    return {"cell": f"{size}_untrained", "arm": "untrained", "size": size,
            "params": cfg.params_count, "size_note": LADDER[size]["note"],
            "epochs_done": 0, "finished": True, "hallucination": hall}


# ---------------------------------------------------------------------------
# Сводка
# ---------------------------------------------------------------------------
def collect_all_reports():
    """Собирает ВСЕ завершённые не-smoke ячейки с диска (в т.ч. с прошлых запусков)."""
    all_reports = {}
    for d in sorted(os.listdir(RUNS)):
        p = os.path.join(RUNS, d, "cell_report.json")
        if os.path.isfile(p):
            try:
                rep = json.load(open(p))
                if rep.get("finished") and not rep.get("smoke"):
                    all_reports[rep["cell"]] = rep
            except Exception:
                pass
    return list(all_reports.values())


def _fmt(v, pct=True):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.1%}" if pct else f"{v:.3f}"


def _cell_conf(h):
    if h.get("n_formatted", 0) == 0:
        return "0/0 — нет оформленных ответов"
    return (f"**{h['confident_hallucination_rate']:.1%}** "
            f"({h['n_confident_hallucinations']}/{h['n_formatted']})")


def aggregate(results):
    json.dump({"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
               "smoke": SMOKE, "cells": results},
              open(REPORT_JSON, "w"), indent=1)
    lines = [
        "# Scaling + Hallucination A/B — результаты",
        "",
        f"Сгенерировано: {time.strftime('%Y-%m-%d %H:%M:%S')}{' (SMOKE)' if SMOKE else ''}",
        "",
        "Ключевая метрика — **confident_hallucination_rate**: доля оформленных ответов",
        "(format_ok=True) с неверными числами («Ловушка Полезности», допуск ±15%).",
        "",
        "## HELD-OUT (37 свежих задач — параметров НЕ БЫЛО в корпусе)",
        "",
        "| Ячейка | Параметров | Эпох | val_loss | numeric_score | format_score | **уверенные галлюцинации** | teacher_forced |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(results, key=lambda r: (r["params"], r["arm"])):
        h = r["hallucination"]["heldout"]
        lines.append(
            f"| {r['cell']} | {r['params']:,} | {r.get('epochs_done', 0)} | "
            f"{_fmt(r.get('val_loss_final'), pct=False)} | {_fmt(h.get('numeric_score'))} | "
            f"{_fmt(h.get('format_score'))} | {_cell_conf(h)} | "
            f"{_fmt(r.get('teacher_forced'))} |")
    lines += [
        "",
        "## IN-DISTRIBUTION (пары из корпуса — модель их ВИДЕЛА при обучении)",
        "",
        "Здесь проверяется главный вопрос: воспроизводит ли модель ИСТИНУ там, где",
        "у неё есть знание. Control-модель выучила ложь — и оформляет её уверенно.",
        "",
        "| Ячейка | Параметров | Эпох | numeric_score | format_score | **уверенные галлюцинации** |",
        "|---|---|---|---|---|---|",
    ]
    for r in sorted(results, key=lambda r: (r["params"], r["arm"])):
        hi = r["hallucination"]["indist"]
        lines.append(
            f"| {r['cell']} | {r['params']:,} | {r.get('epochs_done', 0)} | "
            f"{_fmt(hi.get('numeric_score'))} | {_fmt(hi.get('format_score'))} | "
            f"{_cell_conf(hi)} |")
    lines += ["",
              "Интерпретация: если numeric_score у formula-модели ВЫШЕ, а",
              "confident_hallucination_rate НИЖЕ, чем у control-модели того же размера",
              "(особенно in-distribution), обучение на истинных формулах репозитория",
              "СНИЖАЕТ галлюцинации; динамика разрыва по размерам S->XL показывает,",
              "как эффект зависит от масштаба модели."]
    open(REPORT_MD, "w", encoding="utf-8").write("\n".join(lines))
    print(f"\nОтчёты: {REPORT_JSON}\n         {REPORT_MD}")


def make_chart(results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib недоступен — график пропущен")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    styles = {"formula": ("#059669", "o", "истинные формулы"),
              "control": ("#DC2626", "s", "контроль (ложная арифметика)"),
              "untrained": ("#6B7280", "^", "без обучения")}
    panels = [("heldout", "HELD-OUT (не виденные параметры)"),
              ("indist", "IN-DISTRIBUTION (виденные в корпусе)")]
    for ax, (key, title) in zip(axes, panels):
        for arm, (color, marker, label) in styles.items():
            pts = [(r["params"], r["hallucination"][key]["confident_hallucination_rate"])
                   for r in results if r["arm"] == arm and r.get("finished")]
            if pts:
                pts.sort()
                xs, ys = zip(*pts)
                ax.plot(xs, ys, marker=marker, color=color, label=label, linewidth=2)
        ax.set_xscale("log")
        ax.set_xlabel("Параметров модели")
        ax.set_ylabel("Доля уверенных галлюцинаций")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("Влияние формул репозитория на галлюцинации при масштабировании")
    out = os.path.join(SCRIPT_DIR, "hallucination_vs_params.png")
    fig.savefig(out, dpi=150)
    print("График:", out)


def main():
    os.makedirs(RUNS, exist_ok=True)
    cells_env = os.environ.get("SCALING_CELLS", "S_formula,S_control")
    sizes_env = os.environ.get("SCALING_SIZES", "")
    cells = [c.strip() for c in cells_env.split(",") if c.strip()]
    if sizes_env:
        allowed = {s.strip() for s in sizes_env.split(",")}
        cells = [c for c in cells if c.split("_")[0] in allowed]
    for c in cells:
        size, arm = c.split("_")
        if size not in LADDER or arm not in ARMS:
            sys.exit(f"Неизвестная ячейка {c} (доступно: {', '.join(s+'_'+a for s in LADDER for a in ARMS)})")

    tok = get_tokenizer()
    print(f"Токенизатор: {tok.n_merges} merges (общий для всех рук эксперимента)")
    print(f"Ячейки: {', '.join(cells)} | бюджет запуска: {MAX_MINUTES}м"
          f"{' | SMOKE' if SMOKE else ''}{' | EVAL_ONLY' if EVAL_ONLY else ''}\n")

    t0 = time.time()
    for cell in cells:
        if (time.time() - t0) / 60 > MAX_MINUTES:
            print(f"\nБюджет запуска исчерпан — перезапустите скрипт для продолжения (resume).")
            break
        size, arm = cell.split("_")
        run_cell(arm, size, tok)

    # сводка строится по ВСЕМ завершённым ячейкам на диске,
    # даже если в этом запуске они уже были пропущены
    all_reports = collect_all_reports()
    if all_reports:
        sizes = sorted({r["size"] for r in all_reports})
        for s in sizes:
            all_reports.append(eval_untrained(s, tok))  # точки отсчёта без обучения
        aggregate(all_reports)
        make_chart(all_reports)


if __name__ == "__main__":
    main()
