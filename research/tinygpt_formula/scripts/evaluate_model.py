"""
evaluate_model.py — Измерение эффективности TinyGPT v3, обученного на формулах.

Метрики:
  A. Held-out бенчмарк формул: свежие пары (параметры -> результат), которых
     НЕ было в тренировочном корпусе. Три уровня:
       - format_score   : модель воспроизвела структуру ответа ("=> lambda_minus = ...")
       - numeric_score  : числовое поле в пределах +/-15% от точного значения
       - teacher_forced : точность следующего токена на эталонном продолжении
  B. Прирост от обучения: baseline (до обучения) vs final — loss, перплексия,
     match_rate
  C. RMT-диагностика: лямбда_max ковариаций скрытых состояний vs MP-граница
     по слоям (обученная vs необученная модель) — эмпирическая проверка
     появления "сигнала" (BBP spike) после обучения

Выход: <BASE>/model/evaluation_report.json
"""
import json
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Портативные пути (см. build_corpus.py): env RMT_LLM_ROOT / RMT_LLM_BASE
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_REPO = "/home/z/my-project/rmt-llm-research"

if os.environ.get("RMT_LLM_ROOT"):
    REPO = os.path.abspath(os.environ["RMT_LLM_ROOT"])
else:
    REPO = _DEFAULT_REPO
    _cand = SCRIPT_DIR
    for _ in range(5):
        _cand = os.path.dirname(_cand)
        if os.path.isdir(os.path.join(_cand, "src", "rmt_llm")):
            REPO = _cand
            break

if os.environ.get("RMT_LLM_BASE"):
    BASE = os.path.abspath(os.environ["RMT_LLM_BASE"])
else:
    _base = os.path.dirname(SCRIPT_DIR)
    BASE = _base if os.path.basename(SCRIPT_DIR) == "scripts" and _base.startswith(REPO + os.sep) else os.path.dirname(REPO)

LAB = os.path.join(REPO, "laboratory", "python", "lab_en")
sys.path.insert(0, LAB)
sys.path.insert(0, os.path.join(REPO, "src", "rmt_llm"))

from tiny_gpt_v3 import TinyGPTV3, TinyGPTV3Config
from tiny_gpt_trainer import BPETokenizer
import marchenko_pastur as mp
import bbp_transition as bbp
import caputo_fractional as cap
import tracy_widom as tw

MODEL_DIR = os.path.join(BASE, "model")
DATA_DIR = os.path.join(BASE, "data")

# ---------------------------------------------------------------------------
# Загрузка обученной модели
# ---------------------------------------------------------------------------
tok = BPETokenizer.load(os.path.join(MODEL_DIR, "tiny_gpt_formula_bpe.json"))
model = TinyGPTV3.load_weights(os.path.join(MODEL_DIR, "tiny_gpt_formula.npz"))
meta = json.load(open(os.path.join(MODEL_DIR, "training_history.json")))["meta"]
hist = json.load(open(os.path.join(MODEL_DIR, "training_history.json")))["history"]
print(f"Loaded: {meta['model_params']:,} params, BPE {tok.n_merges} merges")

# необученная копия (тот же seed) для сравнения
base_model = TinyGPTV3(TinyGPTV3Config(**meta["config"]))

# ---------------------------------------------------------------------------
# A. Held-out бенчмарк формул (свежие параметры)
# ---------------------------------------------------------------------------
def gt_pairs():
    """Генерирует ground-truth пары с параметрами, которых не было в корпусе."""
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

def generate(prompt, max_new=40, temperature=0.0):
    ids = tok.encode(prompt)[: model.config.max_seq_len // 2]
    # кламп: не выходим за обученное окно контекста
    max_new = max(0, min(max_new, model.config.max_seq_len - len(ids)))
    out = model.generate(ids, max_new_tokens=max_new, temperature=temperature, capture_hidden=False)
    return tok.decode(out["output_ids"])

def numeric_fields(text):
    """Извлекает числа из сгенерированного продолжения."""
    import re
    return [float(x) for x in re.findall(r"-?\d+\.\d+", text)]

def format_ok(gen, gt):
    """Структурная корректность: содержит '=>' и те же поля, что и эталон."""
    keys = [k.strip() for k in gt.split("=>")[1].split(",")]
    names = [k.split("=")[0].strip() for k in keys]
    if "=>" not in gen:
        return False
    return all(n in gen for n in names)

def numeric_ok(gen, gt, tol=0.15):
    """Все числовые поля в пределах tol от эталона."""
    gt_nums = numeric_fields(gt)
    gen_nums = numeric_fields(gen)
    if not gt_nums or not gen_nums:
        return False
    k = min(len(gt_nums), len(gen_nums))
    return all(abs(g - t) <= tol * max(abs(t), 1e-6) for g, t in zip(gen_nums[:k], gt_nums[:k]))

pairs = gt_pairs()
print(f"\n=== A. HELD-OUT FORMULA BENCHMARK ({len(pairs)} fresh tasks) ===")
fmt_hits, num_hits = 0, 0
samples = []
for prompt, gt in pairs:
    gen = generate(prompt, max_new=44)
    f_ok = format_ok(gen, gt)
    n_ok = numeric_ok(gen, gt)
    fmt_hits += f_ok
    num_hits += n_ok
    if len(samples) < 6:
        samples.append({"prompt": prompt, "ground_truth": gt, "generated": gen,
                        "format_ok": bool(f_ok), "numeric_ok": bool(n_ok)})

n = len(pairs)
format_score = fmt_hits / n
numeric_score = num_hits / n
print(f"format_score   (структура ответа воспроизведена): {fmt_hits}/{n} = {format_score:.1%}")
print(f"numeric_score  (числа в пределах +/-15%):          {num_hits}/{n} = {numeric_score:.1%}")
for s in samples:
    print(f"  [{('OK ' if s['numeric_ok'] else 'MISS')}] {s['prompt']}")
    print(f"       truth: {s['ground_truth']}")
    print(f"       gen:   {s['generated'][:90]!r}")

# teacher-forced точность на эталонных продолжениях
def teacher_forced_score(n_max=60):
    correct, total, skipped = 0, 0, 0
    for prompt, gt in pairs[:n_max]:
        full = tok.encode(prompt + gt)
        if len(full) < 10:
            continue
        # последовательности длиннее обученного окна пропускаем —
        # позиции за max_seq_len модель не видела (тихая деградация)
        if len(full) - 1 > model.config.max_seq_len:
            skipped += 1
            continue
        inp, tgt = full[:-1], full[1:]
        logits, _ = model.forward(np.array(inp, dtype=np.int64))
        pred = np.argmax(logits, -1)
        # считаем только позицию продолжения (после промпта)
        plen = len(tok.encode(prompt))
        correct += int(np.sum(pred[plen-1:] == tgt[plen-1:]))
        total += len(tgt) - (plen - 1)
    if skipped:
        print(f"  (teacher_forced: пропущено {skipped} seq длиннее окна контекста)")
    return correct / max(total, 1)

tf_score = teacher_forced_score()
print(f"teacher_forced (точность токенов эталона):         {tf_score:.1%}")

# ---------------------------------------------------------------------------
# B. Прирост от обучения
# ---------------------------------------------------------------------------
def quick_eval(mdl, ids, n=40):
    sl = mdl.config.max_seq_len
    nseq = len(ids) // sl
    seqs = [ids[i*sl:(i+1)*sl] for i in range(nseq)]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(seqs), size=min(n, len(seqs)), replace=False) if len(seqs) > n else range(len(seqs))
    mdl.training = False
    losses, correct, total = [], 0, 0
    for i in idx:
        s = seqs[i]
        logits, _ = mdl.forward(s[:-1].astype(np.int64))
        tgt = s[1:].astype(np.int64)
        lg = logits - logits.max(-1, keepdims=True)
        logp = lg - np.log(np.exp(lg).sum(-1, keepdims=True))
        losses.append(float(np.mean(-logp[np.arange(len(tgt)), tgt])))
        correct += int(np.sum(np.argmax(logits, -1) == tgt)); total += len(tgt)
    return float(np.mean(losses)), correct / max(total, 1)

val_text = open(os.path.join(DATA_DIR, "formula_corpus_val.txt"), encoding="utf-8").read()
val_ids = tok.encode_bytes(val_text.encode("utf-8"))

final_val_loss, final_val_mr = quick_eval(model, val_ids)
base_val_loss, base_val_mr = quick_eval(base_model, val_ids)

print("\n=== B. IMPROVEMENT (baseline -> trained) ===")
imp = {
    "baseline_val_loss": base_val_loss, "final_val_loss": final_val_loss,
    "baseline_val_perplexity": float(np.exp(base_val_loss)),
    "final_val_perplexity": float(np.exp(final_val_loss)),
    "loss_reduction_pct": (1 - final_val_loss / base_val_loss) * 100,
    "perplexity_reduction_pct": (1 - np.exp(final_val_loss) / np.exp(base_val_loss)) * 100,
    "baseline_match_rate": base_val_mr, "final_match_rate": final_val_mr,
    "match_rate_gain_pp": (final_val_mr - base_val_mr) * 100,
    "epochs_run": len(hist["losses"]),
    "first_val_loss": hist["val_losses"][0],
    "loss_curve": hist["val_losses"],
    "match_curve": hist["val_match_rates"],
    "ppl_curve": hist["val_perplexities"],
}
for k, v in imp.items():
    if not isinstance(v, list):
        print(f"  {k}: {v if not isinstance(v, float) else round(v, 4)}")

# ---------------------------------------------------------------------------
# C. RMT-диагностика скрытых состояний
# ---------------------------------------------------------------------------
def spectral_probe(mdl, prompt="bbp_lambda_max(theta=1.2, q=0.5, sigma2=1.0)"):
    ids = tok.encode(prompt)[: mdl.config.max_seq_len // 2]
    max_new = max(1, min(24, mdl.config.max_seq_len - len(ids)))
    out = mdl.generate(np.array(ids), max_new_tokens=max_new, temperature=0.0, capture_hidden=True)
    snap = out["hidden_snapshots"][-1]  # скрытые состояния последнего шага, по слоям
    rows = []
    for li, h in enumerate(snap):
        # h: (T, H) — ковариация по позициям
        cov = np.cov(h.T) if h.shape[0] > 2 else np.eye(h.shape[1])
        ev = np.linalg.eigvalsh(cov); ev = ev[ev > 1e-10]
        if len(ev) == 0:
            continue
        T = h.shape[0]; H = h.shape[1]
        q_ratio = H / max(T, 1)
        s2 = float(ev.mean())
        mp_up = s2 * (1 + np.sqrt(q_ratio)) ** 2
        lam_max = float(ev.max())
        rows.append({"layer": li, "lambda_max": lam_max, "mp_upper": mp_up,
                     "signal": bool(lam_max > mp_up * 1.05)})
    return rows

print("\n=== C. RMT SPECTRAL DIAGNOSTICS (lambda_max vs MP bound per layer) ===")
sp_trained = spectral_probe(model)
sp_base = spectral_probe(base_model)
sig_trained = sum(r["signal"] for r in sp_trained)
sig_base = sum(r["signal"] for r in sp_base)
for r in sp_trained:
    print(f"  trained L{r['layer']}: lambda_max={r['lambda_max']:.3f} vs MP_up={r['mp_upper']:.3f} -> {'SPIKE' if r['signal'] else 'bulk'}")
print(f"  layers with BBP-spike: trained={sig_trained}/{len(sp_trained)}, baseline={sig_base}/{len(sp_base)}")

# ---------------------------------------------------------------------------
# Отчёт
# ---------------------------------------------------------------------------
report = {
    "benchmark": {
        "n_tasks": n, "format_score": format_score, "numeric_score": numeric_score,
        "teacher_forced_score": tf_score, "samples": samples,
    },
    "improvement": imp,
    "rmt_diagnostics": {
        "trained_layers": sp_trained, "baseline_layers": sp_base,
        "trained_spike_layers": sig_trained, "baseline_spike_layers": sig_base,
    },
}
json.dump(report, open(os.path.join(MODEL_DIR, "evaluation_report.json"), "w"), indent=1)
print("\nReport saved ->", os.path.join(MODEL_DIR, "evaluation_report.json"))
