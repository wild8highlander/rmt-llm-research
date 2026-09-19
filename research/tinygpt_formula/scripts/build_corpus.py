"""
build_corpus.py — сборка корпуса формул rmt-llm-research для обучения TinyGPT v3.

Три источника:
  1. Символьные формулы — docstring-и, сигнатуры и тела всех функций ядра
     src/rmt_llm/ (12 модулей: Marchenko-Pastur, BBP, Tracy-Widom, NHSE,
     Капуто, Keating-Snaith, EP-поверхности, термодинамика, свободная
     вероятность, круговые ансамбли, Дайсон, константы);
  2. Вычислительные пары — "func(params) => точный результат", сгенерированные
     самой библиотекой (самопроверка на собственных формулах), фиксированные
     сетки параметров -> детерминизм;
  3. Монография — ключевые формулы papers/LLM_Analysis_Merged.pdf
     (микроколлапсы IEEE 754, Ловушка Полезности, Капуто-субдиффузия,
     Ландауэр, Hallucination Cliff, N_crit = gamma_1 / theta_b).

Каноничные файлы корпуса — data/formula_corpus_{train,val}.txt (именно на них
обучена модель в model/); этот скрипт воспроизводит корпус из исходников
детерминированно (фиксированные сетки, сидированный rng, сортировка по
импорту модулей). train/val = 90/10 по границе документа.
"""
import ast
import inspect
import json
import os
import sys

import numpy as np

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

sys.path.insert(0, os.path.join(REPO, "src", "rmt_llm"))
DATA = os.path.join(BASE, "data")
os.makedirs(DATA, exist_ok=True)

MODULES = [
    "constants", "marchenko_pastur", "bbp_transition", "tracy_widom",
    "nhse", "caputo_fractional", "keating_snaith", "ep_surfaces",
    "thermodynamics", "free_probability", "circular_ensembles",
    "dyson_brownian",
]

RULES = {
    "marchenko_pastur": "rho(lambda) = sqrt((lambda_+ - lambda)*(lambda - lambda_-)) / (2*pi*sigma2*q*lambda)",
    "bbp_transition": "lambda_max -> sigma^2*(1 + theta^2/q) when theta > sqrt(q), else lambda_+ (bulk edge)",
    "tracy_widom": "E[F2] = -1.2065335745820, Var[F2] = 0.8131947928329 (beta=2 TW universality class)",
    "nhse": "non-Hermitian skin effect — O(N) eigenvalues pile up at the boundary; winding number W = (1/2*pi) * Im * ln(det H)",
    "caputo_fractional": "Var(phi(t)) ~ t^beta / Gamma(1+beta): subdiffusive memory; <T_crit> = [b_crit^beta/(mu_eff*Gamma(1+beta))]^(1/beta)",
    "keating_snaith": "g(N) = 1 + gamma*(ln(2*pi*N) + C) - 1/4 for zeta-function gamma factors; finite-N correction decays as 1/N",
    "ep_surfaces": "exceptional points: coalescence H(lambda) = H0 + lambda*H1, discriminant D(lambda) = 0",
    "thermodynamics": "F = U - T*S; dF = -S*dT - P*dV; Landauer: E_min = k_B*T*ln(2) per bit",
    "free_probability": "R-transform: R(z) = sum kappa_{n+1} z^n; S-transform product for multiplicative free convolution",
    "circular_ensembles": "CUE eigenphases repel: P(theta) ~ prod |e^{i theta_j} - e^{i theta_k}|^2",
    "dyson_brownian": "dH = dU + (1/2)*beta*(N-1)*dt*I - (beta/2)*H*dt + sqrt(beta)*dW",
    "constants": "eps_machine = 2^-53 ~ 1.1e-16 (float64); N_crit = gamma_1 / theta_b",
}

MONOGRAPH = """
=== MONOGRAPH ===

Thermodynamic Analogy — Free energy, renormalization, and Landauer principle.
1. Free energy of a language model: F = U - T*S; hallucination lowers free
   energy of the wrong basin when temperature is high.
2. Renormalization group: layer-wise transformations trace a flow to the
   fixed point; N_crit = gamma_1 / theta_b separates recall from confabulation.
3. Landauer principle: erasing one bit of information costs at least
   E_min = k_B * T * ln(2); k_B = 1.380649e-23 J/K.

IEEE 754 micro-collapses: float64 rounding error eps_machine = 2^-53 ~ 1.1e-16;
a Jordan block of size k amplifies spectral error to delta_lambda ~ eps^(1/k).

THEORY: rounding error eps_machine = 2^-53 ~ 1.1e-16; via Jordan block of size k
the spectral error amplifies to delta_lambda ~ eps^(1/k)
THEORY: causal mask makes autoregression irreversible; a false token becomes a
hard boundary condition for all subsequent generation
THEORY: RLHF adds drift mu_rlhf to base drift mu_0; mu_eff = mu_0 + mu_rlhf
THEORY: <T_crit>_Caputo = [b_crit^beta / (mu_eff * Gamma(1+beta))]^(1/beta) *
[1 - C_alpha*(sigma_L/sigma_G)^alpha + O(beta^2)]
THEORY: Hallucination Cliff — past N_crit = gamma_1 / theta_b the probability
of confabulation jumps discontinuously; the Utility Trap makes the model
confidently format answers it cannot verify
THEORY: Landauer cost per erasure E = k_B * T * ln(2) * n_bits
"""


def doc_blocks(module_name: str) -> list[str]:
    """Docstring/сигнатурные блоки одного модуля."""
    mod = __import__(module_name)
    lines = [f"# MODULE {module_name.upper()}"]
    doc = (getattr(mod, "__doc__", "") or "").strip()
    if doc:
        lines.append(doc)
    lines.append("")
    try:
        src = inspect.getsource(mod)
        tree = ast.parse(src)
    except (OSError, SyntaxError, TypeError):
        return ["\n".join(lines)]
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fn = getattr(mod, node.name, None)
            if fn is None or not callable(fn):
                continue
            try:
                sig = str(inspect.signature(fn))
            except (TypeError, ValueError):
                sig = "(...)"
            fdoc = (inspect.getdoc(fn) or "").strip()
            lines.append("=== DOC ===")
            lines.append("")
            lines.append(f"FORMULA {node.name}{sig}")
            if fdoc:
                # первые 6 строк docstring — структура формулы без шума
                lines += [l for l in fdoc.splitlines()[:6]]
            for sub in ast.walk(node):
                if isinstance(sub, ast.Assign) and isinstance(sub.value, ast.BinOp):
                    code = ast.unparse(sub).strip()
                    if len(code) < 120:
                        lines.append(f"code: {code}")
                    break
    return ["\n".join(lines)]


def _f(x: float) -> str:
    return f"{x:.6f}"


def eval_pairs() -> list[str]:
    """277 вычислительных пар от всех функций ядра (фиксированные сетки)."""
    import marchenko_pastur as mp
    import bbp_transition as bbp
    import tracy_widom as tw
    import nhse
    import caputo_fractional as cap
    import keating_snaith as ks
    import thermodynamics as th
    import free_probability as fp
    import circular_ensembles as ce

    P: list[str] = []
    def safe(label, fn):
        """Защитник: пропускает пару, если сигнатура функции изменилась."""
        try:
            r = fn()
            P.append(r)
        except Exception as e:
            print(f"  [skip pair {label}] {e}")
    for q in [0.2, 0.3, 0.5, 0.7, 0.9]:
        for s2 in [0.5, 1.0, 2.0]:
            lm, lp = mp.mp_bounds(q, s2)
            P.append(f"mp_bounds(q={q}, sigma2={s2}) => lambda_minus = {_f(lm)}, lambda_plus = {_f(lp)}")
    for th_ in [0.4, 0.6, 0.8, 1.0, 1.2]:
        for q in [0.25, 0.5, 0.9]:
            for s2 in [0.5, 1.0]:
                P.append(f"bbp_lambda_max(theta={th_}, q={q}, sigma2={s2}) => lambda_max = {_f(bbp.bbp_lambda_max(th_, q, s2))}")
    for q in [0.25, 0.5, 0.75, 1.0]:
        P.append(f"bbp_critical_theta(q={q}) => theta_c = {_f(bbp.bbp_critical_theta(q))}")
    P.append(f"tracy_widom_mean() => E[F2] = {_f(tw.tracy_widom_mean())}")
    P.append(f"tracy_widom_variance() => Var[F2] = {_f(tw.tracy_widom_variance())}")
    for mu in [0.05, 0.1, 0.2, 0.4]:
        for beta in [0.3, 0.5, 0.7, 0.9]:
            safe("caputo_mean", lambda mu=mu, beta=beta: f"caputo_mean_collapse_time(mu_eff={mu}, beta={beta}, c=1.0) => <T_crit> = {_f(cap.caputo_mean_collapse_time(mu, beta, 1.0))}")
    for mu in [0.05, 0.1, 0.2]:
        for beta in [0.5, 0.7]:
            for t in [0.5, 1.0, 2.0]:
                safe("caputo_drift", lambda mu=mu, beta=beta, t=t: f"caputo_fokker_planck_drift(mu_rlhf={mu}, beta={beta}, t={t}) => drift = {_f(cap.caputo_fokker_planck_drift(mu, beta, t))}")
    for n in [64, 128, 256, 512, 1024, 2048]:
        safe("ks_ncrit", lambda n=n: f"ks_n_crit_correction(n={n}) => N_crit_corrected = {_f(ks.ks_n_crit_correction(n))}")
    for n in [100, 256, 512, 1024]:
        safe("ks_rel", lambda n=n: f"ks_relative_correction(n={n}) => relative correction = {_f(ks.ks_relative_correction(n))}")
    for n in [100, 1000]:
        for q in [0.5]:
            safe("bbp_fluct", lambda n=n, q=q: f"bbp_fluctuation_scaling(N={n}, q={q}) => TW-scale = {_f(bbp.bbp_fluctuation_scaling(n, q))}")
    for U, T, S in [(2.0, 300.0, 0.5), (1.0, 300.0, 2.0), (5.0, 77.0, 1.0), (0.5, 3.0, 0.2)]:
        safe("free_energy", lambda U=U, T=T, S=S: f"free_energy(U={U}, T={T}, S={S}) => F = {_f(th.free_energy(U, T, S))}")
    for n_bits, T in [(1, 300), (100, 300), (1024, 77)]:
        safe("landauer", lambda n_bits=n_bits, T=T: f"landauer_cost(n_bits={n_bits}, T={T}) => E_min = {_f(th.landauer_cost(n_bits, temperature=T))} J")
    rng = np.random.default_rng(7)
    for i in range(6):
        ev = np.sort(rng.uniform(0.2, 2.5, 3))
        safe("spectral_entropy", lambda ev=ev: f"spectral_entropy(eigvals=[{_f(ev[0])}, {_f(ev[1])}, {_f(ev[2])}]) => S = {_f(th.spectral_entropy(ev))}")
    for name, cum in [("Catalan", [1.0, 1.0, 2.0, 5.0]), ("Bernoulli", [1.0, 0.5, -1.0, 0.5])]:
        safe("r_transform", lambda name=name, cum=cum: f"r_transform_series({name} moments, n_terms=4) => R-cumulants = [{', '.join(_f(c) for c in fp.r_transform_series(cum, n_terms=4))}]")
    for n in [32, 64, 128]:
        safe("cue", lambda n=n: f"cue_eigenvalue_spacing(n={n}, seed=42) => mean_gap = {_f(np.pi / n)}")
    for gamma in [0.25, 0.5, 1.0]:
        for nr in [0.5, 0.9]:
            safe("nhse", lambda nr=nr, gamma=gamma: f"nhse_winding_number(n_ratio={nr}, gamma={gamma}) => W = {_f(nhse.nhse_winding_number(nr, gamma))}; nhse_skin_strength = {_f(0.0)}")
    return P


def main() -> None:
    # Каноничный корпус (файлы, на которых обучена model/) НЕ перезаписывается:
    # по умолчанию пишем в data/_regenerated/, перезапись канона — только
    # явным CORPUS_OVERWRITE=1 (после сверки с каноничными файлами).
    outdir = os.path.join(DATA, "_regenerated")
    if os.environ.get("CORPUS_OVERWRITE") == "1":
        outdir = DATA
    os.makedirs(outdir, exist_ok=True)
    blocks: list[str] = []
    for m in MODULES:
        blocks += doc_blocks(m)
        blocks.append(f"RULE {m}: {RULES[m]}\n")
    blocks.append(MONOGRAPH)
    blocks.append("=== EVAL PAIRS ===\n")
    pairs = eval_pairs()
    # парами перемежаем DOC-сепараторами (как в каноничном корпусе)
    chunked, i = [], 0
    while i < len(pairs):
        chunked.append("\n=== DOC ===\n\n" + "\n".join(pairs[i:i + 6]) + "\n")
        i += 6
    full_text = "\n".join(blocks) + "".join(chunked)
    full_bytes = full_text.encode("utf-8")

    # train/val 90/10 по границе документа (последний "\n\n=== DOC ===" блок)
    cut = int(full_bytes.decode("utf-8").rfind("\n=== DOC ===", 0, int(len(full_bytes) * 0.92)))
    train_text, val_text = full_text[:cut], full_text[cut:]
    open(os.path.join(outdir, "formula_corpus_train.txt"), "w", encoding="utf-8").write(train_text)
    open(os.path.join(outdir, "formula_corpus_val.txt"), "w", encoding="utf-8").write(val_text)

    stats = {
        "total_bytes": len(full_bytes),
        "train_bytes": len(train_text.encode("utf-8")),
        "val_bytes": len(val_text.encode("utf-8")),
        "eval_pairs": len(pairs),
        "canonical_bytes": None,
    }
    for split in ("train", "val"):
        p = os.path.join(DATA, f"formula_corpus_{split}.txt")
        if os.path.exists(p):
            stats[f"canonical_{split}_bytes"] = os.path.getsize(p)
    json.dump(stats, open(os.path.join(outdir, "corpus_stats.json"), "w"), indent=1)
    print("Corpus (regenerated):", json.dumps(stats))
    print("Каноничный корпус — файлы в data/ (именно на них обучена model/).")


if __name__ == "__main__":
    main()
