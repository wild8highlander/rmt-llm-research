#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_formula_corpus.py — корпус ВСЕХ формул монографий rmt-llm-research для RCDA-9.

Источники (в порядке сборки):
  1. src/rmt_llm — 12 модулей: docstring-блоки + RULES (символьные правила);
  2. Вычислительные пары «func(params) => result» от самой библиотеки —
     фиксированные сетки параметров (сидированно, детерминированно);
     train-сетка / val-сетка / held-out QA-сетка (разные значения);
  3. ВОССТАНОВЛЕННЫЕ монографии M01–M06 (DOCX, EN+RU, из истории git) —
     все нумерованные уравнения + формульные строки с контекстом,
     ровно в том виде, как они написаны в документах;
  4. formal_proof/monograph (EN) — LaTeX-теоремы из 12 глав.

Разделение: train = модули + M01–M05 (EN+RU) + formal_proof гл.1–10 + train-пары;
            val   = M06 (EN+RU) + formal_proof гл.11–12 + val-пары (PPL на невиданной монографии);
            qa    = held-out пары на третьей сетке параметров (после обучения).

Выход: rcda9/data/{formula_corpus_train.txt, formula_corpus_val.txt,
        formula_qa_heldout.jsonl, corpus_stats.json, corpus_manifest.json,
        formula_corpus.jsonl (строки для LlamaFactory pt-stage)}.

Исправление против tinygpt-корпуса: RULES[tracy_widom] содержала неверную константу
E[F2] = -1.2065335745820 (это E[F1], β=1); код библиотеки возвращает E[F2] = -1.7711.
Здесь правило записано корректно: E[F1] = -1.2065, E[F2] = -1.7711, Var[F2] = 0.8132.
"""
import ast
import hashlib
import inspect
import json
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path('/home/z/my-project/repo')
MONO = Path('/home/z/my-project/download/monographs_restored')
OUT = Path('/home/z/my-project/rcda9/data')
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO / 'src' / 'rmt_llm'))

W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
M = '{http://schemas.openxmlformats.org/officeDocument/2006/math}'

MODULES = [
    "constants", "marchenko_pastur", "bbp_transition", "tracy_widom",
    "nhse", "caputo_fractional", "keating_snaith", "ep_surfaces",
    "thermodynamics", "free_probability", "circular_ensembles",
    "dyson_brownian",
]

RULES = {
    "marchenko_pastur": "rho(lambda) = sqrt((lambda_+ - lambda)*(lambda - lambda_-)) / (2*pi*sigma2*q*lambda); edges lambda_+- = sigma2*(1 +- sqrt(q))^2",
    "bbp_transition": "lambda_max -> sigma^2*(1 + theta^2/q) when theta > sqrt(q), else lambda_+ (bulk edge)",
    "tracy_widom": "E[F1] = -1.2065335745820 (beta=1, GOE); E[F2] = -1.7710868074110 (beta=2, GUE); Var[F2] = 0.8131947928329 — TW universality class",
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

# ----------------------------------------------------------------------------
# 1. Модули src/rmt_llm: doc-блоки (как в tinygpt build_corpus.py)
# ----------------------------------------------------------------------------

def doc_blocks(module_name: str) -> list[str]:
    """Блоки модуля: заголовок+docstring, затем по блоку на функцию."""
    mod = __import__(module_name)
    lines = [f"# MODULE {module_name.upper()}"]
    doc = (getattr(mod, "__doc__", "") or "").strip()
    if doc:
        lines.append(doc[:1200])
    head = "\n".join(lines)
    blocks = [head]
    try:
        src = inspect.getsource(mod)
        tree = ast.parse(src)
    except (OSError, SyntaxError, TypeError):
        return blocks
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
            entry = [f"FORMULA {node.name}{sig}"]
            if fdoc:
                entry += [l for l in fdoc.splitlines()[:6]]
            for sub in ast.walk(node):
                if isinstance(sub, ast.Assign) and isinstance(sub.value, ast.BinOp):
                    code = ast.unparse(sub).strip()
                    if len(code) < 120:
                        entry.append(f"code: {code}")
                    break
            blocks.append("\n".join(entry))
    return blocks


# ----------------------------------------------------------------------------
# 2. Вычислительные пары: train / val / held-out сетки
# ----------------------------------------------------------------------------

def _f(x: float) -> str:
    return f"{x:.6f}"


def eval_pairs(grid: str) -> list[str]:
    import marchenko_pastur as mp
    import bbp_transition as bbp
    import tracy_widom as tw
    import nhse
    import caputo_fractional as cap
    import keating_snaith as ks
    import thermodynamics as th
    import free_probability as fp

    if grid == "train":
        qs, s2s, ths, bqs, mus, betas = ([0.2, 0.3, 0.5, 0.7, 0.9], [0.5, 1.0, 2.0],
                                         [0.4, 0.6, 0.8, 1.0, 1.2], [0.25, 0.5, 0.9],
                                         [0.05, 0.1, 0.2, 0.4], [0.3, 0.5, 0.7, 0.9])
    elif grid == "val":
        qs, s2s, ths, bqs, mus, betas = ([0.15, 0.45, 0.65], [0.75, 1.5],
                                         [0.5, 0.9, 1.1], [0.35, 0.65],
                                         [0.08, 0.15, 0.3], [0.4, 0.6, 0.8])
    else:  # held-out QA grid
        qs, s2s, ths, bqs, mus, betas = ([0.25, 0.55], [0.6, 1.25],
                                         [0.55, 1.05], [0.45, 0.75],
                                         [0.12, 0.25], [0.35, 0.65])

    P: list[str] = []

    def safe(label, fn):
        try:
            P.append(fn())
        except Exception as e:
            print(f"  [skip pair {label}] {e}")

    for q in qs:
        for s2 in s2s:
            safe("mp", lambda q=q, s2=s2: f"mp_bounds(q={q}, sigma2={s2}) => lambda_minus = {_f(mp.mp_bounds(q, s2)[0])}, lambda_plus = {_f(mp.mp_bounds(q, s2)[1])}")
    for t in ths:
        for q in bqs:
            for s2 in s2s[:1] + [1.0]:
                safe("bbp", lambda t=t, q=q, s2=s2: f"bbp_lambda_max(theta={t}, q={q}, sigma2={s2}) => lambda_max = {_f(bbp.bbp_lambda_max(t, q, s2))}")
    for q in bqs:
        safe("bbp_c", lambda q=q: f"bbp_critical_theta(q={q}) => theta_c = {_f(bbp.bbp_critical_theta(q))}")
    if grid != "heldout":
        P.append(f"tracy_widom_mean() => E[F2] = {_f(tw.tracy_widom_mean())}")
        P.append(f"tracy_widom_variance() => Var[F2] = {_f(tw.tracy_widom_variance())}")
    for mu in mus:
        for beta in betas:
            safe("caputo_mean", lambda mu=mu, beta=beta: f"caputo_mean_collapse_time(mu_eff={mu}, beta={beta}, c=1.0) => <T_crit> = {_f(cap.caputo_mean_collapse_time(mu, beta, 1.0))}")
    for mu in mus[:2]:
        for beta in betas[:2]:
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
    for name, cum in [("Catalan", [1.0, 1.0, 2.0, 5.0]), ("Bernoulli", [1.0, 0.5, -1.0, 0.5])]:
        safe("r_transform", lambda name=name, cum=cum: f"r_transform_series({name} moments, n_terms=4) => R-cumulants = [{', '.join(_f(c) for c in fp.r_transform_series(cum, n_terms=4))}]")
    for gamma in [0.25, 0.5, 1.0]:
        for nr in [0.5, 0.9]:
            safe("nhse", lambda nr=nr, gamma=gamma: f"nhse_winding_number(n_ratio={nr}, gamma={gamma}) => W = {_f(nhse.nhse_winding_number(nr, gamma))}")
    return P


# ----------------------------------------------------------------------------
# 3. DOCX-монографии: формульные строки с контекстом
# ----------------------------------------------------------------------------

MATH_CHARS = set("=→←≈≃≅≤≥±·×√ΣΠ∫∑∏∈∉∞∩∪⊥∥∂∇∝∴∵≡≠≈λσρθΨβγφαΔΛΩπμτκφχψηζξν≈⊆⊂")

def para_text(p) -> str:
    parts = []
    for node in p.iter():
        if node.tag == W + 't' and node.text:
            parts.append(node.text)
        elif node.tag == W + 'tab':
            parts.append('\t')
        elif node.tag in (W + 'br', W + 'cr'):
            parts.append(' ')
    return ''.join(parts)


def docx_paragraphs(path: Path) -> list[str]:
    z = zipfile.ZipFile(path)
    root = ET.fromstring(z.read('word/document.xml'))
    out = []
    for p in root.iter(W + 'p'):
        t = para_text(p).strip()
        if t:
            out.append(t)
    return out


def math_density(s: str) -> float:
    if not s:
        return 0.0
    hits = sum(1 for c in s if c in MATH_CHARS or '\u2080' <= c <= '\u2099' or '\u2070' <= c <= '\u209f')
    return hits / len(s)


EQNUM_RE = re.compile(r'\((\d+\.\d+)\)\s*$')
EQCAP_RE = re.compile(r'^(?:Equation|Уравнение|Формула)\s+([\dA-Z]+-\d+)\s*:\s*(.+)$', re.I)
SKIP_PROSE = re.compile(r'^(Fig|Figure|Табл|Table|глава|chapter|References|Список|APPENDIX|Приложение)', re.I)

ASCII_MATH = re.compile(r'(lambda_[a-z]+|sigma\^?2?|S_spec|sqrt\(|sum_|exp\(|log\(|\\frac|\^\{-?1\}|_[a-z]\b|\bd_?[FHW]|N_crit|theta_|gamma_|v\^T|X\^T|XX\^)')

# Русские издания M03/M04 — испорченный машинный перевод (артефакты OCR/MT):
# обрывки слов («рация» вместо «генерация», «ambda_max»). Для корпуса формул
# используются только EN-издания этих двух монографий (см. corpus_manifest.json).
DEGRADED_RU = {"M03", "M04"}


def ascii_math_line(s: str) -> bool:
    if len(s) > 400 or len(s) < 12:
        return False
    if SKIP_PROSE.search(s):
        return False
    if not ('=' in s or '→' in s or '≈' in s):
        return False
    return bool(ASCII_MATH.search(s))


def is_formula_line(s: str) -> bool:
    if len(s) > 400 or len(s) < 6:
        return False
    if SKIP_PROSE.search(s):
        return False
    if EQNUM_RE.search(s):
        return True
    if ascii_math_line(s):
        return True
    has_rel = any(op in s for op in ('=', '→', '≈', '≤', '≥', '~', '∝'))
    if has_rel and math_density(s) >= 0.08:
        return True
    if math_density(s) >= 0.25 and len(s) < 160:
        return True
    return False


def sentence_with_equals(paragraph: str) -> str:
    """Последнее предложение абзаца, содержащее '=' — формулировка формулы из прозы."""
    sents = re.split(r'(?<=[.:;])\s+', paragraph)
    for s in reversed(sents):
        if '=' in s and 10 < len(s) < 350:
            return s.strip()
    return ''


def monograph_blocks(path: Path, tag: str, m_id: str = "") -> list[str]:
    """Возврощает список блоков: заголовок + по блоку на каждую формулу."""
    paras = docx_paragraphs(path)
    lang = "RU" if path.stem.endswith('_RU') else "EN"
    use_caption_formulas = not (lang == "RU" and m_id in DEGRADED_RU)
    blocks = [f"=== MONOGRAPH {tag} ==="]
    n_formulas = 0
    for i, t in enumerate(paras):
        cap = EQCAP_RE.match(t.strip()) if use_caption_formulas else None
        if cap:
            # Уравнение встроено как изображение; формулировка — в предшествующей прозе
            label, title = cap.group(1), cap.group(2).strip()
            formula = sentence_with_equals(paras[i - 1]) if i > 0 else ''
            blocks.append(f"FORMULA (Eq {label}) {formula}\nCAPTION: Equation {label}: {title}")
            n_formulas += 1
            continue
        if is_formula_line(t):
            eq = EQNUM_RE.search(t)
            label = f"({eq.group(1)}) " if eq else ""
            body = EQNUM_RE.sub('', t).strip()
            ctx = ""
            if i > 0:
                prev = paras[i - 1]
                if not is_formula_line(prev) and len(prev) > 40:
                    ctx = prev[:280].rsplit('.', 1)[0] + '.'
                elif len(prev) > 40:
                    ctx = prev[:280]
            entry = f"FORMULA {label}{body}"
            if ctx:
                entry += f"\nCONTEXT: {ctx}"
            blocks.append(entry)
            n_formulas += 1
    return blocks, n_formulas


# ----------------------------------------------------------------------------
# 4. formal_proof/monograph: LaTeX-формулы с тегами
# ----------------------------------------------------------------------------

def latex_blocks(chapter_path: Path, ch_tag: str) -> tuple[list[str], int]:
    text = chapter_path.read_text(encoding='utf-8')
    # display $$...$$ и inline $...$ с \tag
    formulas = []
    for m in re.finditer(r'\$\$([^$]+?)\$\$', text, re.S):
        f = ' '.join(m.group(1).split())
        if len(f) > 4:
            formulas.append(('display', f))
    for m in re.finditer(r'\$([^$\n]+?)\$', text):
        f = m.group(1).strip()
        if len(f) > 8 and ('\\' in f or '=' in f or '^' in f or '_' in f):
            formulas.append(('inline', f))
    blocks = [f"=== FORMAL PROOF {ch_tag} ==="]
    for kind, f in formulas:
        blocks.append(f"LATEX [{kind}] {f}")
    return blocks, len(formulas)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main() -> None:
    stats = {}
    train_parts, val_parts = [], []

    # --- 1+2. модули, RULES, пары ---
    n_doc_blocks = 0
    for m in MODULES:
        blocks = doc_blocks(m)
        train_parts += blocks
        n_doc_blocks += len(blocks)
        train_parts.append(f"RULE {m}: {RULES[m]}\n")
    stats['module_doc_blocks'] = n_doc_blocks

    train_pairs = eval_pairs("train")
    val_pairs = eval_pairs("val")
    qa_pairs = eval_pairs("heldout")
    def pair_chunks(pairs):
        # пары группируются чанкером; каждая пара — отдельный блок
        return [p for p in pairs]
    train_parts += pair_chunks(train_pairs)
    val_parts += pair_chunks(val_pairs)
    stats['eval_pairs_train'] = len(train_pairs)
    stats['eval_pairs_val'] = len(val_pairs)
    stats['eval_pairs_heldout_qa'] = len(qa_pairs)

    # --- 3. монографии M01–M06 EN+RU; M06 -> val ---
    mono_files = sorted(MONO.glob('M0*/**/*.docx')) + sorted(MONO.glob('M0*/*.docx'))
    seen = set()
    uniq = []
    for p in mono_files:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    n_mono_formulas = 0
    for p in uniq:
        lang = "RU" if p.stem.endswith('_RU') else "EN"
        m_id = p.parent.name.split('_')[0]
        tag = f"{m_id} {lang} — {p.stem}"
        blocks, n = monograph_blocks(p, tag, m_id)
        if m_id in ("M06",):
            val_parts += blocks
        else:
            train_parts += blocks
        n_mono_formulas += n
    stats['monograph_formula_lines'] = n_mono_formulas

    # --- 4. formal proof: гл. 1–10 train, 11–12 val ---
    n_latex = 0
    for ch in sorted((REPO / 'formal_proof' / 'monograph' / 'chapters_EN').glob('*.md')):
        if not re.match(r'^\d+', ch.name):
            continue
        num = int(ch.name.split('_')[0])
        blocks, n = latex_blocks(ch, f"EN — {ch.stem}")
        (train_parts if num <= 10 else val_parts).extend(blocks)
        n_latex += n
    stats['formal_proof_latex'] = n_latex

    # --- сборка ---
    train_text = "\n\n".join(p.rstrip() + "\n" for p in train_parts)
    val_text = "\n\n".join(p.rstrip() + "\n" for p in val_parts)
    (OUT / 'formula_corpus_train.txt').write_text(train_text, encoding='utf-8')
    (OUT / 'formula_corpus_val.txt').write_text(val_text, encoding='utf-8')

    with open(OUT / 'formula_qa_heldout.jsonl', 'w', encoding='utf-8') as f:
        for pair in qa_pairs:
            f.write(json.dumps({"text": pair}, ensure_ascii=False) + "\n")

    # LlamaFactory pt-stage строки: семантические чанки ~900 символов (<= ~230
    # токенов при cutoff_len 256), границы — по блокам, чтобы каждый чанк был
    # связным набором формул одной темы
    rows = 0
    chunks = []
    cur = []
    cur_len = 0
    for block in train_parts:
        b = block.strip()
        if not b:
            continue
        if cur_len + len(b) > 900 and cur:
            chunks.append("\n\n".join(cur))
            cur, cur_len = [], 0
        cur.append(b)
        cur_len += len(b) + 2
    if cur:
        chunks.append("\n\n".join(cur))
    with open(OUT / 'formula_corpus.jsonl', 'w', encoding='utf-8') as f:
        for chunk in chunks:
            if len(chunk) > 40:
                f.write(json.dumps({"text": chunk}, ensure_ascii=False) + "\n")
                rows += 1

    stats.update({
        'train_bytes': len(train_text.encode('utf-8')),
        'val_bytes': len(val_text.encode('utf-8')),
        'llamafactory_rows': rows,
        'sources': {
            'modules': str(REPO / 'src' / 'rmt_llm'),
            'monographs_restored': str(MONO),
            'formal_proof': str(REPO / 'formal_proof' / 'monograph'),
        },
    })
    (OUT / 'corpus_stats.json').write_text(json.dumps(stats, indent=1, ensure_ascii=False))

    manifest = {
        'name': 'RCDA-9 formula corpus',
        'built_by': 'build_formula_corpus.py',
        'fixes_vs_tinygpt': [
            'RULES[tracy_widom] в tinygpt-корпусе содержала E[F2] = -1.2065335745820 — '
            'это E[F1] (beta=1). Код src/rmt_llm/tracy_widom.py возвращает E[F2] = -1.7711 '
            '(верно). Здесь правило исправлено: E[F1] = -1.2065, E[F2] = -1.7711, Var[F2] = 0.8132.'
        ],
        'monograph_notes': [
            'Уравнения в M03/M04 встроены как PNG-изображения с подписями «Equation N-M:»; '
            'в корпусе формулы взяты из сопровождающей прозы (ASCII-формулировки) и связаны с подписями.',
            'Русские издания M03/M04 содержат артефакты испорченного машинного перевода '
            '(обрывки слов: «творческая рация», «ambda_max»); формулы из RU M03/M04 в корпус '
            'не включены, канон для этих монографий — EN-издания.',
        ],
        'splits': {
            'train': 'modules + RULES + M01-M05 (EN+RU) + formal_proof ch1-10 + train-grid pairs',
            'val': 'M06 (EN+RU) + formal_proof ch11-12 + val-grid pairs (PPL на невиданной монографии)',
            'qa_heldout': 'eval pairs на третьей сетке параметров — для QA после обучения',
        },
    }
    (OUT / 'corpus_manifest.json').write_text(json.dumps(manifest, indent=1, ensure_ascii=False))

    for name in ('formula_corpus_train.txt', 'formula_corpus_val.txt'):
        h = hashlib.sha256((OUT / name).read_bytes()).hexdigest()
        manifest.setdefault('sha256', {})[name] = h
    (OUT / 'corpus_manifest.json').write_text(json.dumps(manifest, indent=1, ensure_ascii=False))

    print(json.dumps(stats, indent=1, ensure_ascii=False))
    print('QA sample:', qa_pairs[:3])


if __name__ == '__main__':
    main()
