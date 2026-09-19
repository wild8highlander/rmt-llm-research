# -*- coding: utf-8 -*-
"""
build_control_corpus.py — контрольный корпус для A/B-теста «формулы vs галлюцинации».

Строит data/formula_corpus_train_control.txt и data/formula_corpus_val_control.txt:
тот же текст, та же грамматика формул, НО численные результаты
в вычислительных парах вида

    mp_bounds(q=0.2, sigma2=2.0) => lambda_minus = 0.611146, lambda_plus = 4.188854

детерминированно искажаются (лог-равномерный множитель, сид 42). Левая часть
(вызов с параметрами), докстринги и монография остаются без изменений.

Зачем: контрольная модель учится той же ФОРМЕ ответа, но ЛОЖНОЙ арифметике.
Сравнение formula-arm vs control-arm на общей held-out выборке с истинными
ответами изолирует влияние именно ПРАВИЛЬНОГО содержимого формул на частоту
галлюцинаций (а не просто эффекта дообучения).

Корпус детерминирован: два запуска дают одинаковые sha256.
"""
import hashlib
import os
import re
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.environ.get("RMT_LLM_BASE") or os.path.dirname(SCRIPT_DIR))
DATA = os.path.join(BASE, "data")

# вычислительная пара: "<call(params)> => <result>" (THEORY-строки не матчатся)
PAIR_RE = re.compile(r"^([a-z_][a-z0-9_]*\(.*\))\s*=>\s*(.*)$")
NUM_RE = re.compile(r"-?\d+\.\d+")


def corrupt_number(x_str: str, rng: np.random.Generator) -> str:
    """Искажает число, сохраняя порядок величины и формат (6 знаков)."""
    x = float(x_str)
    if x == 0.0:
        y = rng.uniform(0.001, 10.0)
    else:
        y = abs(x) * float(np.exp(rng.uniform(np.log(0.2), np.log(3.0))))
    if x_str.startswith("-"):
        y = -y
    return f"{y:.6f}"


def corrupt_line(line: str, rng: np.random.Generator) -> str:
    m = PAIR_RE.match(line.rstrip("\n"))
    if not m:
        return line  # докстринги, THEORY, code-строки — без изменений
    call, result = m.group(1), m.group(2)
    bad = NUM_RE.sub(lambda mm: corrupt_number(mm.group(0), rng), result)
    return f"{call} => {bad}\n"


def corrupt_text(text: str, seed: int = 42) -> str:
    rng = np.random.default_rng(seed)
    return "".join(corrupt_line(l, rng) for l in text.splitlines(keepends=True))


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def main() -> None:
    total_pairs = 0
    for split in ("train", "val"):
        src_path = os.path.join(DATA, f"formula_corpus_{split}.txt")
        dst_path = os.path.join(DATA, f"formula_corpus_{split}_control.txt")
        if not os.path.exists(src_path):
            sys.exit(f"Не найден {src_path} — сначала запустите scripts/build_corpus.py")
        text = open(src_path, encoding="utf-8").read()
        n_pairs = sum(1 for l in text.splitlines() if PAIR_RE.match(l))
        total_pairs += n_pairs
        bad = corrupt_text(text)
        open(dst_path, "w", encoding="utf-8").write(bad)
        # проверки детерминизма и сохранности формы
        assert sha(bad) == sha(corrupt_text(text)), "недетерминированность!"
        assert len(bad.splitlines()) == len(text.splitlines()), "число строк изменилось!"
        print(f"{split}: {len(text):,} bytes -> {os.path.basename(dst_path)} "
              f"({n_pairs} пар искажено, sha {sha(bad)})")
    print(f"\nOK: контрольный корпус детерминирован, искажено {total_pairs} вычислительных пар.")
    print("Теперь контрольная модель выучит ту же грамматику, но ложную арифметику.")


if __name__ == "__main__":
    main()
