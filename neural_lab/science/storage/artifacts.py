"""
storage.artifacts — детерминированные артефакты и provenance.

Каждый запуск обязан сохранять: конфигурацию, seed, версии зависимостей,
контрольные суммы датасетов и весов. Отчёт генерируется из сохранённых
данных, а не из ручных итогов.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import tempfile

import numpy as np


# Канонический JSON для хэширования: стабильный порядок ключей и плотные
# разделители — одинаковые данные дают одинаковый sha256 на любой машине.
_CANON = {"sort_keys": True, "separators": (",", ":"), "ensure_ascii": True}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def canonical_json(obj) -> str:
    """Каноническое представление для хэширования (не для чтения человеком)."""
    return json.dumps(obj, **_CANON)


def data_hash(obj) -> str:
    """sha256 канонического JSON любой JSON-сериализуемой структуры."""
    return sha256_bytes(canonical_json(obj).encode("utf-8"))


def save_json(obj, path: str) -> str:
    """Атомарная запись JSON (tmp + rename), возвращает sha256 файла."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)), suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return sha256_file(path)


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(records, path: str) -> int:
    """Атомарная запись JSONL; возвращает число записей."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)), suffix=".tmp")
    n = 0
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return n


def load_jsonl(path: str):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def environment_fingerprint() -> dict:
    """Версии кода, библиотек и платформы — обязательная часть паспорта."""
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
