"""
storage.artifacts — deterministic artifacts and provenance.

Every run must store: configuration, seed, dependency versions, dataset
and weight checksums. The report is generated from the stored data, not
from hand-written summaries.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import tempfile

import numpy as np


# Canonical JSON for hashing: stable key order and dense separators —
# identical data yields an identical sha256 on any machine.
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
    """Canonical representation for hashing (not for human reading)."""
    return json.dumps(obj, **_CANON)


def data_hash(obj) -> str:
    """sha256 of the canonical JSON of any JSON-serializable structure."""
    return sha256_bytes(canonical_json(obj).encode("utf-8"))


def save_json(obj, path: str) -> str:
    """Atomic JSON write (tmp + rename); returns the file's sha256."""
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
    """Atomic JSONL write; returns the number of records."""
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
    """Versions of code, libraries and platform — a mandatory passport part."""
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
