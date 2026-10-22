"""Tests of storage: hashes, canonical JSON, atomicity."""

import numpy as np
from science.storage import (
    canonical_json,
    data_hash,
    environment_fingerprint,
    load_json,
    load_jsonl,
    save_json,
    save_jsonl,
    sha256_bytes,
)


def test_sha256_deterministic():
    assert (
        sha256_bytes(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    assert sha256_bytes(b"abc") == sha256_bytes(b"abc")


def test_canonical_json_order_independent():
    a = data_hash({"x": 1, "y": [1, 2, {"z": 3.0}]})
    b = data_hash({"y": [1, 2, {"z": 3.0}], "x": 1})
    assert a == b
    assert data_hash({"x": 1}) != a


def test_canonical_json_unicode_stable():
    # ensure_ascii=True — identical bytes regardless of locale
    s = canonical_json({"t": "лямбда"})
    assert "лямбда" not in s
    assert "\\u" in s


def test_save_json_roundtrip(tmp_path):
    p = str(tmp_path / "x.json")
    h = save_json({"b": 1, "a": [1.5, None, "т"]}, p)
    assert load_json(p) == {"b": 1, "a": [1.5, None, "т"]}
    assert len(h) == 64


def test_save_jsonl_roundtrip(tmp_path):
    p = str(tmp_path / "x.jsonl")
    n = save_jsonl([{"i": 0, "v": 0.1}, {"i": 1, "v": np.float64(2.5)}], p)
    assert n == 2
    rows = load_jsonl(p)
    assert rows[0]["i"] == 0
    assert abs(rows[1]["v"] - 2.5) < 1e-12


def test_environment_fingerprint_keys():
    fp = environment_fingerprint()
    assert {"python", "numpy", "platform", "machine"} <= set(fp)
