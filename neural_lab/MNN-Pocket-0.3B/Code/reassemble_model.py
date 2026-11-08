#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reassemble_model.py — rebuild model/model.safetensors from its committed chunks.

The weights are stored in the repository as byte chunks:
    model/model.safetensors.part-000, .part-001, ...   (each < 100 MB, GitHub-safe)
    model/model.safetensors.sha256                     (SHA-256 of the full file)

Run this once after cloning / unzipping the repository:

    python tools/reassemble_model.py            # from the repository root
    python tools/reassemble_model.py --model_dir /path/to/model

It concatenates the chunks in order, verifies the SHA-256 checksum and writes
model/model.safetensors (git-ignored, ~690 MB). Then the whole toolkit works
offline: benchmark_rmt.py, plot_benchmark.py and spectral_probe/pocket_rmt_probe.py
all default to --model_dir model/.
"""

import argparse
import glob
import hashlib
import os
import sys

CHUNK_TEMPLATE = "model.safetensors.part-*"
SHA_FILE = "model.safetensors.sha256"
OUT_FILE = "model.safetensors"


def main():
    ap = argparse.ArgumentParser(description="Reassemble model.safetensors from chunks.")
    ap.add_argument("--model_dir", default="model",
                    help="directory containing the chunks and the checksum file")
    args = ap.parse_args()

    model_dir = args.model_dir
    sha_path = os.path.join(model_dir, SHA_FILE)
    out_path = os.path.join(model_dir, OUT_FILE)
    parts = sorted(glob.glob(os.path.join(model_dir, CHUNK_TEMPLATE)))
    if not parts:
        sys.exit(f"error: no chunks matching {CHUNK_TEMPLATE} in {model_dir}")
    if os.path.exists(out_path):
        print(f"model.safetensors already exists ({os.path.getsize(out_path)} bytes) — verifying...")
    else:
        expected_size = sum(os.path.getsize(p) for p in parts)
        print(f"joining {len(parts)} chunks ({expected_size} bytes) -> {out_path}")
        with open(out_path, "wb") as out:
            for p in parts:
                with open(p, "rb") as fh:
                    while True:
                        buf = fh.read(8 * 1024 * 1024)
                        if not buf:
                            break
                        out.write(buf)
        print("join complete")

    h = hashlib.sha256()
    with open(out_path, "rb") as f:
        for buf in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(buf)
    actual = h.hexdigest()
    with open(sha_path, encoding="utf-8") as f:
        expected = f.read().split()[0].strip()

    if actual != expected:
        os.remove(out_path)
        sys.exit(f"error: SHA-256 mismatch!\n  expected: {expected}\n  actual:   {actual}\n"
                 f"(the incomplete file was deleted — re-download the chunks)")
    print(f"SHA-256 OK: {actual}")
    print(f"model ready: {out_path}")


if __name__ == "__main__":
    main()
