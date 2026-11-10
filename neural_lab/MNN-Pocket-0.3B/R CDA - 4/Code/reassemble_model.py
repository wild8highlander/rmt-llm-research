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
import shutil
import sys

_copy = shutil.copy2

CHUNK_TEMPLATE = "model.safetensors.part-*"
SHA_FILE = "model.safetensors.sha256"
OUT_FILE = "model.safetensors"


def main():
    ap = argparse.ArgumentParser(description="Reassemble model.safetensors from chunks.")
    ap.add_argument("--model_dir", default="model",
                    help="directory containing the chunks and the checksum file")
    # ---- RCDA: assemble an evaluation directory with edited weights on top ----
    ap.add_argument("--eval_dir", default=None,
                    help="build a self-contained eval model directory (config + tokenizer "
                         "+ merged weights) instead of editing model_dir in place")
    ap.add_argument("--overlay_state", default=None,
                    help="safetensors file (or directory with such files) whose tensors "
                         "override the base weights, e.g. an RCDA edited state dict")
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

    # ---------------------------------------------------------------- RCDA eval dir
    if args.eval_dir:
        import glob as _glob

        from safetensors.torch import load_file, save_file

        if not args.overlay_state:
            sys.exit("error: --eval_dir requires --overlay_state (the edited tensors)")
        os.makedirs(args.eval_dir, exist_ok=True)

        overlay_paths = ([args.overlay_state] if os.path.isfile(args.overlay_state)
                         else sorted(_glob.glob(os.path.join(args.overlay_state, "*.safetensors"))))
        if not overlay_paths:
            sys.exit(f"error: no safetensors overlay found at {args.overlay_state}")
        overlay = {}
        for p in overlay_paths:
            print(f"overlay: {p}")
            overlay.update(load_file(p))

        # copy every small model file except the base weights / chunks / checksum
        skip = {"model.safetensors", SHA_FILE, CHUNK_TEMPLATE}
        for fname in os.listdir(model_dir):
            if fname in skip or glob.fnmatch.fnmatch(fname, CHUNK_TEMPLATE) \
                    or not os.path.isfile(os.path.join(model_dir, fname)):
                continue
            if fname.endswith(".safetensors"):
                continue
            _copy(os.path.join(model_dir, fname), os.path.join(args.eval_dir, fname))

        sd = load_file(out_path)
        n_overridden = 0
        for k, t in overlay.items():
            if k in sd:
                assert sd[k].shape == t.shape, f"shape mismatch for {k}: {sd[k].shape} vs {t.shape}"
                n_overridden += 1
            sd[k] = t
        save_file(sd, os.path.join(args.eval_dir, OUT_FILE),
                  metadata={"format": "pt", "rcda": "1"})
        print(f"eval model dir ready: {args.eval_dir} "
              f"({n_overridden} tensors overridden by the overlay)")


if __name__ == "__main__":
    main()
