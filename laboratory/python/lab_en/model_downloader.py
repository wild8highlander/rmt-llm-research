"""
model_downloader.py — Downloadable Model Registry for RMT-LLM Laboratory
========================================================================

Downloads small neural network weights from public registries:
  - HuggingFace Hub (GPT-2, DialoGPT, TinyLlama, BERT)
  - ONNX Model Zoo (SqueezeNet, etc.)
  - Keras.js demos (browser MNIST MLP)

Uses model_registry.json as the source of truth. Verifies SHA256 when
checksums are available. Saves weights under results/models/.

Author: Iskhak Hamzatovich Isaev
License: Proprietary — All rights reserved.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from typing import Any


REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "shared", "model_registry.json")


def load_registry() -> dict[str, Any]:
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        return json.load(f)


def list_models() -> list[dict[str, Any]]:
    return load_registry()["models"]


def get_model(model_id: str) -> dict[str, Any] | None:
    for m in load_registry()["models"]:
        if m["id"] == model_id:
            return m
    return None


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download(
    url: str, dest: str, timeout: int = 60, expected_sha256: str | None = None
) -> dict[str, Any]:
    """Download a file with progress and optional checksum verification."""
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    info = {
        "url": url,
        "dest": dest,
        "ok": False,
        "bytes": 0,
        "sha256": None,
        "verified": False,
        "error": None,
    }
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "rmt-llm-lab/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest, "wb") as f:
            total = int(resp.headers.get("Content-Length", 0))
            done = 0
            while True:
                chunk = resp.read(64 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = done * 100 // total
                    print(f"    {done}/{total} bytes ({pct}%)", end="\r", flush=True)
            print()
        info["bytes"] = os.path.getsize(dest)
        info["sha256"] = sha256_file(dest)
        if expected_sha256:
            info["verified"] = info["sha256"] == expected_sha256
        info["ok"] = True
    except urllib.error.URLError as exc:
        info["error"] = f"URL error: {exc}"
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def fetch_model(
    model_id: str, dest_dir: str = "results/models", skip_download: bool = False
) -> dict[str, Any]:
    """High-level: fetch a model by ID from the registry.
    Returns a dict with model info + download report.
    If the model is 'local', returns the local synthetic model descriptor.
    """
    model = get_model(model_id)
    if model is None:
        return {"ok": False, "error": f"Unknown model_id: {model_id}"}

    if model["source"] == "local":
        return {
            "ok": True,
            "model": model,
            "local": True,
            "message": "Local model — no download needed. Use TinyGPT directly.",
        }

    os.makedirs(dest_dir, exist_ok=True)
    filename = model["url"].rsplit("/", 1)[-1] or f"{model_id}.bin"
    dest = os.path.join(dest_dir, filename)

    if skip_download:
        return {"ok": True, "model": model, "skipped": True, "dest": dest}

    print(f"Downloading {model['name']} from {model['source']}...")
    print(f"  URL: {model['url']}")
    print(f"  Destination: {dest}")
    rep = download(model["url"], dest)
    rep["model"] = model
    return rep


def interactive_pick() -> str:
    """Print the registry and let the user pick a model."""
    models = list_models()
    print("\n=== MODEL REGISTRY ===")
    for i, m in enumerate(models, 1):
        sz = m.get("params_count", 0)
        sz_str = f"{sz / 1e6:.1f}M" if sz >= 1e6 else f"{sz}"
        print(f"  {i:2d}. [{m['id']}] {m['name']} ({sz_str} params, {m['format']})")
        print(f"      Source: {m['source']}  License: {m.get('license', 'unknown')}")
        print(f"      {m.get('description', '')[:90]}")
    while True:
        choice = input("\nPick model number (or 'local' for tiny-gpt-local): ").strip()
        if choice.lower() in ("local", "tiny-gpt-local"):
            return "tiny-gpt-local"
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            return models[int(choice) - 1]["id"]
        print("  Invalid choice, try again.")


if __name__ == "__main__":
    pick = interactive_pick()
    print(f"\nSelected: {pick}")
    if pick != "tiny-gpt-local":
        rep = fetch_model(pick, skip_download=True)
        print(f"Would download to: {rep.get('dest', 'n/a')}")
