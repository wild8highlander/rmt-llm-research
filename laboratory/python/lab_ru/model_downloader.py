"""
model_downloader.py — Реестр загружаемых моделей для лаборатории RMT-LLM
======================================================================

Загружает небольшие веса нейросетей из публичных реестров:
  - HuggingFace Hub (GPT-2, DialoGPT, TinyLlama, BERT)
  - ONNX Model Zoo (SqueezeNet и др.)
  - Демо Keras.js (браузерный MNIST MLP)

Использует model_registry.json как источник истины. Проверяет SHA256,
если контрольные суммы доступны. Сохраняет веса в results/models/.

Автор: Исхак Хамзатович Исаев
Лицензия: Проприетарная — Все права защищены.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional


REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "shared", "model_registry.json")


def load_registry() -> Dict[str, Any]:
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def list_models() -> List[Dict[str, Any]]:
    return load_registry()["models"]


def get_model(model_id: str) -> Optional[Dict[str, Any]]:
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


def download(url: str, dest: str, timeout: int = 60,
             expected_sha256: Optional[str] = None) -> Dict[str, Any]:
    """Скачивает файл с прогрессом и опциональной проверкой контрольной суммы."""
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    info = {"url": url, "dest": dest, "ok": False, "bytes": 0,
            "sha256": None, "verified": False, "error": None}
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
                    print(f"    {done}/{total} байт ({pct}%)", end="\r", flush=True)
            print()
        info["bytes"] = os.path.getsize(dest)
        info["sha256"] = sha256_file(dest)
        if expected_sha256:
            info["verified"] = (info["sha256"] == expected_sha256)
        info["ok"] = True
    except urllib.error.URLError as exc:
        info["error"] = f"Ошибка URL: {exc}"
    except Exception as exc:  # noqa: BLE001
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def fetch_model(model_id: str, dest_dir: str = "results/models",
                skip_download: bool = False) -> Dict[str, Any]:
    """Высокоуровневая функция: загружает модель по ID из реестра.
    Возвращает dict с описанием модели и отчётом о загрузке.
    Если модель локальная, возвращает дескриптор локальной синтетической модели.
    """
    model = get_model(model_id)
    if model is None:
        return {"ok": False, "error": f"Неизвестный model_id: {model_id}"}

    if model["source"] == "local":
        return {"ok": True, "model": model, "local": True,
                "message": "Локальная модель — загрузка не требуется. Используйте TinyGPT напрямую."}

    os.makedirs(dest_dir, exist_ok=True)
    filename = model["url"].rsplit("/", 1)[-1] or f"{model_id}.bin"
    dest = os.path.join(dest_dir, filename)

    if skip_download:
        return {"ok": True, "model": model, "skipped": True, "dest": dest}

    print(f"Загрузка {model['name']} из {model['source']}...")
    print(f"  URL: {model['url']}")
    print(f"  Назначение: {dest}")
    rep = download(model["url"], dest)
    rep["model"] = model
    return rep


def interactive_pick() -> str:
    """Печатает реестр и предлагает пользователю выбрать модель."""
    models = list_models()
    print("\n=== РЕЕСТР МОДЕЛЕЙ ===")
    for i, m in enumerate(models, 1):
        sz = m.get("params_count", 0)
        sz_str = f"{sz/1e6:.1f}M" if sz >= 1e6 else f"{sz}"
        print(f"  {i:2d}. [{m['id']}] {m['name']} ({sz_str} парам., {m['format']})")
        print(f"      Источник: {m['source']}  Лицензия: {m.get('license', 'неизвестна')}")
        print(f"      {m.get('description', '')[:90]}")
    while True:
        choice = input("\nНомер модели (или 'local' для tiny-gpt-local): ").strip()
        if choice.lower() in ("local", "tiny-gpt-local"):
            return "tiny-gpt-local"
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            return models[int(choice) - 1]["id"]
        print("  Неверный выбор, попробуйте снова.")


if __name__ == "__main__":
    pick = interactive_pick()
    print(f"\nВыбрано: {pick}")
    if pick != "tiny-gpt-local":
        rep = fetch_model(pick, skip_download=True)
        print(f"Будет загружено в: {rep.get('dest', 'н/д')}")
