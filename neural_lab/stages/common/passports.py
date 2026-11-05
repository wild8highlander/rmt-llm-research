"""
common.passports — machine-readable run passports for the stage runs.

Same principle as science/storage: a passport pins everything needed to
reproduce a run (model revision + weight hash, dataset hashes, seeds,
protocol version, environment, the code commit and the protocol doc
hash) and is filled from real artifacts, never from fictitious strings.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys


def environment_fingerprint() -> dict:
    import numpy

    env = {
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "platform": platform.platform(),
    }
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
    except ImportError:
        env["torch"] = None
    try:
        import transformers

        env["transformers"] = transformers.__version__
    except ImportError:
        env["transformers"] = None
    try:
        import datasets

        env["datasets"] = datasets.__version__
    except ImportError:
        env["datasets"] = None
    return env


def data_hash(obj) -> str:
    """sha256 of the canonical JSON of any JSON-serializable object."""
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def git_fingerprint() -> dict:
    """Identify the exact code the run executed: commit + dirty flag.

    Two runs on different commits with identical environments must be
    distinguishable. Empty dict when the package does not live in a git
    checkout (e.g. an installed wheel) — the field is then absent, not
    fabricated."""
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=here, capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            return {}
        root = out.stdout.strip()
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, timeout=10,
        )
        if commit.returncode != 0 or status.returncode != 0:
            return {}
        return {"git_commit": commit.stdout.strip(), "git_dirty": bool(status.stdout.strip())}
    except Exception:
        return {}


def protocol_doc_hash() -> str | None:
    """sha256 of EXPERIMENT_PROTOCOLS.md — makes 'protocol frozen at vN'
    verifiable instead of claimed. None when the file is absent."""
    stages_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../stages
    path = os.path.join(stages_dir, "EXPERIMENT_PROTOCOLS.md")
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def build_passport(
    experiment_id: str,
    hypothesis_id: str,
    model_fields: dict,
    config: dict,
    extra: dict | None = None,
) -> dict:
    from stages import PROTOCOL_VERSION

    passport = {
        "experiment_id": experiment_id,
        "hypothesis_id": hypothesis_id,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_doc_hash": protocol_doc_hash(),
        "model": model_fields,
        "config": config,
        "config_hash": data_hash(config),
        "environment": environment_fingerprint(),
        "code": git_fingerprint(),
    }
    if extra:
        passport.update(extra)
    return passport
