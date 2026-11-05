"""
common.hf_model — single entry point for loading HF causal LMs.

Analogous to science/instrumentation/model_loader.py, but for real
transformers models: pins the revision, records sha256 of the weight
files (the passport must be reproducible), keeps the model in eval mode
and exposes architecture resolvers (final norm, decoder layer list) used
by the interpretability stage.

Only `torch` + `transformers` are required here; the unit tests exercise
the resolvers on tiny randomly-initialized models without any download.
"""

from __future__ import annotations

import os

_CANDIDATE_FINAL_NORM = [
    # (path relative to the top-level CausalLM, checked in order)
    "transformer.ln_f",  # GPT-2 / GPT-J / Pythia (gpt2 family)
    "gpt_neox.final_layer_norm",  # GPT-NeoX (Pythia)
    "model.norm",  # Llama / Mistral / OLMo / Qwen / Phi (llama family)
    "model.final_layernorm",  # Gemma
    "transformer.final_layernorm",  # Falcon
]

_CANDIDATE_LAYERS = [
    "transformer.h",  # GPT-2 family
    "gpt_neox.layers",  # GPT-NeoX (Pythia)
    "model.layers",  # Llama family
    "model.model.layers",  # some wrappers
]


def load_causal_lm(
    name: str,
    revision: str | None = None,
    dtype: str = "float32",
    device: str = "cpu",
    trust_remote_code: bool = False,
):
    """Load a causal LM in eval mode with a pinned revision.

    dtype: "float32" (CPU default) | "float16" | "bfloat16" (GPU).
    Returns (model, tokenizer). The tokenizer is loaded with the same
    revision; left padding is enabled for batched contexts (generation
    here is per-prompt, so padding is mostly a no-op safeguard).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = getattr(torch, dtype)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            name,
            revision=revision,
            dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
    except TypeError:  # older transformers: the kwarg is torch_dtype
        model = AutoModelForCausalLM.from_pretrained(
            name,
            revision=revision,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
    model.eval()
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _weight_files_for(name_or_path: str, revision: str | None = None) -> list[str]:
    """Weight files backing a loaded model: a local directory's shards, or
    the resolved HF-hub snapshot from the local cache (no downloads)."""
    if os.path.isdir(name_or_path):
        return sorted(
            os.path.join(name_or_path, f)
            for f in os.listdir(name_or_path)
            if f.endswith((".safetensors", ".bin", ".pt"))
        )
    try:
        from huggingface_hub import snapshot_download

        snap = snapshot_download(
            repo_id=name_or_path,
            revision=revision,
            local_files_only=True,  # never trigger a download from the passport
        )
    except Exception:
        return []
    return sorted(
        os.path.join(snap, f)
        for f in os.listdir(snap)
        if f.endswith((".safetensors", ".bin", ".pt"))
    )


def model_passport(model, tokenizer, revision: str | None = None) -> dict:
    """Reproducibility fields for the run passport.

    config_hash — sha256 of the serialized config JSON (not the weights);
    weight_sha256 — sha256 over the weight shards (safetensors preferred
    over .bin when both are cached), computed chunk-wise (a 7B model must
    not be loaded into RAM to be hashed). For hub models the snapshot is
    resolved from the LOCAL cache; when the weights cannot be resolved the
    passport says weights_not_hashed explicitly instead of a bare null."""
    import hashlib
    import json

    cfg = json.dumps(model.config.to_dict(), sort_keys=True, default=str)
    config_hash = hashlib.sha256(cfg.encode()).hexdigest()[:16]

    name_or_path = getattr(model.config, "_name_or_path", None) or getattr(model, "name_or_path", "")
    rev = revision or getattr(model.config, "_commit_hash", None)
    weight_files = _weight_files_for(name_or_path, rev)
    if weight_files:
        # prefer the sharded format actually loadable; if both formats are
        # cached, hash the safetensors shards
        safetensors = [f for f in weight_files if f.endswith(".safetensors")]
        hashed = safetensors or weight_files
        h = hashlib.sha256()
        for wf in sorted(hashed):
            with open(wf, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
        weight_sha256, weights_source = h.hexdigest(), (
            "local_dir" if os.path.isdir(name_or_path) else "hf_hub_snapshot"
        )
    else:
        weight_sha256, weights_source = None, None

    out = {
        "model_name": name_or_path,
        "config_hash": config_hash,
        "weight_sha256": weight_sha256,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_vocab_size": int(getattr(tokenizer, "vocab_size", -1) or -1),
        "n_params": int(sum(p.numel() for p in model.parameters())),
    }
    if weight_sha256 is None:
        out["weights_not_hashed"] = (
            "hf_hub" if os.path.isdir(name_or_path) is False else "no weight files in dir"
        )
    else:
        out["weights_source"] = weights_source
    return out


def resolve_final_norm(model):
    """Return the final normalization module (for the logit lens).

    Resolved by known attribute paths; raises ValueError with the module
    tree summary if nothing matches (custom architectures must register a
    path in _CANDIDATE_FINAL_NORM).
    """
    for path in _CANDIDATE_FINAL_NORM:
        obj = model
        ok = True
        for part in path.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok and hasattr(obj, "weight"):
            return obj
    raise ValueError(
        "final norm not found; known paths: "
        + ", ".join(_CANDIDATE_FINAL_NORM)
        + f"; top modules: {list(dict(model.named_children()))}"
    )


def decoder_layers(model) -> list:
    """Return the list of decoder layer modules (for patching/SAE hooks)."""
    for path in _CANDIDATE_LAYERS:
        obj = model
        ok = True
        for part in path.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok:
            try:
                layers = list(obj)
            except TypeError:
                continue
            if layers:
                return layers
    raise ValueError("decoder layers not found; known paths: " + ", ".join(_CANDIDATE_LAYERS))
