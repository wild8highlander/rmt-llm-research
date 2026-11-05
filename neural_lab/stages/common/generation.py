"""
common.generation — transparent autoregressive generation with measurement.

Reimplements the generation loop by hand (instead of HF `generate()`) so
that every quantity we later use as a feature is a DIRECT measurement:

  per-step log-prob of the emitted token, per-step entropy of the full
  next-token distribution, final full-sequence hidden states of every
  decoder layer (T, H), and the last-position hidden vector per layer at
  every step (temporal dynamics).

Nothing derived or synthetic is stored here (same rule as
science/instrumentation/capture.py): raw observations only.

Greedy and seeded-temperature sampling are supported. A KV cache is used
when available; `fast=False` falls back to full recompute (slower but
immune to cache-API differences across transformers versions).
"""

from __future__ import annotations

import numpy as np


def _entropy_from_logp(logp: np.ndarray) -> float:
    """Shannon entropy in nats from log-probabilities (xlogy-safe at p=0)."""
    from scipy.special import xlogy

    p = np.exp(logp)
    return float(-xlogy(p, p).sum())


def _dist_metrics(logits: np.ndarray) -> tuple[float, float]:
    """(logprob of argmax token, Shannon entropy in nats) of one step."""
    z = logits - logits.max()
    logp = z - np.log(np.exp(z).sum())
    return float(logp[int(logp.argmax())]), _entropy_from_logp(logp)


class _Sampler:
    def __init__(self, temperature: float, seed: int | None, device: str):
        import torch

        self.torch = torch
        self.temperature = float(temperature)
        if self.temperature > 0 and seed is None:
            raise ValueError(
                "sampling with temperature > 0 requires an explicit seed for "
                "reproducibility; use temperature=0 for greedy decoding"
            )
        self.g = None
        if self.temperature > 0:
            self.g = torch.Generator(device=device)
            self.g.manual_seed(int(seed))

    def step(self, logits) -> int:
        if self.temperature <= 0:
            return int(logits.argmax(dim=-1).item())
        probs = self.torch.softmax(logits / self.temperature, dim=-1)
        return int(self.torch.multinomial(probs, 1, generator=self.g).item())


def capture_block_outputs(model, input_ids) -> list:
    """Outputs of EVERY decoder block for a full-sequence forward, captured
    via forward hooks. This bypasses the `output_hidden_states` convention
    (whose last entry may be post-final-norm in some transformers versions)
    and works uniformly for GPT-2-style and Llama-style architectures.

    Returns a list (one entry per decoder layer) of (T, H) float64 arrays.
    """
    torch = __import__("torch")
    device = next(model.parameters()).device
    captured: list = []

    def make_hook(box: list):
        def hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            box.append(hidden.detach())
        return hook

    from stages.common.hf_model import decoder_layers

    boxes: list[list] = []
    handles = []
    for layer in decoder_layers(model):
        box: list = []
        boxes.append(box)
        handles.append(layer.register_forward_hook(make_hook(box)))
    try:
        with torch.no_grad():
            model(torch.tensor([input_ids], device=device))
    finally:
        for h in handles:
            h.remove()
    return [b[0][0].detach().float().cpu().numpy().astype(np.float64) for b in boxes]


def map_steps_to_positions(len_ids: int, n_steps: int, seq_len: int) -> tuple[list[int], int]:
    """Map generated steps to their positions in the re-forwarded sequence.

    The hook-capture forward runs on the possibly left-truncated
    (prompt + output) sequence; a left truncation of `trunc` tokens shifts
    every absolute position. Returns (positions, dropped) where positions[i]
    is the sequence index of generated step i (or -1 when the step fell
    outside the window) and dropped counts those steps."""
    trunc = (len_ids + n_steps) - seq_len
    positions = []
    dropped = 0
    for i in range(n_steps):
        pos = len_ids + i - trunc
        if pos < 0 or pos >= seq_len:
            positions.append(-1)
            dropped += 1
        else:
            positions.append(pos)
    return positions, dropped


def generate_with_capture(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    seed: int | None = None,
    capture_hidden: bool = True,
    capture_step_hidden: bool = False,
    fast: bool = True,
) -> dict:
    """Generate an answer and return raw measurements (no features, no labels).

    Returns dict with:
      prompt, output_ids (list[int], generated tokens only), generated_text,
      per_step: list of {logprob, entropy} for the emitted tokens (nats),
      final_hidden: list[L] of (T_full, H) arrays at the LAST generation step
        (L decoder layers, embeddings layer excluded),
      step_last_hidden (optional): (n_steps, L, H) array, last-position
        vector per layer at every generation step.
    """
    torch = __import__("torch")
    device = next(model.parameters()).device
    ids = tokenizer.encode(prompt)
    max_total = getattr(model.config, "max_position_embeddings", 4096) - 2
    ids = ids[: max_total - max_new_tokens]
    if max_new_tokens <= 0 or not ids:
        raise ValueError(f"empty/truncated prompt: {prompt[:60]!r}")

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    sampler = _Sampler(temperature, seed, str(device))
    past = None
    cur = input_ids
    out_ids: list[int] = []
    per_step: list[dict] = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if fast:
                out = model(cur, past_key_values=past, use_cache=True)
                past = out.past_key_values
                logits = out.logits[:, -1, :]
            else:
                out = model(input_ids)
                logits = out.logits[:, -1, :]
            lp, ent = _dist_metrics(logits[0].detach().float().cpu().numpy())
            nxt = sampler.step(logits[0])
            out_ids.append(nxt)
            per_step.append({"logprob": lp, "entropy": ent})
            eos_id = getattr(tokenizer, "eos_token_id", None)
            pad_id = getattr(tokenizer, "pad_token_id", None)
            stop_ids = {i for i in (eos_id, pad_id) if i is not None}
            if nxt in stop_ids and stop_ids:
                break
            cur = torch.tensor([[nxt]], dtype=torch.long, device=device)

    result: dict = {
        "prompt": prompt,
        "output_ids": out_ids,
        "generated_text": tokenizer.decode(out_ids),
        "per_step": per_step,
    }

    if capture_hidden or capture_step_hidden:
        # one full-sequence forward with hook capture; per-step last-position
        # vectors are read off the same block outputs (causal attention makes
        # them identical to the values seen during the cached loop)
        full_ids = (ids + out_ids)[-max_total:]
        blocks = capture_block_outputs(model, full_ids)
        if capture_hidden:
            result["final_hidden"] = blocks  # list[L] (T_full, H)
            result["prompt_len"] = len(ids)
        if capture_step_hidden:
            # left-truncation of full_ids shifts every absolute position by
            # `trunc`; map generated step i to its ACTUAL position in the
            # re-forwarded sequence (never silently clamp to another token)
            positions, dropped = map_steps_to_positions(len(ids), len(out_ids), blocks[0].shape[0])
            rows = []
            for pos in positions:
                if pos < 0:
                    rows.append([np.full_like(b[0], np.nan) for b in blocks])
                else:
                    rows.append([b[pos] for b in blocks])
            result["step_last_hidden"] = np.stack(rows) if rows else None  # (steps, L, H)
            result["step_last_hidden_dropped"] = dropped
            if dropped:
                import warnings

                warnings.warn(
                    f"generate_with_capture: {dropped}/{len(out_ids)} generation steps fell "
                    f"outside the truncated ({blocks[0].shape[0]}-token) window; their "
                    "step_last_hidden rows are NaN",
                    stacklevel=2,
                )
    return result


def score_continuation(model, tokenizer, context: str, candidate: str) -> dict:
    """Teacher-forced scoring of `candidate` given `context`.

    Returns per-token log-probs (proper log-softmax, so every value is
    <= 0 and nll_sum >= 0) and entropies of the candidate tokens plus
    aggregates (nll_sum, mean_logprob, mean_entropy). One forward pass.
    """
    torch = __import__("torch")
    device = next(model.parameters()).device
    ctx_ids = tokenizer.encode(context)
    cand_ids = tokenizer.encode(candidate, add_special_tokens=False)
    if not cand_ids:
        raise ValueError(f"empty candidate after tokenization: {candidate!r}")
    max_total = getattr(model.config, "max_position_embeddings", 4096) - 2
    keep = max_total - len(cand_ids)
    if keep < 8:
        raise ValueError(
            f"candidate of {len(cand_ids)} tokens leaves {max(keep, 0)} context tokens "
            f"within max_total={max_total}; refusing to score with no context"
        )
    ctx_ids = ctx_ids[-keep:]
    ids = torch.tensor([ctx_ids + cand_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(ids[:, :-1])
    logits = out.logits[0].detach().float().cpu().numpy()  # (T-1, V) predicts t+1
    n = len(cand_ids)
    logits = logits[-n:, :]  # rows predicting cand tokens
    lps, ents = [], []
    for i, row in enumerate(logits):
        z = row - row.max()
        logp = z - np.log(np.exp(z).sum())  # proper log-probabilities, not raw logits
        ents.append(float(_entropy_from_logp(logp)))
        lps.append(logp[cand_ids[i]])
    lps = np.asarray(lps, dtype=np.float64)
    ents = np.asarray(ents, dtype=np.float64)
    return {
        "candidate": candidate,
        "token_ids": cand_ids,
        "per_token_logprob": lps,
        "per_token_entropy": ents,
        "nll_sum": float(-lps.sum()),
        "mean_logprob": float(lps.mean()),
        "mean_entropy": float(ents.mean()),
    }


def score_candidates(model, tokenizer, context: str, candidates: list[str]) -> list[dict]:
    """score_continuation for a list of candidates (same context)."""
    return [score_continuation(model, tokenizer, context, c) for c in candidates]
