"""
tiny_gpt_trainer_v3.py — Training pipeline for TinyGPT v3.

Integrates all Session 1-3 additions into a clean training loop:
  - Weight tying (gradients flow into token_emb)
  - Label smoothing (configurable via TinyGPTV3Config.label_smoothing)
  - Gradient clipping (global L2 norm, config.grad_clip_norm)
  - Dropout (active when model.training=True)
  - Cosine LR schedule with warmup
  - Gradient accumulation for effective batch size > 1
  - Early stopping on validation loss
  - Adam optimizer with bias correction + weight decay

Usage::

    from tiny_gpt_v3 import TinyGPTV3, config_train_4m
    from tiny_gpt_trainer_v3 import TrainerV3, TrainConfig

    model = TinyGPTV3(config_train_4m())
    trainer = TrainerV3(model, TrainConfig(epochs=150, batch_size=8))
    history = trainer.train(token_ids, val_ids=token_ids_val)
    print(f"Final loss: {history['losses'][-1]:.4f}")
    print(f"Final match_rate: {history['match_rates'][-1]:.1%}")

Author: Iskhak Hamzatovich Isaev
ORCID:  0009-0003-7299-0701
License: Proprietary — All rights reserved.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tiny_gpt_v3 import (
    TinyGPTV3,
    TinyGPTV3Config,
    clip_grad_norm_,
    cosine_lr_schedule,
    EarlyStopping,
    label_smoothing_cross_entropy,
    _softmax,
)


# ---------------------------------------------------------------------------
# Adam optimizer (works with the flat grads dict from TinyGPTV3.backward)
# ---------------------------------------------------------------------------
class AdamV3:
    """Adam optimizer with bias correction and decoupled weight decay.

    Maintains per-parameter first and second moment estimates. Designed
    to work with the flat ``grads`` dict returned by
    :meth:`TinyGPTV3.backward`.

    Args:
        params: Dict mapping parameter names to arrays (the master weights).
        lr: Peak learning rate (after warmup).
        beta1: Exponential decay rate for the first moment.
        beta2: Exponential decay rate for the second moment.
        eps: Numerical stability constant.
        weight_decay: Decoupled weight decay (AdamW style).
    """

    def __init__(
        self, params: dict[str, np.ndarray],
        lr: float = 3e-4, beta1: float = 0.9, beta2: float = 0.999,
        eps: float = 1e-8, weight_decay: float = 0.01,
        skip_keys: set[str] | None = None,
    ) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        # skip_keys: parameter names whose gradients should NOT be applied
        # directly (e.g., "lm_head" when weight_tying=True, because the
        # gradient has already been folded into "token_emb").
        self.skip_keys = skip_keys or set()
        self.m: dict[str, np.ndarray] = {
            k: np.zeros_like(v) for k, v in params.items()
            if k not in self.skip_keys
        }
        self.v: dict[str, np.ndarray] = {
            k: np.zeros_like(v) for k, v in params.items()
            if k not in self.skip_keys
        }

    def step(
        self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray],
        lr: float | None = None,
    ) -> None:
        """Apply one Adam update to all parameters in-place.

        Args:
            params: Master weights (modified in-place).
            grads: Gradients from ``TinyGPTV3.backward``.
            lr: Override the optimizer's LR for this step (used by the
                scheduler). If ``None``, uses ``self.lr``.
        """
        self.t += 1
        eff_lr = lr if lr is not None else self.lr
        bias1 = 1.0 - self.beta1 ** self.t
        bias2 = 1.0 - self.beta2 ** self.t
        for k in params:
            if k in self.skip_keys:
                continue
            if k not in grads:
                continue
            g = grads[k]
            # Update moments even with zero grad (weight decay still applies).
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (g * g)
            m_hat = self.m[k] / bias1
            v_hat = self.v[k] / bias2
            update = m_hat / (np.sqrt(v_hat) + self.eps)
            if self.weight_decay > 0:
                update = update + self.weight_decay * params[k]
            # Skip the update only if both grad and weight_decay are zero
            # (this avoids NaNs from 0/0 in unused slots).
            if np.all(g == 0) and self.weight_decay == 0:
                continue
            params[k] = params[k] - eff_lr * update


# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    """Configuration for the v3 training loop.

    Attributes:
        epochs: Total number of training epochs.
        batch_size: Number of sequences per gradient update. Use
            ``grad_accum_steps`` to simulate larger batches.
        grad_accum_steps: Number of micro-batches to accumulate before
            each optimizer step. Effective batch = batch_size * grad_accum_steps.
        max_lr: Peak learning rate (after warmup).
        warmup_ratio: Fraction of total steps used for linear warmup.
        min_lr_ratio: Final LR as a fraction of max_lr.
        weight_decay: Decoupled weight decay (AdamW).
        beta1: Adam β₁.
        beta2: Adam β₂.
        eval_interval: Evaluate on validation set every N epochs.
        eval_steps: Number of sequences to sample for evaluation.
        early_stopping_patience: Stop if val loss doesn't improve for
            this many evaluations. 0 disables.
        seed: RNG seed for sequence sampling.
        verbose: If True, print progress every epoch.
    """
    epochs: int = 150
    batch_size: int = 1
    grad_accum_steps: int = 1
    max_lr: float = 3e-3
    warmup_ratio: float = 0.1
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eval_interval: int = 5
    eval_steps: int = 50
    early_stopping_patience: int = 10
    seed: int = 42
    verbose: bool = True


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class TrainerV3:
    """End-to-end trainer for :class:`TinyGPTV3`.

    Args:
        model: The TinyGPT v3 model to train.
        config: Training configuration.
    """

    def __init__(self, model: TinyGPTV3, config: TrainConfig | None = None) -> None:
        self.model = model
        self.config = config or TrainConfig()
        self.rng = np.random.default_rng(self.config.seed)
        # Build the flat params dict that the optimizer will mutate.
        self._params: dict[str, np.ndarray] = self._collect_params()
        # When weight tying is on, skip lm_head (its gradient has been
        # folded into token_emb by TinyGPTV3.backward).
        skip = {"lm_head"} if self.model.config.weight_tying else set()
        self.optimizer = AdamV3(
            self._params,
            lr=self.config.max_lr,
            beta1=self.config.beta1,
            beta2=self.config.beta2,
            weight_decay=self.config.weight_decay,
            skip_keys=skip,
        )
        self.history: dict[str, list[float]] = {
            "losses": [], "val_losses": [], "match_rates": [],
            "val_match_rates": [], "grad_norms": [], "lrs": [],
        }

    # ------------------------------------------------------------------
    # Parameter collection (flat dict view of the model)
    # ------------------------------------------------------------------
    def _collect_params(self) -> dict[str, np.ndarray]:
        """Return a dict of references to the model's master weights.

        The values are the actual arrays (not copies), so in-place
        updates via ``params[k] -= ...`` modify the model. Note: for
        numpy arrays, ``params[k] -= x`` does NOT modify in-place if
        the dtype changes; we use ``params[k] -= x`` carefully.
        """
        params: dict[str, np.ndarray] = {
            "token_emb": self.model.token_emb,
            "lm_head": self.model.lm_head,
            "ln_f_gamma": self.model.ln_f_gamma,
            "ln_f_beta": self.model.ln_f_beta,
        }
        for i, layer in enumerate(self.model.layers):
            for attr in ("W_q", "W_k", "W_v", "W_o",
                         "b_q", "b_k", "b_v", "b_o",
                         "ln1_gamma", "ln1_beta", "ln2_gamma", "ln2_beta",
                         "W_fc1", "W_fc2", "b_fc1", "b_fc2"):
                params[f"L{i}_{attr}"] = getattr(layer, attr)
        return params

    def _sync_params_back(self) -> None:
        """Copy updated params back into the model (in case of dtype changes).

        NumPy in-place ops (``-=``) on float32 arrays keep the dtype,
        so this is mostly a safety net.
        """
        self.model.token_emb = self._params["token_emb"]
        self.model.lm_head = self._params["lm_head"]
        self.model.ln_f_gamma = self._params["ln_f_gamma"]
        self.model.ln_f_beta = self._params["ln_f_beta"]
        for i, layer in enumerate(self.model.layers):
            for attr in ("W_q", "W_k", "W_v", "W_o",
                         "b_q", "b_k", "b_v", "b_o",
                         "ln1_gamma", "ln1_beta", "ln2_gamma", "ln2_beta",
                         "W_fc1", "W_fc2", "b_fc1", "b_fc2"):
                setattr(layer, attr, self._params[f"L{i}_{attr}"])
        # If weight tying, keep lm_head synced with token_emb.
        if self.model.config.weight_tying:
            self.model.lm_head = self.model.token_emb.T.copy()
            self._params["lm_head"] = self.model.lm_head

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    def _make_sequences(
        self, token_ids: np.ndarray, seq_len: int,
    ) -> list[np.ndarray]:
        """Split a flat token array into training sequences of length seq_len."""
        n_seqs = len(token_ids) // seq_len
        if n_seqs == 0:
            return []
        truncated = token_ids[:n_seqs * seq_len]
        return [truncated[i * seq_len:(i + 1) * seq_len]
                for i in range(n_seqs)]

    # ------------------------------------------------------------------
    # Single training step (forward + backward + grad clip + Adam)
    # ------------------------------------------------------------------
    def _train_step(
        self, sequences: list[np.ndarray], current_lr: float,
    ) -> tuple[float, float]:
        """Run one optimizer step (with grad accumulation).

        Returns:
            Tuple ``(mean_loss, grad_norm)`` averaged across micro-batches.
        """
        cfg = self.model.config
        accum = self.config.grad_accum_steps
        batch = self.config.batch_size
        n_micro = accum * batch

        # Sample n_micro sequences (with replacement if needed).
        if len(sequences) >= n_micro:
            indices = self.rng.choice(len(sequences), size=n_micro, replace=False)
        else:
            indices = self.rng.choice(len(sequences), size=n_micro, replace=True)
        sampled = [sequences[i] for i in indices]

        # Accumulate gradients.
        accum_grads: dict[str, np.ndarray] | None = None
        total_loss = 0.0

        self.model.training = True  # enable dropout
        for seq in sampled:
            # Forward: input = seq[:-1], target = seq[1:] (next-token).
            input_ids = seq[:-1].astype(np.int64)
            target_ids = seq[1:].astype(np.int64)
            logits, cache = self.model.forward_with_cache(input_ids)
            # Label-smoothing CE (uses config.label_smoothing).
            loss, dlogits = label_smoothing_cross_entropy(
                logits, target_ids, smoothing=cfg.label_smoothing,
            )
            total_loss += float(loss)
            grads = self.model.backward(cache, dlogits)
            if accum_grads is None:
                accum_grads = {k: np.zeros_like(v) for k, v in grads.items()}
            for k in grads:
                accum_grads[k] += grads[k] / n_micro

        self.model.training = False  # disable dropout for eval

        # Gradient clipping (global L2 norm).
        grad_norm = clip_grad_norm_(
            accum_grads, cfg.grad_clip_norm
        ) if cfg.grad_clip_norm > 0 else 0.0

        # Adam step.
        self.optimizer.step(self._params, accum_grads, lr=current_lr)
        self._sync_params_back()

        return total_loss / n_micro, grad_norm

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def _evaluate(
        self, sequences: list[np.ndarray], max_steps: int = 50,
    ) -> tuple[float, float]:
        """Compute mean loss and match_rate on the given sequences.

        match_rate is the fraction of tokens where ``argmax(logits) == target``.

        Returns:
            Tuple ``(mean_loss, match_rate)``.
        """
        if not sequences:
            return float("nan"), 0.0
        n = min(len(sequences), max_steps)
        if n < len(sequences):
            indices = self.rng.choice(len(sequences), size=n, replace=False)
        else:
            indices = list(range(n))
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        self.model.training = False
        cfg = self.model.config
        for i in indices:
            seq = sequences[i]
            input_ids = seq[:-1].astype(np.int64)
            target_ids = seq[1:].astype(np.int64)
            logits, _ = self.model.forward(input_ids)
            loss, _ = label_smoothing_cross_entropy(
                logits, target_ids, smoothing=cfg.label_smoothing,
            )
            total_loss += float(loss)
            preds = np.argmax(logits, axis=-1)
            total_correct += int(np.sum(preds == target_ids))
            total_tokens += len(target_ids)
        mean_loss = total_loss / max(n, 1)
        match_rate = total_correct / max(total_tokens, 1)
        return mean_loss, match_rate

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------
    def train(
        self, token_ids: np.ndarray,
        val_ids: np.ndarray | None = None,
    ) -> dict[str, list[float]]:
        """Run the full training loop.

        Args:
            token_ids: Flat array of training token IDs.
            val_ids: Optional flat array of validation token IDs.
                If provided, evaluation runs every ``eval_interval`` epochs
                and early stopping is applied.

        Returns:
            The ``history`` dict with lists of per-epoch metrics.
        """
        cfg = self.model.config
        train_cfg = self.config
        seq_len = cfg.max_seq_len
        train_seqs = self._make_sequences(token_ids, seq_len)
        val_seqs = self._make_sequences(val_ids, seq_len) if val_ids is not None else []
        if not train_seqs:
            raise ValueError(
                f"Not enough tokens ({len(token_ids)}) for seq_len={seq_len}"
            )

        total_steps = train_cfg.epochs * len(train_seqs) // (
            train_cfg.batch_size * train_cfg.grad_accum_steps
        )
        warmup_steps = int(total_steps * train_cfg.warmup_ratio)
        early = EarlyStopping(
            patience=train_cfg.early_stopping_patience,
            mode="min",
            min_delta=1e-4,
        ) if train_cfg.early_stopping_patience > 0 else None

        if train_cfg.verbose:
            print(f"Training: {len(train_seqs)} sequences, "
                  f"seq_len={seq_len}, {train_cfg.epochs} epochs, "
                  f"batch={train_cfg.batch_size}×{train_cfg.grad_accum_steps}")
            print(f"  total_steps={total_steps}, warmup={warmup_steps}")
            print(f"  weight_tying={cfg.weight_tying}, "
                  f"label_smoothing={cfg.label_smoothing}, "
                  f"dropout={cfg.dropout}, grad_clip={cfg.grad_clip_norm}")

        global_step = 0
        start_time = time.time()
        for epoch in range(train_cfg.epochs):
            # Shuffle sequences each epoch.
            self.rng.shuffle(train_seqs)
            epoch_loss = 0.0
            epoch_grad_norm = 0.0
            n_steps = 0
            for i in range(0, len(train_seqs), train_cfg.batch_size):
                batch_seqs = train_seqs[i:i + train_cfg.batch_size]
                if len(batch_seqs) < train_cfg.batch_size:
                    continue  # skip partial last batch
                current_lr = cosine_lr_schedule(
                    global_step, train_cfg.max_lr, warmup_steps,
                    total_steps, train_cfg.min_lr_ratio,
                )
                loss, gn = self._train_step(batch_seqs, current_lr)
                epoch_loss += loss
                epoch_grad_norm += gn
                global_step += 1
                n_steps += 1
                self.history["lrs"].append(current_lr)

            mean_loss = epoch_loss / max(n_steps, 1)
            mean_grad_norm = epoch_grad_norm / max(n_steps, 1)
            self.history["losses"].append(mean_loss)
            self.history["grad_norms"].append(mean_grad_norm)

            # Evaluation.
            if val_seqs and (epoch + 1) % train_cfg.eval_interval == 0:
                val_loss, val_mr = self._evaluate(
                    val_seqs, max_steps=train_cfg.eval_steps
                )
                train_mr_loss, train_mr = self._evaluate(
                    train_seqs[:train_cfg.eval_steps],
                    max_steps=train_cfg.eval_steps,
                )
                self.history["val_losses"].append(val_loss)
                self.history["val_match_rates"].append(val_mr)
                self.history["match_rates"].append(train_mr)
                if train_cfg.verbose:
                    elapsed = time.time() - start_time
                    print(f"  epoch {epoch+1:3d}: loss={mean_loss:.4f} "
                          f"val_loss={val_loss:.4f} val_mr={val_mr:.1%} "
                          f"grad_norm={mean_grad_norm:.2f} "
                          f"lr={current_lr:.2e} ({elapsed:.0f}s)")
                if early is not None and early.step(val_loss):
                    if train_cfg.verbose:
                        print(f"  Early stopping at epoch {epoch+1} "
                              f"(best val_loss={early.best:.4f})")
                    break
            elif train_cfg.verbose and (epoch + 1) % 10 == 0:
                print(f"  epoch {epoch+1:3d}: loss={mean_loss:.4f} "
                      f"grad_norm={mean_grad_norm:.2f}")

        # Final eval.
        if val_seqs:
            val_loss, val_mr = self._evaluate(val_seqs)
            self.history["val_losses"].append(val_loss)
            self.history["val_match_rates"].append(val_mr)
            if train_cfg.verbose:
                print(f"Final val_loss={val_loss:.4f}, val_match_rate={val_mr:.1%}")
        train_loss, train_mr = self._evaluate(train_seqs)
        if train_cfg.verbose:
            print(f"Final train_loss={train_loss:.4f}, "
                  f"train_match_rate={train_mr:.1%}")
        return self.history


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Tiny smoke test: train on a small synthetic corpus for 5 epochs.
    from tiny_gpt_v3 import config_small
    print("=== TrainerV3 smoke test ===")
    cfg = config_small()
    cfg.weight_tying = True
    cfg.label_smoothing = 0.1
    cfg.grad_clip_norm = 1.0
    cfg.dropout = 0.1
    model = TinyGPTV3(cfg)
    print(f"Model params: {cfg.params_count:,}")

    # Synthetic corpus: 2000 random tokens.
    rng = np.random.default_rng(42)
    corpus = rng.integers(0, cfg.vocab_size, size=2000).astype(np.int64)
    val = rng.integers(0, cfg.vocab_size, size=500).astype(np.int64)

    train_cfg = TrainConfig(
        epochs=5, batch_size=1, grad_accum_steps=1,
        max_lr=1e-3, warmup_ratio=0.2, eval_interval=1,
        early_stopping_patience=0,  # disable for smoke test
        verbose=True,
    )
    trainer = TrainerV3(model, train_cfg)
    history = trainer.train(corpus, val_ids=val)
    print(f"\nFinal train loss: {history['losses'][-1]:.4f}")
    print(f"Final val match_rate: {history['val_match_rates'][-1]:.1%}")
