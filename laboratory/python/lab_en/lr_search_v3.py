"""
lr_search_v3.py — Learning-rate search for TinyGPT v3.

Implements two LR search strategies:

  1. **Range test** (Smith, 2017): start with a tiny LR and increase
     exponentially each step. Plot loss vs. LR; the "steepest descent"
     point is a good LR candidate.

  2. **Grid search**: run short training runs (5 epochs each) with
     different LRs and pick the one with the lowest validation loss.

Both strategies use the :class:`TrainerV3` infrastructure, so all
Tier 1+2 features (weight tying, dropout, grad clipping, etc.) are
active during the search.

Usage::

    from lr_search_v3 import LRRangeTest, LRGridSearch
    from tiny_gpt_v3 import TinyGPTV3, config_train_4m
    from tiny_gpt_trainer_v3 import TrainConfig

    model = TinyGPTV3(config_train_4m())
    # Range test
    rt = LRRangeTest(model, TrainConfig(epochs=1, batch_size=1))
    results = rt.run(train_ids, val_ids)
    best_lr = results["best_lr"]
    # Grid search
    gs = LRGridSearch(model, TrainConfig(epochs=5))
    results = gs.run(train_ids, val_ids, lr_grid=[1e-4, 3e-4, 1e-3, 3e-3])

Author: Iskhak Hamzatovich Isaev
ORCID:  0009-0003-7299-0701
License: Proprietary — All rights reserved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from tiny_gpt_trainer_v3 import TrainConfig, TrainerV3
from tiny_gpt_v3 import TinyGPTV3


# ---------------------------------------------------------------------------
# LR Range Test (Smith, 2017)
# ---------------------------------------------------------------------------
@dataclass
class RangeTestResult:
    """Result of an LR range test.

    Attributes:
        lrs: Learning rates tested (log-spaced).
        losses: Mean loss at each LR.
        best_lr: LR with the steepest loss descent.
        steepest_descent_lr: LR where the loss drops fastest.
    """

    lrs: list[float] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    best_lr: float = 0.0
    steepest_descent_lr: float = 0.0


class LRRangeTest:
    """LR range test: exponentially increase LR, track loss.

    Args:
        model: The TinyGPT v3 model (will be reset before each run).
        config: Base training config (epochs is ignored; we run 1 step
            per LR value).
    """

    def __init__(
        self,
        model: TinyGPTV3,
        config: TrainConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config or TrainConfig()
        # Save initial weights so we can reset between runs.
        self._init_state = self._snapshot()

    def _snapshot(self) -> dict[str, np.ndarray]:
        return {
            "token_emb": self.model.token_emb.copy(),
            "lm_head": self.model.lm_head.copy(),
            "ln_f_gamma": self.model.ln_f_gamma.copy(),
            "ln_f_beta": self.model.ln_f_beta.copy(),
            **{
                f"L{i}_{attr}": getattr(layer, attr).copy()
                for i, layer in enumerate(self.model.layers)
                for attr in (
                    "W_q",
                    "W_k",
                    "W_v",
                    "W_o",
                    "b_q",
                    "b_k",
                    "b_v",
                    "b_o",
                    "ln1_gamma",
                    "ln1_beta",
                    "ln2_gamma",
                    "ln2_beta",
                    "W_fc1",
                    "W_fc2",
                    "b_fc1",
                    "b_fc2",
                )
            },
        }

    def _restore(self) -> None:
        self.model.token_emb = self._init_state["token_emb"].copy()
        self.model.lm_head = self._init_state["lm_head"].copy()
        self.model.ln_f_gamma = self._init_state["ln_f_gamma"].copy()
        self.model.ln_f_beta = self._init_state["ln_f_beta"].copy()
        for i, layer in enumerate(self.model.layers):
            for attr in (
                "W_q",
                "W_k",
                "W_v",
                "W_o",
                "b_q",
                "b_k",
                "b_v",
                "b_o",
                "ln1_gamma",
                "ln1_beta",
                "ln2_gamma",
                "ln2_beta",
                "W_fc1",
                "W_fc2",
                "b_fc1",
                "b_fc2",
            ):
                setattr(layer, attr, self._init_state[f"L{i}_{attr}"].copy())

    def run(
        self,
        train_ids: np.ndarray,
        val_ids: np.ndarray | None = None,
        lr_min: float = 1e-5,
        lr_max: float = 1e-1,
        n_steps: int = 30,
        verbose: bool = False,
    ) -> RangeTestResult:
        """Run the range test.

        Args:
            train_ids: Training token IDs.
            val_ids: Optional validation IDs (unused in range test).
            lr_min: Starting LR.
            lr_max: Ending LR.
            n_steps: Number of LR values to test (log-spaced).
            verbose: If True, print progress.

        Returns:
            :class:`RangeTestResult` with the LR schedule and losses.
        """
        cfg = self.model.config
        seq_len = cfg.max_seq_len
        # Build sequences.
        trainer = TrainerV3(self.model, self.config)
        train_seqs = trainer._make_sequences(train_ids, seq_len)
        if not train_seqs:
            raise ValueError("Not enough tokens for seq_len")

        # Log-spaced LR schedule.
        lrs = np.logspace(math.log10(lr_min), math.log10(lr_max), n_steps)
        losses: list[float] = []

        rng = np.random.default_rng(self.config.seed)
        for i, lr in enumerate(lrs):
            # Pick a random sequence.
            seq = train_seqs[rng.integers(0, len(train_seqs))]
            # Forward + backward.
            input_ids = seq[:-1].astype(np.int64)
            target_ids = seq[1:].astype(np.int64)
            logits, cache = self.model.forward_with_cache(input_ids)
            from tiny_gpt_v3 import label_smoothing_cross_entropy

            loss, dlogits = label_smoothing_cross_entropy(
                logits,
                target_ids,
                smoothing=cfg.label_smoothing,
            )
            grads = self.model.backward(cache, dlogits)
            # Apply gradient clipping.
            from tiny_gpt_v3 import clip_grad_norm_

            if cfg.grad_clip_norm > 0:
                clip_grad_norm_(grads, cfg.grad_clip_norm)
            # Apply Adam update with this LR.
            trainer.optimizer.step(trainer._params, grads, lr=float(lr))
            trainer._sync_params_back()
            losses.append(float(loss))
            if verbose and (i + 1) % 5 == 0:
                print(f"  step {i + 1}/{n_steps}: lr={lr:.2e} loss={loss:.4f}")

        # Find the steepest descent (most negative d(loss)/d(log_lr)).
        log_lrs = np.log10(lrs)
        d_loss = np.diff(losses)
        d_lr = np.diff(log_lrs)
        # Avoid division by zero.
        gradients = np.where(np.abs(d_lr) > 1e-12, d_loss / d_lr, 0.0)
        # The steepest descent is the most negative gradient.
        if len(gradients) > 0:
            best_idx = int(np.argmin(gradients))
            steepest_lr = float(lrs[best_idx])
        else:
            steepest_lr = float(lrs[len(lrs) // 2])

        # Best LR = steepest descent LR (conservative: divide by 10).
        best_lr = steepest_lr / 10.0

        return RangeTestResult(
            lrs=lrs.tolist(),
            losses=losses,
            best_lr=best_lr,
            steepest_descent_lr=steepest_lr,
        )


# ---------------------------------------------------------------------------
# Grid Search
# ---------------------------------------------------------------------------
@dataclass
class GridSearchResult:
    """Result of an LR grid search.

    Attributes:
        lr_results: Dict mapping LR to final val loss.
        best_lr: LR with the lowest val loss.
        best_val_loss: The lowest val loss achieved.
    """

    lr_results: dict[float, float] = field(default_factory=dict)
    best_lr: float = 0.0
    best_val_loss: float = float("inf")


class LRGridSearch:
    """Grid search over learning rates.

    Runs a short training run for each LR in the grid, then picks the
    one with the lowest validation loss.

    Args:
        model: The TinyGPT v3 model (reset before each LR).
        config: Base training config.
    """

    def __init__(
        self,
        model: TinyGPTV3,
        config: TrainConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config or TrainConfig()
        self._init_state = self._snapshot()

    def _snapshot(self) -> dict[str, np.ndarray]:
        return {
            "token_emb": self.model.token_emb.copy(),
            "lm_head": self.model.lm_head.copy(),
            "ln_f_gamma": self.model.ln_f_gamma.copy(),
            "ln_f_beta": self.model.ln_f_beta.copy(),
            **{
                f"L{i}_{attr}": getattr(layer, attr).copy()
                for i, layer in enumerate(self.model.layers)
                for attr in (
                    "W_q",
                    "W_k",
                    "W_v",
                    "W_o",
                    "b_q",
                    "b_k",
                    "b_v",
                    "b_o",
                    "ln1_gamma",
                    "ln1_beta",
                    "ln2_gamma",
                    "ln2_beta",
                    "W_fc1",
                    "W_fc2",
                    "b_fc1",
                    "b_fc2",
                )
            },
        }

    def _restore(self) -> None:
        self.model.token_emb = self._init_state["token_emb"].copy()
        self.model.lm_head = self._init_state["lm_head"].copy()
        self.model.ln_f_gamma = self._init_state["ln_f_gamma"].copy()
        self.model.ln_f_beta = self._init_state["ln_f_beta"].copy()
        for i, layer in enumerate(self.model.layers):
            for attr in (
                "W_q",
                "W_k",
                "W_v",
                "W_o",
                "b_q",
                "b_k",
                "b_v",
                "b_o",
                "ln1_gamma",
                "ln1_beta",
                "ln2_gamma",
                "ln2_beta",
                "W_fc1",
                "W_fc2",
                "b_fc1",
                "b_fc2",
            ):
                setattr(layer, attr, self._init_state[f"L{i}_{attr}"].copy())

    def run(
        self,
        train_ids: np.ndarray,
        val_ids: np.ndarray,
        lr_grid: list[float] | None = None,
        verbose: bool = False,
    ) -> GridSearchResult:
        """Run the grid search.

        Args:
            train_ids: Training token IDs.
            val_ids: Validation token IDs.
            lr_grid: List of LRs to try. Default: [1e-4, 3e-4, 1e-3, 3e-3].
            verbose: If True, print progress.

        Returns:
            :class:`GridSearchResult` with the best LR.
        """
        if lr_grid is None:
            lr_grid = [1e-4, 3e-4, 1e-3, 3e-3]

        results: dict[float, float] = {}
        best_lr = 0.0
        best_val_loss = float("inf")

        for lr in lr_grid:
            if verbose:
                print(f"  Testing lr={lr:.2e}...")
            # Reset model.
            self._restore()
            # Train with this LR.
            cfg = TrainConfig(**self.config.__dict__)
            cfg.max_lr = lr
            cfg.verbose = False
            trainer = TrainerV3(self.model, cfg)
            history = trainer.train(train_ids, val_ids=val_ids)
            # Record final val loss.
            if history["val_losses"]:
                final_val = history["val_losses"][-1]
            else:
                final_val = history["losses"][-1]
            results[lr] = final_val
            if verbose:
                print(f"    final val_loss={final_val:.4f}")
            if final_val < best_val_loss:
                best_val_loss = final_val
                best_lr = lr

        return GridSearchResult(
            lr_results=results,
            best_lr=best_lr,
            best_val_loss=best_val_loss,
        )


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("LR Search v3 — smoke test")
    print("=" * 60)
    from tiny_gpt_v3 import config_small

    cfg = config_small()
    cfg.weight_tying = True
    cfg.label_smoothing = 0.0
    cfg.dropout = 0.0
    cfg.grad_clip_norm = 1.0
    model = TinyGPTV3(cfg)
    print(f"Model params: {cfg.params_count:,}")

    rng = np.random.default_rng(42)
    corpus = rng.integers(0, cfg.vocab_size, size=1000).astype(np.int64)
    val = rng.integers(0, cfg.vocab_size, size=300).astype(np.int64)

    print("\n--- LR Range Test ---")
    rt = LRRangeTest(model, TrainConfig(epochs=1, verbose=False))
    rt_result = rt.run(corpus, val, n_steps=15, verbose=True)
    print(f"\nSteepest descent LR: {rt_result.steepest_descent_lr:.2e}")
    print(f"Recommended LR (÷10): {rt_result.best_lr:.2e}")

    print("\n--- LR Grid Search ---")
    gs = LRGridSearch(
        model, TrainConfig(epochs=3, eval_interval=1, early_stopping_patience=0, verbose=False)
    )
    gs_result = gs.run(corpus, val, lr_grid=[1e-3, 3e-3, 1e-2], verbose=True)
    print(f"\nBest LR: {gs_result.best_lr:.2e}")
    print(f"Best val_loss: {gs_result.best_val_loss:.4f}")
    for lr, loss in gs_result.lr_results.items():
        print(f"  lr={lr:.2e}: val_loss={loss:.4f}")
