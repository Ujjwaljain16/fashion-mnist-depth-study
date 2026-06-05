"""
train.py — Training loop, gradient capture, and experiment orchestration.

Design decisions:
  - train_one_run() is a pure function: same inputs → same outputs (given seed).
  - Gradient norms are captured on the last mini-batch of each epoch for all
    hidden Linear layers. See _capture_layer_grad_norms() for the rationale.
  - AUTO_RESUME: if run_id exists in results CSV, training is skipped entirely.
  - Experiments 1 and 2 share ReLU runs; AUTO_RESUME handles deduplication
    so no run is executed twice.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models import ConfigurableMLP
from src.utils import (
    CONFIG,
    MAX_DEPTH_LAYERS,
    append_epoch_rows,
    append_summary_row,
    compute_convergence_epoch,
    get_dataloaders,
    make_run_id,
    run_exists,
    set_seed,
)


# ─────────────────────────────────────────────────────────────────────────────
# Result Container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RunResult:
    """
    Immutable-by-convention container for all metrics from one training run.

    Gradient storage:
        per_layer_grads[epoch_idx][layer_idx] = L2 norm of that layer's
        weight gradient tensor, captured after backward on the last batch.

    Gradient attenuation ratio:
        first_hidden_grad / last_hidden_grad (computed at the final epoch).
        A higher value means early layers receive much weaker gradient signal
        than late layers — the defining signature of vanishing gradients.
    """
    run_id: str
    seed: int
    depth: int
    activation: str
    use_batchnorm: bool
    parameter_count: int
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    train_accs: list[float] = field(default_factory=list)
    val_accs: list[float] = field(default_factory=list)
    per_layer_grads: list[list[float]] = field(default_factory=list)
    test_acc: float = 0.0
    convergence_epoch: int = 0
    gradient_attenuation_ratio: float = 0.0
    elapsed_seconds: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Gradient Capture
# ─────────────────────────────────────────────────────────────────────────────

def _capture_layer_grad_norms(model: ConfigurableMLP) -> list[float]:
    """
    Capture the L2 norm of each hidden Linear layer's weight gradient.

    Must be called AFTER loss.backward() and BEFORE optimizer.step().
    At this point all .weight.grad tensors are populated and unmodified.

    Why last mini-batch, not epoch-average:
        Averaging gradients across all batches requires accumulating tensors
        every step: O(L × steps_per_epoch) memory and time overhead.
        For this study, last-batch norms are a stable, reproducible proxy
        adequate for diagnosing gradient health. The per-epoch trend in the
        plots is informative; exact magnitude is secondary.

    Why not average-over-batch:
        Per-batch averaging would be smoother but adds:
          - Storage: L × steps tensors kept alive per epoch
          - Compute: L running sum operations per batch
        For ~500K param, 8-layer models across 54 runs this is non-trivial.
        Last-batch gives equivalent diagnostic value.

    Args:
        model: ConfigurableMLP with gradients already computed.

    Returns:
        List of float, length = number of hidden Linear layers.
        0.0 for any layer whose .grad is None (shouldn't occur during training).
    """
    norms: list[float] = []
    for layer in model.get_hidden_linear_layers():
        if layer.weight.grad is not None:
            norms.append(layer.weight.grad.norm(2).item())
        else:
            norms.append(0.0)
    return norms


# ─────────────────────────────────────────────────────────────────────────────
# Core Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def train_one_run(
    model: ConfigurableMLP,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    convergence_threshold: float = 0.95,
    verbose: bool = True,
) -> RunResult:
    """
    Execute one complete training run and return a populated RunResult.

    Training loop structure:
        for epoch:
            model.train()
            for batch:
                zero_grad → forward → loss → backward → [capture grads if last batch] → step
            model.eval()
            compute val_loss, val_acc
        model.eval()
        compute test_acc
        compute convergence_epoch, gradient_attenuation_ratio

    Args:
        model:                  Initialised ConfigurableMLP (not yet on device).
        train_loader:           Training DataLoader.
        val_loader:             Validation DataLoader.
        test_loader:            Test DataLoader (evaluated once, post-training).
        epochs:                 Number of training epochs.
        lr:                     Adam learning rate.
        device:                 Compute device.
        convergence_threshold:  Fraction of max val_acc defining convergence.
        verbose:                Print per-epoch progress.

    Returns:
        RunResult with all metrics populated. run_id and seed are set by the
        caller (ExperimentRunner._run_single) after this function returns.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    result = RunResult(
        run_id="",
        seed=0,
        depth=model.depth,
        activation=model.activation_name,
        use_batchnorm=model.use_batchnorm,
        parameter_count=model.count_parameters(),
    )

    start = time.time()

    for epoch in range(1, epochs + 1):
        # ── Training Phase ────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        last_batch_grads: list[float] = []
        num_batches = len(train_loader)

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            # Capture gradients on last mini-batch of this epoch
            # (after backward, before step — grads are populated and unmodified)
            if batch_idx == num_batches - 1:
                last_batch_grads = _capture_layer_grad_norms(model)

            optimizer.step()

            epoch_loss += loss.item() * x.size(0)
            epoch_correct += (logits.detach().argmax(1) == y).sum().item()
            epoch_total += x.size(0)

        train_loss = epoch_loss / epoch_total
        train_acc  = epoch_correct / epoch_total

        # ── Validation Phase ──────────────────────────────────────────────
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss_sum += loss.item() * x.size(0)
                val_correct += (logits.argmax(1) == y).sum().item()
                val_total += x.size(0)

        val_loss = val_loss_sum / val_total
        val_acc  = val_correct  / val_total

        result.train_losses.append(round(train_loss, 6))
        result.val_losses.append(round(val_loss, 6))
        result.train_accs.append(round(train_acc, 6))
        result.val_accs.append(round(val_acc, 6))
        result.per_layer_grads.append(last_batch_grads)

        if verbose:
            print(
                f"  Epoch {epoch:3d}/{epochs} | "
                f"Loss {train_loss:.4f}/{val_loss:.4f} | "
                f"Acc  {train_acc:.4f}/{val_acc:.4f}"
            )

    # ── Post-Training ─────────────────────────────────────────────────────
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            test_correct += (model(x).argmax(1) == y).sum().item()
            test_total += x.size(0)

    result.test_acc = round(test_correct / test_total, 6)
    result.convergence_epoch = compute_convergence_epoch(
        result.val_accs, convergence_threshold
    )

    # Gradient attenuation ratio at final epoch
    # Definition: first_hidden_grad / last_hidden_grad
    # Interpretation: ratio >> 1 → early layers barely trained → vanishing gradient
    if result.per_layer_grads:
        final_grads = result.per_layer_grads[-1]
        if len(final_grads) >= 2 and final_grads[-1] > 1e-12:
            result.gradient_attenuation_ratio = round(
                final_grads[0] / final_grads[-1], 6
            )
        else:
            result.gradient_attenuation_ratio = 0.0

    result.elapsed_seconds = round(time.time() - start, 1)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Experiment Runner
# ─────────────────────────────────────────────────────────────────────────────

class ExperimentRunner:
    """
    Orchestrates all training runs across the three experiments.

    Key behaviours:
      - AUTO_RESUME: skips any run_id already in results CSV, enabling safe
        re-runs after Colab disconnects without duplicating data.
      - Shared data loaders: created once with seed=42 for reproducible val
        split, then reused across all runs. Per-run randomness comes from
        set_seed(seed) which affects mini-batch ordering.
      - Results persistence: epoch rows and summary rows are written after
        each run, so partial progress is preserved even if the session ends.
    """

    def __init__(
        self,
        config: dict[str, Any],
        fast_mode: bool = False,
        auto_resume: bool = True,
        device: torch.device | None = None,
        num_workers: int = 0,
        verbose: bool = True,
    ) -> None:
        self.config = config
        self.fast_mode = fast_mode
        self.auto_resume = auto_resume
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.num_workers = num_workers
        self.verbose = verbose

        self.epochs: int = config["fast_epochs"] if fast_mode else config["epochs"]
        self.seeds: list[int] = config["fast_seeds"] if fast_mode else config["seeds"]

        # Data loaders — lazily initialised once, reused across all runs
        self._train_loader: DataLoader | None = None
        self._val_loader:   DataLoader | None = None
        self._test_loader:  DataLoader | None = None

    # ── Internal Utilities ────────────────────────────────────────────────────

    def _ensure_dataloaders(self) -> None:
        """
        Initialise DataLoaders on first call using a fixed seed (42) for the
        train/val split. Subsequent calls are no-ops.

        The split seed is intentionally fixed and independent of the run seed,
        so all runs share the identical validation partition.
        """
        if self._train_loader is not None:
            return
        if self.verbose:
            print("  Initialising DataLoaders (split seed=42)…")
        self._train_loader, self._val_loader, self._test_loader = get_dataloaders(
            data_dir=self.config["data_dir"],
            batch_size=self.config["batch_size"],
            val_size=self.config["val_size"],
            train_size=self.config["train_size"],
            norm_mean=self.config["norm_mean"],
            norm_std=self.config["norm_std"],
            seed=42,
            num_workers=self.num_workers,
        )

    def _build_epoch_row(
        self,
        result: RunResult,
        depth: int,
        width: int,
        activation: str,
        use_batchnorm: bool,
        epoch_idx: int,
    ) -> dict[str, Any]:
        """
        Build one epoch-level CSV row.

        Always emits layer_0_grad … layer_7_grad columns (filled with None
        for layers beyond the model's depth) so the CSV schema is fixed
        regardless of model depth.
        """
        row: dict[str, Any] = {
            "run_id":          result.run_id,
            "seed":            result.seed,
            "depth":           depth,
            "activation":      activation,
            "batchnorm":       int(use_batchnorm),
            "parameter_count": result.parameter_count,
            "epoch":           epoch_idx + 1,
            "train_loss":      result.train_losses[epoch_idx],
            "val_loss":        result.val_losses[epoch_idx],
            "train_acc":       result.train_accs[epoch_idx],
            "val_acc":         result.val_accs[epoch_idx],
        }
        # Emit all possible gradient columns; None for layers that don't exist
        epoch_grads = (
            result.per_layer_grads[epoch_idx]
            if epoch_idx < len(result.per_layer_grads)
            else []
        )
        for li in range(MAX_DEPTH_LAYERS):
            row[f"layer_{li}_grad"] = (
                round(epoch_grads[li], 8) if li < len(epoch_grads) else None
            )
        return row

    def _save_results(
        self,
        result: RunResult,
        depth: int,
        width: int,
        activation: str,
        use_batchnorm: bool,
    ) -> None:
        """Write epoch-level rows and summary row to CSV files."""
        epoch_rows = [
            self._build_epoch_row(result, depth, width, activation, use_batchnorm, i)
            for i in range(len(result.train_losses))
        ]
        append_epoch_rows(epoch_rows, self.config["results_path"])

        summary_row: dict[str, Any] = {
            "run_id":                    result.run_id,
            "seed":                      result.seed,
            "depth":                     depth,
            "width":                     width,
            "activation":                activation,
            "batchnorm":                 int(use_batchnorm),
            "parameter_count":           result.parameter_count,
            "test_acc":                  result.test_acc,
            "convergence_epoch":         result.convergence_epoch,
            "gradient_attenuation_ratio": result.gradient_attenuation_ratio,
        }
        append_summary_row(summary_row, self.config["summary_path"])

    def _run_single(
        self,
        depth: int,
        activation: str,
        use_batchnorm: bool,
        seed: int,
    ) -> RunResult | None:
        """
        Execute a single training run, respecting AUTO_RESUME.

        Steps:
          1. Build run_id; check CSV — skip if AUTO_RESUME and already exists.
          2. set_seed(seed)  — controls model init + DataLoader shuffle order.
          3. Build ConfigurableMLP with correct width from CONFIG.
          4. Call train_one_run().
          5. Attach run_id and seed to result.
          6. Save to CSV.

        Returns None if the run was skipped (AUTO_RESUME).
        """
        run_id = make_run_id(depth, activation, use_batchnorm, seed, self.epochs)

        if self.auto_resume and run_exists(run_id, self.config["results_path"]):
            if self.verbose:
                print(f"  [SKIP] {run_id} — already in results CSV.")
            return None

        width = self.config["widths"][depth]
        set_seed(seed)
        self._ensure_dataloaders()

        model = ConfigurableMLP(
            input_dim=self.config["input_dim"],
            num_classes=self.config["num_classes"],
            depth=depth,
            width=width,
            activation=activation,
            use_batchnorm=use_batchnorm,
        )

        if self.verbose:
            print(f"\n{'─'*60}")
            print(f"  Run: {run_id}")
            print(f"  {model}")
            print(f"  Device: {self.device}")
            print(f"{'─'*60}")

        result = train_one_run(
            model=model,
            train_loader=self._train_loader,
            val_loader=self._val_loader,
            test_loader=self._test_loader,
            epochs=self.epochs,
            lr=self.config["lr"],
            device=self.device,
            convergence_threshold=self.config["convergence_threshold"],
            verbose=self.verbose,
        )
        result.run_id = run_id
        result.seed   = seed

        self._save_results(result, depth, width, activation, use_batchnorm)

        if self.verbose:
            print(
                f"\n  ✓ Test Acc: {result.test_acc:.4f} | "
                f"Convergence: Epoch {result.convergence_epoch} | "
                f"Grad Ratio: {result.gradient_attenuation_ratio:.4f} | "
                f"Time: {result.elapsed_seconds:.1f}s"
            )

        return result

    # ── Experiment Entry Points ───────────────────────────────────────────────

    def run_experiment_1(self) -> None:
        """
        Experiment 1: Depth Comparison (ReLU, no BatchNorm).

        Research Question — RQ1:
            Does increasing depth improve performance under a fixed parameter budget?

        Configurations: depths [2, 4, 8] × seeds × ReLU, no BN.
        """
        print("\n" + "═" * 60)
        print("EXPERIMENT 1: Depth Comparison")
        print("  Activation=ReLU | BatchNorm=False | All depths")
        print("═" * 60)
        for depth in self.config["depths"]:
            for seed in self.seeds:
                self._run_single(depth=depth, activation="ReLU",
                                 use_batchnorm=False, seed=seed)

    def run_experiment_2(self) -> None:
        """
        Experiment 2: Activation Study (ReLU vs Sigmoid, all depths).

        Research Questions — RQ2 + RQ3:
            How does depth affect gradient flow?
            What role does activation choice play?

        Note: ReLU runs were already executed in Experiment 1.
              AUTO_RESUME prevents re-training them.
        """
        print("\n" + "═" * 60)
        print("EXPERIMENT 2: Activation Study")
        print("  ReLU vs Sigmoid | All depths | BatchNorm=False")
        print("  (ReLU runs reused from Experiment 1 via AUTO_RESUME)")
        print("═" * 60)
        for depth in self.config["depths"]:
            for activation in self.config["activations"]:
                for seed in self.seeds:
                    self._run_single(depth=depth, activation=activation,
                                     use_batchnorm=False, seed=seed)

    def run_experiment_3(self) -> None:
        """
        Experiment 3: BatchNorm Recovery (8L Sigmoid ± BatchNorm).

        Research Question — RQ4:
            Can BatchNorm restore trainability to a deep Sigmoid network?

        Note: 8L Sigmoid (no BN) was already run in Experiment 2.
              AUTO_RESUME prevents re-training.
        """
        print("\n" + "═" * 60)
        print("EXPERIMENT 3: BatchNorm Recovery")
        print("  depth=8 | Sigmoid | BN=False vs BN=True")
        print("═" * 60)
        for use_batchnorm in [False, True]:
            for seed in self.seeds:
                self._run_single(depth=8, activation="Sigmoid",
                                 use_batchnorm=use_batchnorm, seed=seed)

    def run_all(self) -> None:
        """Run all three experiments in sequence."""
        self.run_experiment_1()
        self.run_experiment_2()
        self.run_experiment_3()
        print("\n" + "═" * 60)
        print("ALL EXPERIMENTS COMPLETE")
        print(f"  Results → {self.config['results_path']}")
        print(f"  Summary → {self.config['summary_path']}")
        print("═" * 60)
