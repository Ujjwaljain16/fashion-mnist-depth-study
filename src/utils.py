"""
utils.py — Configuration, seeding, data loading, and results I/O.

This module is the single source of truth for all hyperparameters (CONFIG).
All other modules import from here; nothing is duplicated.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchvision import datasets


# ─────────────────────────────────────────────────────────────────────────────
# Master Configuration — only change hyperparameters here
# ─────────────────────────────────────────────────────────────────────────────

CONFIG: dict[str, Any] = {
    # Dataset
    "dataset":      "FashionMNIST",
    "input_dim":    784,         # 28 × 28 flattened
    "num_classes":  10,
    "train_size":   54_000,
    "val_size":     6_000,
    "norm_mean":    0.5,
    "norm_std":     0.5,

    # Training (full mode)
    "batch_size":   256,
    "lr":           1e-3,
    "optimizer":    "Adam",
    "loss":         "CrossEntropyLoss",
    "epochs":       50,
    "seeds":        [42, 123, 7],

    # Training (FAST_MODE)
    "fast_epochs":  10,
    "fast_seeds":   [42],

    # Architecture grid — widths chosen to keep params ≈ 500K each depth
    "depths":       [2, 4, 8],
    "widths":       {2: 413, 4: 296, 8: 215},

    # Activations tested
    "activations":  ["ReLU", "Sigmoid"],

    # Convergence: epoch first reaching (threshold × max val_acc)
    "convergence_threshold": 0.95,

    # Output paths (relative to project root; notebook sets CWD to project root)
    "data_dir":        "data",
    "results_path":    "results/results.csv",
    "summary_path":    "results/summary.csv",
    "figures_dir":     "results/figures",
    "checkpoints_dir": "results/checkpoints",
    "exports_dir":     "results/exports",
}

# Number of gradient columns emitted per row (= max depth).
# Always emit layer_0_grad … layer_7_grad; shallower models get None.
# Public constant: imported by train.py to build fixed-width CSV rows.
MAX_DEPTH_LAYERS: int = max(CONFIG["depths"])   # 8

CLASS_NAMES: list[str] = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """
    Set all global random seeds for full reproducibility.

    Must be called at three points per run:
      1. Before dataset split  → consistent validation split
      2. Before model init     → consistent weight initialisation
      3. Before training loop  → consistent DataLoader shuffle ordering

    Design note:
      torch.use_deterministic_algorithms(True) is intentionally omitted.
      It raises RuntimeError on some CUDA operations in Colab GPU environments.
      torch.backends.cudnn.deterministic = True achieves the same result for
      all linear and conv operations without introducing runtime crashes.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloaders(
    data_dir: str,
    batch_size: int,
    val_size: int,
    train_size: int,
    norm_mean: float,
    norm_std: float,
    seed: int = 42,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Download FashionMNIST and return (train, val, test) DataLoaders.

    Normalisation:
        After ToTensor(), pixel values ∈ [0.0, 1.0].
        After Normalize(0.5, 0.5), values ∈ [−1.0, 1.0].
        Centering near zero accelerates convergence and reduces initial
        Sigmoid saturation — justified by the spec and standard for MNIST-family.

    Args:
        data_dir:    Root directory for dataset download (created if absent).
        batch_size:  Mini-batch size.
        val_size:    Samples held out from train for validation.
        train_size:  Remaining training samples.
        norm_mean:   Normalisation mean (per-channel scalar for grayscale).
        norm_std:    Normalisation std.
        seed:        RNG seed for reproducible train/val split. Use the same
                     value (42) across all experiments so all runs share the
                     identical validation partition.
        num_workers: DataLoader workers. Use 0 on Windows (Jupyter limitation).

    Returns:
        (train_loader, val_loader, test_loader)
    """
    # Download standard uint8 datasets
    os.makedirs(data_dir, exist_ok=True)
    full_train = datasets.FashionMNIST(data_dir, train=True, download=True)
    test_set = datasets.FashionMNIST(data_dir, train=False, download=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Vectorized Pre-processing (CRITICAL FOR COLAB T4 PERFORMANCE)
    # Standard FashionMNIST.__getitem__ uses slow PIL Image conversion per item.
    # We bypass this by vectorizing the entire dataset instantly into VRAM/RAM.
    # ─────────────────────────────────────────────────────────────────────────
    x_train = full_train.data.float().unsqueeze(1) / 255.0
    x_train = (x_train - norm_mean) / norm_std
    y_train = full_train.targets

    x_test = test_set.data.float().unsqueeze(1) / 255.0
    x_test = (x_test - norm_mean) / norm_std
    y_test = test_set.targets

    full_train_ds = TensorDataset(x_train, y_train)
    test_dataset_vec = TensorDataset(x_test, y_test)

    # Reproducible split — generator is independent of the run seed
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_train_ds, [train_size, val_size], generator=generator
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=False, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset_vec, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False, pin_memory=True
    )
    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# Run Identifier
# ─────────────────────────────────────────────────────────────────────────────

def make_run_id(depth: int, activation: str, batchnorm: bool, seed: int, epochs: int) -> str:
    """
    Generate a deterministic unique identifier for a single run.

    Format: d{depth}_{activation}_bn{0/1}_s{seed}_e{epochs}
    Example: d8_relu_bn0_s42_e50

    Why this matters:
        If the user switches from FAST_MODE (10 epochs) to full mode (50 epochs) while
        AUTO_RESUME=True, the full-mode runs would incorrectly be skipped if epochs were
        not part of the run_id.

    To switch modes safely:
        1. Delete results/results.csv and results/summary.csv, OR
        2. Set AUTO_RESUME=False to force re-training.
    """
    return f"d{depth}_{activation.lower()}_bn{int(batchnorm)}_s{seed}_e{epochs}"


# ─────────────────────────────────────────────────────────────────────────────
# Results I/O — pandas-based for safe column-heterogeneous writes
# ─────────────────────────────────────────────────────────────────────────────

def run_exists(run_id: str, results_path: str) -> bool:
    """
    Return True if run_id already exists in the results CSV.

    Used by AUTO_RESUME to skip completed runs without re-training.
    """
    p = Path(results_path)
    if not p.exists():
        return False
    try:
        df = pd.read_csv(p, usecols=["run_id"])
        return run_id in df["run_id"].values
    except (pd.errors.EmptyDataError, ValueError):
        return False


def _safe_write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write DataFrame to CSV using true append mode for fault tolerance."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    df.to_csv(path, mode='a', header=write_header, index=False)


def append_epoch_rows(rows: list[dict[str, Any]], results_path: str) -> None:
    """
    Append epoch-level rows to results CSV safely.
    Uses pandas to_csv(mode='a') to ensure atomic, non-destructive appends.
    """
    if not rows:
        return
    p = Path(results_path)
    df_new = pd.DataFrame(rows)
    _safe_write_csv(df_new, p)


def append_summary_row(row: dict[str, Any], summary_path: str) -> None:
    """Append a single per-run summary row to summary CSV."""
    p = Path(summary_path)
    df_new = pd.DataFrame([row])
    _safe_write_csv(df_new, p)


def load_results(results_path: str) -> pd.DataFrame:
    """Load epoch-level results CSV. Returns empty DataFrame if absent."""
    p = Path(results_path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def load_summary(summary_path: str) -> pd.DataFrame:
    """Load per-run summary CSV. Returns empty DataFrame if absent."""
    p = Path(summary_path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def compute_aggregate_summary(summary_path: str) -> pd.DataFrame:
    """
    Aggregate summary.csv across seeds → mean ± std per model configuration.

    Groups by (depth, width, activation, batchnorm, parameter_count) and
    computes mean and std for test_acc, convergence_epoch, and
    gradient_attenuation_ratio.

    Returns:
        DataFrame with one row per unique model configuration.
    """
    df = load_summary(summary_path)
    if df.empty:
        return df

    group_cols = ["depth", "width", "activation", "batchnorm", "parameter_count"]
    agg = (
        df.groupby(group_cols)
        .agg(
            mean_test_acc=("test_acc", "mean"),
            std_test_acc=("test_acc", "std"),
            mean_convergence_epoch=("convergence_epoch", "mean"),
            std_convergence_epoch=("convergence_epoch", "std"),
            mean_gradient_ratio=("gradient_attenuation_ratio", "mean"),
            std_gradient_ratio=("gradient_attenuation_ratio", "std"),
            n_seeds=("seed", "count"),
        )
        .reset_index()
    )
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# Metric Helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_convergence_epoch(
    val_accs: list[float],
    threshold: float = 0.95,
) -> int:
    """
    Return the first epoch (1-indexed) where val_acc ≥ threshold × max(val_accs).

    Definition: "Epoch required to reach 95% of maximum validation accuracy
    observed during training."

    This definition is more stable than "99% of final accuracy" because it:
      - Is robust to accuracy dips in later epochs (plateau fluctuations)
      - Correctly handles cases where the final epoch is not the best epoch
      - Provides a monotone target (max so far) rather than a retroactive one

    Returns 0 if the threshold is never reached (degenerate / non-converging run).
    """
    if not val_accs:
        return 0
    target = threshold * max(val_accs)
    for epoch_idx, acc in enumerate(val_accs):
        if acc >= target:
            return epoch_idx + 1   # 1-indexed epoch number
    return 0
