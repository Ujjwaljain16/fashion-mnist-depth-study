"""
plots.py — All required figure functions for the Fashion-MNIST depth study.

All functions read from results CSV files and are fully decoupled from training.
This means figures can be regenerated without re-running experiments.

Figures generated:
    Fig 1:  Validation Accuracy vs Epoch (2L/4L/8L ReLU, mean ± std shading)
    Fig 2:  Validation Loss vs Epoch     (2L/4L/8L ReLU, mean ± std shading)
    Fig 3:  Final Test Accuracy Bar Chart (all configs, error bars = std)
    Fig 4A: Gradient Norm vs Layer Depth (8L ReLU vs 8L Sigmoid, final epoch) [PRIMARY]
    Fig 4B: Gradient Norm vs Epoch       (first + last hidden layer, log scale) [SUPPLEMENTAL]
    Fig 5:  Activation Heatmap           (3×2 grid, cell = mean test accuracy)
    Fig 6:  BatchNorm Recovery           (val_acc + gradient_ratio, 2-panel)

Shared conventions:
    - Colorblind-safe palette (seaborn "colorblind")
    - PNG saved at 300 DPI to figures_dir
    - ± shading = ±1 std across seeds
    - Gradient axes use log scale (better visualises orders-of-magnitude differences)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns


# ─────────────────────────────────────────────────────────────────────────────
# Shared Style Constants
# ─────────────────────────────────────────────────────────────────────────────

_PALETTE = sns.color_palette("colorblind")
_DEPTH_COLORS: dict[int, Any]       = {2: _PALETTE[0], 4: _PALETTE[1], 8: _PALETTE[2]}
_ACTIVATION_COLORS: dict[str, Any]  = {"ReLU": _PALETTE[0], "Sigmoid": _PALETTE[2]}
_BN_COLORS: dict[int, Any]          = {0: _PALETTE[2], 1: _PALETTE[4]}

plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.size":       11,
    "axes.titlesize":  13,
    "axes.labelsize":  12,
    "legend.fontsize": 10,
    "figure.dpi":      150,
    "savefig.dpi":     300,
    "axes.grid":       True,
    "grid.alpha":      0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _save(fig: plt.Figure, figures_dir: str, filename: str) -> None:
    """Save figure to disk and display inline."""
    os.makedirs(figures_dir, exist_ok=True)
    filepath = os.path.join(figures_dir, filename)
    fig.savefig(filepath, bbox_inches="tight", dpi=300)
    print(f"  Saved → {filepath}")
    plt.show()
    plt.close(fig)


def _get_grad_cols(df: pd.DataFrame) -> list[str]:
    """Return sorted list of layer_N_grad column names present in df."""
    cols = [c for c in df.columns if c.startswith("layer_") and c.endswith("_grad")]
    return sorted(cols, key=lambda c: int(c.split("_")[1]))


def _load_and_validate(path: str, label: str = "results") -> pd.DataFrame | None:
    """Load CSV; return None with a warning if absent or empty."""
    p = Path(path)
    if not p.exists():
        print(f"  [WARNING] {label} CSV not found at {path}. Run experiments first.")
        return None
    try:
        df = pd.read_csv(p)
        if df.empty:
            print(f"  [WARNING] {label} CSV is empty.")
            return None
        return df
    except Exception as exc:
        print(f"  [ERROR] Could not read {path}: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Validation Accuracy vs Epoch (Depth Comparison)
# ─────────────────────────────────────────────────────────────────────────────

def plot_val_accuracy_curves(
    results_path: str,
    figures_dir: str,
    activation: str = "ReLU",
    batchnorm: int = 0,
) -> None:
    """
    Figure 1: Validation Accuracy vs Epoch for 2L, 4L, 8L.

    Addresses RQ1: Does increasing depth improve performance?

    Args:
        results_path: Path to epoch-level results CSV.
        figures_dir:  Directory to save PNG.
        activation:   Filter to this activation type. Default "ReLU".
        batchnorm:    Filter to this BN flag (0=off, 1=on). Default 0.
    """
    df = _load_and_validate(results_path, "Experiment 1 results")
    if df is None:
        return

    mask = (df["activation"] == activation) & (df["batchnorm"] == batchnorm)
    df = df[mask]
    if df.empty:
        print(f"  [WARNING] No rows matching activation={activation}, batchnorm={batchnorm}.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for depth in [2, 4, 8]:
        sub = df[df["depth"] == depth]
        if sub.empty:
            continue
        grouped = sub.groupby("epoch")["val_acc"].agg(["mean", "std"]).reset_index()
        epochs = grouped["epoch"].values
        mean   = grouped["mean"].values
        std    = grouped["std"].fillna(0).values

        c = _DEPTH_COLORS[depth]
        ax.plot(epochs, mean, color=c, label=f"{depth}L MLP", linewidth=2.0)
        ax.fill_between(epochs, mean - std, mean + std, color=c, alpha=0.15)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title(f"Fig 1: Validation Accuracy vs Epoch\n({activation}, BatchNorm={'On' if batchnorm else 'Off'})")
    ax.legend(loc="lower right")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))
    ax.set_ylim(bottom=0.0, top=1.01)
    fig.tight_layout()
    _save(fig, figures_dir, "fig1_val_accuracy.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Validation Loss vs Epoch
# ─────────────────────────────────────────────────────────────────────────────

def plot_val_loss_curves(
    results_path: str,
    figures_dir: str,
    activation: str = "ReLU",
    batchnorm: int = 0,
) -> None:
    """
    Figure 2: Validation Loss vs Epoch for 2L, 4L, 8L.

    Complements Fig 1 — loss curves can reveal convergence speed differences
    even when accuracy differences are small.

    Args:
        results_path: Path to epoch-level results CSV.
        figures_dir:  Directory to save PNG.
        activation:   Filter to this activation type. Default "ReLU".
        batchnorm:    Filter to this BN flag. Default 0.
    """
    df = _load_and_validate(results_path, "Experiment 1 results")
    if df is None:
        return

    mask = (df["activation"] == activation) & (df["batchnorm"] == batchnorm)
    df = df[mask]
    if df.empty:
        print(f"  [WARNING] No data for activation={activation}, batchnorm={batchnorm}.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for depth in [2, 4, 8]:
        sub = df[df["depth"] == depth]
        if sub.empty:
            continue
        grouped = sub.groupby("epoch")["val_loss"].agg(["mean", "std"]).reset_index()
        epochs = grouped["epoch"].values
        mean   = grouped["mean"].values
        std    = grouped["std"].fillna(0).values

        c = _DEPTH_COLORS[depth]
        ax.plot(epochs, mean, color=c, label=f"{depth}L MLP", linewidth=2.0)
        ax.fill_between(epochs, mean - std, mean + std, color=c, alpha=0.15)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss (CrossEntropy)")
    ax.set_title(f"Fig 2: Validation Loss vs Epoch\n({activation}, BatchNorm={'On' if batchnorm else 'Off'})")
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, figures_dir, "fig2_val_loss.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Final Test Accuracy Bar Chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_test_accuracy_bar(
    summary_path: str,
    figures_dir: str,
) -> None:
    """
    Figure 3: Bar chart of mean test accuracy ± std for all 6 base configurations
    (3 depths × 2 activations, no BatchNorm).

    Error bars = std across seeds. Bar values annotated as "X.XXX ± Y.YYY".
    Addresses RQ1 and RQ3.

    Args:
        summary_path: Path to per-run summary CSV.
        figures_dir:  Directory to save PNG.
    """
    df = _load_and_validate(summary_path, "summary")
    if df is None:
        return

    # No-BN configs only; BN shown in Fig 6
    df = df[df["batchnorm"] == 0].copy()
    agg = (
        df.groupby(["depth", "activation"])
        .agg(mean_acc=("test_acc", "mean"), std_acc=("test_acc", "std"))
        .reset_index()
    )
    agg["label"] = agg.apply(
        lambda r: f"{r['depth']}L\n{r['activation']}", axis=1
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    bar_colors = [
        _DEPTH_COLORS[row["depth"]] if row["activation"] == "ReLU"
        else _ACTIVATION_COLORS["Sigmoid"]
        for _, row in agg.iterrows()
    ]

    bars = ax.bar(
        agg["label"],
        agg["mean_acc"],
        yerr=agg["std_acc"].fillna(0),
        color=bar_colors,
        edgecolor="black",
        linewidth=0.8,
        capsize=5,
        alpha=0.85,
        error_kw={"elinewidth": 1.5},
    )

    for bar, (_, row) in zip(bars, agg.iterrows()):
        m = row["mean_acc"]
        s = row["std_acc"] if not pd.isna(row["std_acc"]) else 0.0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{m:.3f}\n±{s:.3f}",
            ha="center", va="bottom", fontsize=8.5,
        )

    ax.set_ylabel("Mean Test Accuracy")
    ax.set_title("Fig 3: Final Test Accuracy by Configuration\n(mean ± std across seeds, no BatchNorm)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))

    y_min = max(0.0, agg["mean_acc"].min() - 0.06)
    y_max = min(1.01, agg["mean_acc"].max() + 0.10)
    ax.set_ylim(y_min, y_max)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color=_DEPTH_COLORS[d], label=f"{d}L ReLU") for d in [2, 4, 8]
    ] + [Patch(color=_ACTIVATION_COLORS["Sigmoid"], label="Sigmoid (all depths)")]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9)

    fig.tight_layout()
    _save(fig, figures_dir, "fig3_test_accuracy_bar.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4A — Gradient Norm vs Layer Depth (PRIMARY gradient figure)
# ─────────────────────────────────────────────────────────────────────────────

def plot_gradient_norm_by_layer(
    results_path: str,
    figures_dir: str,
) -> None:
    """
    Figure 4A (PRIMARY): Gradient L2 norm vs layer index at the FINAL epoch.

    Compares 8L ReLU vs 8L Sigmoid (no BatchNorm).

    Reading the plot:
        X-axis: Layer index 1…8 (1 = closest to input, 8 = closest to output)
        Y-axis: L2 norm of weight gradient (log scale)

        A declining curve from right to left indicates that early layers
        receive weaker gradient signal — the signature of vanishing gradients.

    The gradient attenuation ratio = first_layer_norm / last_layer_norm.
    A ratio >> 1 for Sigmoid vs ~1 for ReLU directly demonstrates the problem.

    Addresses RQ2 and RQ3.
    """
    df = _load_and_validate(results_path, "Experiment 2 results")
    if df is None:
        return

    df8 = df[(df["depth"] == 8) & (df["batchnorm"] == 0)].copy()
    grad_cols = _get_grad_cols(df8)

    if not grad_cols:
        print("  [WARNING] No gradient columns found. Run experiments first.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for activation, color, label in [
        ("ReLU",    _ACTIVATION_COLORS["ReLU"],    "8L ReLU"),
        ("Sigmoid", _ACTIVATION_COLORS["Sigmoid"], "8L Sigmoid"),
    ]:
        sub = df8[df8["activation"] == activation]
        if sub.empty:
            print(f"  [WARNING] No data for 8L {activation}.")
            continue

        final_epoch = sub["epoch"].max()
        final = sub[sub["epoch"] == final_epoch]

        means = final[grad_cols].mean()
        stds  = final[grad_cols].std().fillna(0)
        layer_indices = [int(c.split("_")[1]) + 1 for c in grad_cols]

        # Filter out None/NaN columns (from shallower models stored in same CSV)
        valid_mask = means.notna() & (means > 0)
        valid_cols = [c for c, v in zip(grad_cols, valid_mask) if v]
        valid_idx  = [int(c.split("_")[1]) + 1 for c in valid_cols]
        m_vals = means[valid_cols].values
        s_vals = stds[valid_cols].values

        if len(m_vals) == 0:
            print(f"  [WARNING] No valid gradient data for 8L {activation}.")
            continue

        ax.plot(valid_idx, m_vals, "o-", color=color, label=label,
                linewidth=2.0, markersize=7)
        ax.fill_between(
            valid_idx,
            np.maximum(m_vals - s_vals, 1e-10),
            m_vals + s_vals,
            color=color, alpha=0.15,
        )

        # Annotate attenuation ratio
        if len(m_vals) >= 2 and m_vals[-1] > 1e-12:
            ratio = m_vals[0] / m_vals[-1]
            ax.annotate(
                f"ratio={ratio:.2f}×",
                xy=(valid_idx[0], m_vals[0]),
                xytext=(valid_idx[0] + 0.5, m_vals[0] * 2.5),
                fontsize=9, color=color,
                arrowprops=dict(arrowstyle="->", color=color, lw=1.0),
            )

    ax.set_xlabel("Layer Index  (1 = input side, 8 = output side)")
    ax.set_ylabel("Gradient L2 Norm  (log scale)")
    ax.set_title("Fig 4A: Gradient Norm vs Layer Depth at Final Epoch\n"
                 "(8L ReLU vs 8L Sigmoid — Primary Gradient Figure)")
    ax.set_yscale("log")
    ax.legend(loc="upper left")
    ax.set_xticks(range(1, 9))
    fig.tight_layout()
    _save(fig, figures_dir, "fig4a_gradient_by_layer.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4B — Gradient Norm vs Epoch (SUPPLEMENTAL)
# ─────────────────────────────────────────────────────────────────────────────

def plot_gradient_norm_over_epochs(
    results_path: str,
    figures_dir: str,
) -> None:
    """
    Figure 4B (SUPPLEMENTAL): Gradient norm trajectory over training.

    Shows the FIRST hidden layer and LAST hidden layer for both 8L ReLU
    and 8L Sigmoid, across epochs (mean ± std over seeds).

    Reveals whether gradient vanishing is present from the start (initialisation
    problem) or develops during training (optimisation problem).

    Addresses RQ2.
    """
    df = _load_and_validate(results_path, "Experiment 2 results")
    if df is None:
        return

    df8 = df[(df["depth"] == 8) & (df["batchnorm"] == 0)].copy()
    grad_cols = _get_grad_cols(df8)
    if len(grad_cols) < 2:
        print("  [WARNING] Insufficient gradient columns for Fig 4B.")
        return

    # First and last hidden layer gradient columns
    first_col = grad_cols[0]
    last_col  = grad_cols[7] if len(grad_cols) > 7 else grad_cols[-1]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    panel_labels = [
        ("Layer 1  (Input Side)",  first_col),
        ("Layer 8  (Output Side)", last_col),
    ]

    for ax, (panel_title, col) in zip(axes, panel_labels):
        for activation, color in [
            ("ReLU",    _ACTIVATION_COLORS["ReLU"]),
            ("Sigmoid", _ACTIVATION_COLORS["Sigmoid"]),
        ]:
            sub = df8[df8["activation"] == activation]
            if sub.empty or col not in sub.columns:
                continue
            grp = sub.groupby("epoch")[col].agg(["mean", "std"]).reset_index()
            grp["std"] = grp["std"].fillna(0)
            # Drop rows where gradient is zero/NaN (first batch warmup)
            grp = grp[grp["mean"] > 0]
            if grp.empty:
                continue

            epochs = grp["epoch"].values
            mean   = grp["mean"].values
            std    = grp["std"].values

            ax.plot(epochs, mean, color=color, label=f"8L {activation}", linewidth=2.0)
            ax.fill_between(
                epochs,
                np.maximum(mean - std, 1e-10),
                mean + std,
                color=color, alpha=0.15,
            )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Gradient L2 Norm  (log scale)")
        ax.set_title(f"Fig 4B: {panel_title}")
        ax.set_yscale("log")
        ax.legend(loc="best")

    fig.suptitle(
        "Fig 4B: Gradient Norm Over Training — 8L ReLU vs 8L Sigmoid  (Supplemental)",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    _save(fig, figures_dir, "fig4b_gradient_over_epochs.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Activation Comparison Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_activation_heatmap(
    summary_path: str,
    figures_dir: str,
) -> None:
    """
    Figure 5: Heatmap with rows = depth, columns = activation.
    Cell value = mean test accuracy (± std annotated in cell).

    Provides immediate visual comparison of all 6 base configurations.
    No-BN configs only (BN comparison is Fig 6).

    Addresses RQ1 + RQ3 simultaneously.
    """
    df = _load_and_validate(summary_path, "summary")
    if df is None:
        return

    df = df[df["batchnorm"] == 0].copy()
    agg = (
        df.groupby(["depth", "activation"])
        .agg(mean_acc=("test_acc", "mean"), std_acc=("test_acc", "std"))
        .reset_index()
    )
    if agg.empty:
        print("  [WARNING] No data for heatmap.")
        return

    pivot_mean = agg.pivot(index="depth", columns="activation", values="mean_acc")
    pivot_std  = agg.pivot(index="depth", columns="activation", values="std_acc")

    # Annotation: "X.XXX\n±Y.YYY"
    annot = pd.DataFrame(
        index=pivot_mean.index, columns=pivot_mean.columns, dtype=object
    )
    for d in pivot_mean.index:
        for a in pivot_mean.columns:
            if pd.isna(pivot_mean.loc[d, a]):
                annot.loc[d, a] = "N/A"
                continue
            m = pivot_mean.loc[d, a]
            s = pivot_std.loc[d, a] if not pd.isna(pivot_std.loc[d, a]) else 0.0
            annot.loc[d, a] = f"{m:.3f}\n±{s:.3f}"

    fig, ax = plt.subplots(figsize=(6, 4))
    vmin = max(0.0, float(pivot_mean.values[~np.isnan(pivot_mean.values)].min()) - 0.05)
    vmax = min(1.0, float(pivot_mean.values[~np.isnan(pivot_mean.values)].max()) + 0.01)

    sns.heatmap(
        pivot_mean,
        annot=annot, fmt="",
        cmap="YlOrRd",
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Mean Test Accuracy"},
        ax=ax, vmin=vmin, vmax=vmax,
        annot_kws={"size": 11},
    )
    ax.set_title("Fig 5: Test Accuracy Heatmap  (Depth × Activation)")
    ax.set_xlabel("Activation Function")
    ax.set_ylabel("Number of Hidden Layers (Depth)")
    fig.tight_layout()
    _save(fig, figures_dir, "fig5_activation_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — BatchNorm Recovery
# ─────────────────────────────────────────────────────────────────────────────

def plot_batchnorm_recovery(
    results_path: str,
    summary_path: str,
    figures_dir: str,
) -> None:
    """
    Figure 6: Two-panel figure for Experiment 3 (8L Sigmoid ±BatchNorm).

    Panel A: Validation accuracy curves (mean ± std)
    Panel B: Gradient attenuation ratio over epochs
             (first_layer_grad / last_layer_grad, log scale)

    A narrowing of the gap in Panel A, combined with a reduced ratio in Panel B,
    would confirm that BatchNorm improves trainability via gradient flow
    rather than purely by increasing capacity.

    Addresses RQ4.
    """
    df = _load_and_validate(results_path, "Experiment 3 results")
    if df is None:
        return

    df8sig = df[(df["depth"] == 8) & (df["activation"] == "Sigmoid")].copy()
    grad_cols = _get_grad_cols(df8sig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Panel A: Validation Accuracy ──────────────────────────────────────
    ax = axes[0]
    for bn_val, color, label in [
        (0, _BN_COLORS[0], "8L Sigmoid  (No BN)"),
        (1, _BN_COLORS[1], "8L Sigmoid  + BatchNorm"),
    ]:
        sub = df8sig[df8sig["batchnorm"] == bn_val]
        if sub.empty:
            continue
        grp = sub.groupby("epoch")["val_acc"].agg(["mean", "std"]).reset_index()
        epochs = grp["epoch"].values
        mean   = grp["mean"].values
        std    = grp["std"].fillna(0).values

        ax.plot(epochs, mean, color=color, label=label, linewidth=2.0)
        ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.15)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Panel A: Validation Accuracy")
    ax.legend(loc="lower right")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))
    ax.set_ylim(0.0, 1.01)

    # ── Panel B: Gradient Attenuation Ratio ───────────────────────────────
    ax = axes[1]
    if len(grad_cols) >= 2:
        first_col = grad_cols[0]
        last_col  = grad_cols[7] if len(grad_cols) > 7 else grad_cols[-1]

        for bn_val, color, label in [
            (0, _BN_COLORS[0], "8L Sigmoid  (No BN)"),
            (1, _BN_COLORS[1], "8L Sigmoid  + BatchNorm"),
        ]:
            sub = df8sig[df8sig["batchnorm"] == bn_val].copy()
            if sub.empty:
                continue

            # Compute per-row gradient attenuation ratio safely
            last_vals = sub[last_col].replace(0.0, np.nan)
            sub = sub.assign(grad_ratio=sub[first_col] / last_vals)

            grp = sub.groupby("epoch")["grad_ratio"].agg(["mean", "std"]).reset_index()
            grp = grp[grp["mean"].notna() & (grp["mean"] > 0)]
            if grp.empty:
                continue

            epochs = grp["epoch"].values
            mean   = grp["mean"].values
            std    = grp["std"].fillna(0).values

            ax.plot(epochs, mean, color=color, label=label, linewidth=2.0)
            ax.fill_between(
                epochs,
                np.maximum(mean - std, 1e-3),
                mean + std,
                color=color, alpha=0.15,
            )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Gradient Attenuation Ratio\n(Layer 1 grad / Layer 8 grad)  [log]")
        ax.set_title("Panel B: Gradient Attenuation Ratio\n(higher = more vanishing)")
        ax.set_yscale("log")
        ax.legend(loc="best")
    else:
        axes[1].text(0.5, 0.5, "Gradient data not available.\nRe-run experiments.",
                     ha="center", va="center", transform=axes[1].transAxes, fontsize=11)

    fig.suptitle(
        "Fig 6: BatchNorm Recovery — 8L Sigmoid  (Experiment 3)",
        fontsize=14, y=1.03,
    )
    fig.tight_layout()
    _save(fig, figures_dir, "fig6_batchnorm_recovery.png")


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: Generate All Figures
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_figures(
    results_path: str,
    summary_path: str,
    figures_dir: str,
) -> None:
    """
    Generate all 7 required figures in sequence.

    Each function is defensive: it prints a warning and returns gracefully
    if the required CSV data is not yet available.

    Args:
        results_path: Path to epoch-level results CSV.
        summary_path: Path to per-run summary CSV.
        figures_dir:  Directory to save all PNG files.
    """
    print("\n" + "═" * 60)
    print("GENERATING ALL FIGURES")
    print("═" * 60)

    print("\n[Fig 1] Validation Accuracy Curves (Depth Comparison, ReLU)")
    plot_val_accuracy_curves(results_path, figures_dir)

    print("\n[Fig 2] Validation Loss Curves (Depth Comparison, ReLU)")
    plot_val_loss_curves(results_path, figures_dir)

    print("\n[Fig 3] Final Test Accuracy Bar Chart")
    plot_test_accuracy_bar(summary_path, figures_dir)

    print("\n[Fig 4A] Gradient Norm vs Layer Depth (8L, final epoch)")
    plot_gradient_norm_by_layer(results_path, figures_dir)

    print("\n[Fig 4B] Gradient Norm vs Epoch (supplemental)")
    plot_gradient_norm_over_epochs(results_path, figures_dir)

    print("\n[Fig 5] Activation Comparison Heatmap")
    plot_activation_heatmap(summary_path, figures_dir)

    print("\n[Fig 6] BatchNorm Recovery")
    plot_batchnorm_recovery(results_path, summary_path, figures_dir)

    print("\n" + "═" * 60)
    print(f"All figures saved to: {figures_dir}")
    print("═" * 60)
