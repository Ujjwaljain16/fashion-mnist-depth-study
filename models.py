"""
models.py — Configurable MLP for the Fashion-MNIST depth study.

Provides a single class, ConfigurableMLP, that supports:
  - Arbitrary depth (number of hidden layers)
  - Configurable width (units per hidden layer)
  - ReLU or Sigmoid activation
  - Optional BatchNorm1d (Linear → BN → Activation, standard placement)
  - Explicit weight initialisation per activation (Kaiming / Xavier)
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn


ActivationType = Literal["ReLU", "Sigmoid"]


class ConfigurableMLP(nn.Module):
    """
    Multi-Layer Perceptron with configurable depth, width, activation, and BatchNorm.

    Architecture:
        nn.Flatten()
        → [ Linear(in, W) → [BatchNorm1d(W)] → Activation ]  ×  depth
        → Linear(W, num_classes)

    No activation or BatchNorm is applied on the output layer; its outputs
    are raw logits fed directly into CrossEntropyLoss.

    BatchNorm placement:
        Linear → BatchNorm1d → Activation.
        This is the original Ioffe & Szegedy (2015) placement and the de-facto
        standard for MLP experiments. It normalises pre-activations, keeping
        inputs to Sigmoid in its linear (non-saturating) region.

    Weight initialisation:
        ReLU  → Kaiming Uniform (fan_in):  compensates for ReLU's half-zeroing.
                 Derived from He et al. (2015), "Delving Deep into Rectifiers."
        Sigmoid → Xavier Uniform:  designed for symmetric saturating activations.
                 Derived from Glorot & Bengio (2010), "Understanding the Difficulty
                 of Training Deep Feedforward Neural Networks."
        Output layer → Xavier Uniform (activation-agnostic; feeds CrossEntropy).

    Args:
        input_dim:     Number of input features. Default 784 (28×28 flat).
        num_classes:   Number of output classes. Default 10.
        depth:         Number of hidden layers.
        width:         Units per hidden layer.
        activation:    "ReLU" or "Sigmoid".
        use_batchnorm: Insert BatchNorm1d after each hidden Linear if True.
    """

    def __init__(
        self,
        input_dim: int = 784,
        num_classes: int = 10,
        depth: int = 2,
        width: int = 413,
        activation: ActivationType = "ReLU",
        use_batchnorm: bool = False,
    ) -> None:
        super().__init__()

        if activation not in ("ReLU", "Sigmoid"):
            raise ValueError(f"Unsupported activation '{activation}'. Use 'ReLU' or 'Sigmoid'.")
        if depth < 1:
            raise ValueError(f"depth must be ≥ 1, got {depth}.")

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.depth = depth
        self.width = width
        self.activation_name: ActivationType = activation
        self.use_batchnorm = use_batchnorm

        self.network: nn.Sequential = self._build_layers()
        self._init_weights()

    # ── Layer Construction ────────────────────────────────────────────────────

    def _build_layers(self) -> nn.Sequential:
        """
        Build the full layer stack as nn.Sequential.

        Layer order within each hidden block:
            Linear → [BatchNorm1d] → Activation

        The output layer is a bare Linear(width, num_classes) with no
        BatchNorm or activation, producing raw logits.
        """
        act_cls = nn.ReLU if self.activation_name == "ReLU" else nn.Sigmoid

        layers: list[nn.Module] = [nn.Flatten()]

        # Hidden layer 1: input_dim → width
        layers.append(nn.Linear(self.input_dim, self.width))
        if self.use_batchnorm:
            layers.append(nn.BatchNorm1d(self.width))
        layers.append(act_cls())

        # Hidden layers 2 … depth: width → width
        for _ in range(self.depth - 1):
            layers.append(nn.Linear(self.width, self.width))
            if self.use_batchnorm:
                layers.append(nn.BatchNorm1d(self.width))
            layers.append(act_cls())

        # Output layer
        layers.append(nn.Linear(self.width, self.num_classes))

        return nn.Sequential(*layers)

    # ── Weight Initialisation ─────────────────────────────────────────────────

    def _init_weights(self) -> None:
        """
        Apply activation-appropriate weight initialisation to all Linear layers.

        ReLU → Kaiming Uniform (fan_in, nonlinearity='relu'):
            Ensures the variance of activations is preserved across layers by
            accounting for ReLU zeroing ~50% of neurons. PyTorch's nn.Linear
            default is Kaiming Uniform, so this re-initialisation is a no-op
            for ReLU models but makes the choice explicit and auditable.

        Sigmoid → Xavier Uniform:
            Designed for symmetric activations. Scales weights by
            √(6 / (fan_in + fan_out)) to preserve variance at
            initialisation (before saturation sets in). Sigmoid models
            still exhibit vanishing gradients at depth — this demonstrates
            that initialisation alone cannot fully fix the problem.

        Output layer → Xavier Uniform regardless of activation,
            as it feeds into CrossEntropyLoss (no activation).

        Biases → zeros throughout.
        BatchNorm → γ=1, β=0 (identity at init, per original BN paper).
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.activation_name == "ReLU":
                    nn.init.kaiming_uniform_(
                        module.weight, mode="fan_in", nonlinearity="relu"
                    )
                else:
                    nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)    # γ = 1
                nn.init.zeros_(module.bias)     # β = 0

    # ── Forward Pass ──────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 1, 28, 28) or (B, 784).
               nn.Flatten handles both.

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        return self.network(x)

    # ── Introspection Helpers ─────────────────────────────────────────────────

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_hidden_linear_layers(self) -> list[nn.Linear]:
        """
        Return a list of the hidden (non-output) nn.Linear modules in
        forward order (index 0 = closest to input).

        Used by the gradient capture routine to log per-layer gradient norms.
        The output Linear is excluded because it is not a hidden layer and
        its gradient is dominated by the loss directly.
        """
        linears = [m for m in self.network if isinstance(m, nn.Linear)]
        # linears[-1] is the output layer; all others are hidden
        return linears[:-1]

    def __repr__(self) -> str:
        return (
            f"ConfigurableMLP("
            f"depth={self.depth}, "
            f"width={self.width}, "
            f"activation={self.activation_name}, "
            f"batchnorm={self.use_batchnorm}, "
            f"params={self.count_parameters():,})"
        )
