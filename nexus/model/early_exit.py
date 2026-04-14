"""Dynamic depth (early exit) and dynamic width gating."""

from __future__ import annotations

import torch
import torch.nn as nn


class ExitClassifier(nn.Module):
    """Per-token exit confidence classifier for early exit."""

    def __init__(self, hidden_dim: int, intermediate_dim: int = 128):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim)

        Returns:
            (batch, seq_len) per-token exit confidence scores.
        """
        return self.classifier(x).squeeze(-1)


class DynamicWidthGate(nn.Module):
    """Decides how much of the hidden dimension each token uses."""

    def __init__(self, hidden_dim: int, threshold: float = 0.3):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim)

        Returns:
            (batch, seq_len, hidden_dim) with some dimensions masked off.
        """
        width_scores = torch.sigmoid(self.gate(x))
        mask = (width_scores > self.threshold).float()
        # Straight-through estimator
        mask_ste = width_scores + (mask - width_scores).detach()
        return x * mask_ste

    def get_active_fraction(self, x: torch.Tensor) -> float:
        """Return the average fraction of active dimensions."""
        with torch.no_grad():
            width_scores = torch.sigmoid(self.gate(x))
            mask = (width_scores > self.threshold).float()
            return mask.mean().item()


class EarlyExitManager(nn.Module):
    """
    Manages early exit across layers.
    During inference, tokens that are confident enough can skip remaining layers.
    """

    def __init__(self, num_layers: int, hidden_dim: int, threshold: float = 0.8):
        super().__init__()
        self.threshold = threshold
        self.exit_classifiers = nn.ModuleList([
            ExitClassifier(hidden_dim) for _ in range(num_layers)
        ])

    def should_exit(self, hidden_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Check if tokens should exit at this layer.

        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            layer_idx: current layer index.

        Returns:
            (batch, seq_len) boolean mask of tokens that should exit.
        """
        confidence = self.exit_classifiers[layer_idx](hidden_states)
        return confidence > self.threshold

    def get_exit_loss(
        self, hidden_states: torch.Tensor, layer_idx: int, targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute auxiliary exit loss for training.

        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            layer_idx: current layer index
            targets: (batch, seq_len) target exit decisions (1=should exit)

        Returns:
            Scalar loss.
        """
        confidence = self.exit_classifiers[layer_idx](hidden_states)
        return nn.functional.binary_cross_entropy(confidence, targets.float())
