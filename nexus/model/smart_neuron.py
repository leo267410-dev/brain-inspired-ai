"""Adaptive neuron gating and evolutionary neuron management."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class AdaptiveNeuronGate(nn.Module):
    """
    Per-neuron gate that learns which neurons to activate per input.
    Uses straight-through estimator for gradient flow through hard threshold.
    """

    def __init__(self, hidden_dim: int, threshold: float = 0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gate_proj = nn.Linear(hidden_dim, hidden_dim)
        self.threshold = nn.Parameter(torch.tensor(threshold))

        # Statistics tracking
        self.register_buffer("activation_count", torch.zeros(hidden_dim))
        self.register_buffer("total_count", torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim)

        Returns:
            Gated (batch, seq_len, hidden_dim) with inactive neurons zeroed.
        """
        gate_score = torch.sigmoid(self.gate_proj(x))  # (B, L, D)
        mask = (gate_score > self.threshold).float()
        # Straight-through estimator for gradient flow
        mask_ste = gate_score + (mask - gate_score).detach()

        # Track statistics
        if self.training:
            self.activation_count += mask.sum(dim=(0, 1)).detach()
            self.total_count += mask.shape[0] * mask.shape[1]

        return x * mask_ste

    def get_utilization(self) -> torch.Tensor:
        """Return per-neuron utilization ratio."""
        if self.total_count.item() == 0:
            return torch.ones_like(self.activation_count)
        return self.activation_count / self.total_count


class SmartNeuronLayer(nn.Module):
    """A layer wrapper that applies adaptive neuron gating after a sublayer."""

    def __init__(self, hidden_dim: int, threshold: float = 0.3):
        super().__init__()
        self.gate = AdaptiveNeuronGate(hidden_dim, threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply neuron gating."""
        return self.gate(x)


class EvolutionaryNeuronManager:
    """
    Manages evolutionary pruning of neurons across the model.
    Periodically kills the least-utilized neurons and reinitializes them.
    """

    def __init__(self, prune_percentile: float = 5.0):
        self.prune_percentile = prune_percentile
        self.step_count = 0

    def evolutionary_step(
        self, smart_neuron_layers: List[SmartNeuronLayer], prune_percentile: float = 5.0,
    ) -> int:
        """
        Kill bottom N% utilized neurons and reinitialize them.

        Args:
            smart_neuron_layers: list of SmartNeuronLayer modules.
            prune_percentile: bottom percentile to prune.

        Returns:
            Total number of neurons pruned across all layers.
        """
        total_pruned = 0
        for layer in smart_neuron_layers:
            utilization = layer.gate.get_utilization()
            if utilization.sum().item() == 0:
                continue

            threshold = torch.quantile(
                utilization.float(), prune_percentile / 100.0,
            )
            dead_mask = utilization < threshold

            num_dead = dead_mask.sum().item()
            if num_dead == 0:
                continue

            # Reinitialize dead neurons
            with torch.no_grad():
                layer.gate.gate_proj.weight.data[dead_mask] = (
                    torch.randn_like(layer.gate.gate_proj.weight.data[dead_mask]) * 0.02
                )
                layer.gate.gate_proj.bias.data[dead_mask] = 0.0

            # Reset stats for reinitialized neurons
            layer.gate.activation_count[dead_mask] = 0
            total_pruned += num_dead

        self.step_count += 1
        return total_pruned
