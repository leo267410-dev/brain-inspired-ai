"""Hebbian lateral connections with structural plasticity."""

from __future__ import annotations

import torch
import torch.nn as nn


class HebbianLateralConnections(nn.Module):
    """
    Sparse lateral connections between neurons in the same layer.
    Updated via Hebbian learning (no backprop needed).
    """

    def __init__(
        self,
        hidden_dim: int,
        connectivity: float = 0.1,
        alpha: float = 0.01,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.alpha = nn.Parameter(torch.tensor(alpha))

        # Sparse connection mask (e.g. 10% connectivity)
        num_connections = int(hidden_dim * hidden_dim * connectivity)
        indices = torch.randint(0, hidden_dim, (2, num_connections))
        values = torch.randn(num_connections) * 0.001

        self.register_buffer("connection_indices", indices)
        self.connection_values = nn.Parameter(values)
        # Dense mask for reference
        mask = torch.sparse_coo_tensor(
            indices, torch.ones(num_connections), (hidden_dim, hidden_dim),
        ).to_dense().bool()
        self.register_buffer("connection_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply lateral connections.

        Args:
            x: (batch, seq_len, hidden_dim)

        Returns:
            (batch, seq_len, hidden_dim) with lateral signal added.
        """
        W = torch.sparse_coo_tensor(
            self.connection_indices,
            self.connection_values,
            (self.hidden_dim, self.hidden_dim),
        ).to_dense().to(x.device)

        lateral_signal = torch.matmul(x, W.T)
        return x + self.alpha * lateral_signal

    @torch.no_grad()
    def hebbian_update(
        self, activations: torch.Tensor, lr: float = 0.01, decay: float = 0.001,
    ) -> None:
        """
        Update lateral connections using Hebbian rule.

        Args:
            activations: (batch, seq_len, hidden_dim) — detached activations.
            lr: Hebbian learning rate.
            decay: weight decay factor.
        """
        act_flat = activations.reshape(-1, self.hidden_dim)

        i_indices = self.connection_indices[0]
        j_indices = self.connection_indices[1]

        correlation = (act_flat[:, i_indices] * act_flat[:, j_indices]).mean(dim=0)
        self.connection_values.data += lr * correlation - decay * self.connection_values.data

    @torch.no_grad()
    def structural_plasticity(
        self, prune_fraction: float = 0.05, grow_fraction: float = 0.05,
    ) -> None:
        """Prune weak connections and grow new random ones."""
        values = self.connection_values.data.abs()
        num_prune = int(len(values) * prune_fraction)

        if num_prune == 0:
            return

        # Prune weakest
        _, prune_indices = torch.topk(values, num_prune, largest=False)

        # Replace with new random connections
        new_i = torch.randint(0, self.hidden_dim, (num_prune,), device=values.device)
        new_j = torch.randint(0, self.hidden_dim, (num_prune,), device=values.device)

        self.connection_indices[0, prune_indices] = new_i
        self.connection_indices[1, prune_indices] = new_j
        self.connection_values.data[prune_indices] = torch.randn(
            num_prune, device=values.device,
        ) * 0.001


class CrossLayerLateral(nn.Module):
    """
    Sparse lateral connections between adjacent layers.
    Takes output from layer N-1 and current layer N.
    """

    def __init__(
        self,
        hidden_dim: int,
        connectivity: float = 0.1,
        alpha: float = 0.01,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.alpha = nn.Parameter(torch.tensor(alpha))

        num_connections = int(hidden_dim * hidden_dim * connectivity)
        indices = torch.randint(0, hidden_dim, (2, num_connections))
        values = torch.randn(num_connections) * 0.001

        self.register_buffer("connection_indices", indices)
        self.connection_values = nn.Parameter(values)

    def forward(self, x_current: torch.Tensor, x_prev: torch.Tensor) -> torch.Tensor:
        """
        Apply cross-layer lateral connections.

        Args:
            x_current: (batch, seq_len, hidden_dim) current layer output.
            x_prev: (batch, seq_len, hidden_dim) previous layer output.

        Returns:
            (batch, seq_len, hidden_dim) with cross-layer signal added.
        """
        W = torch.sparse_coo_tensor(
            self.connection_indices,
            self.connection_values,
            (self.hidden_dim, self.hidden_dim),
        ).to_dense().to(x_current.device)

        lateral_signal = torch.matmul(x_prev, W.T)
        return x_current + self.alpha * lateral_signal

    @torch.no_grad()
    def hebbian_update(
        self,
        act_current: torch.Tensor,
        act_prev: torch.Tensor,
        lr: float = 0.01,
        decay: float = 0.001,
    ) -> None:
        """Hebbian update based on co-activation across layers."""
        curr_flat = act_current.reshape(-1, self.hidden_dim)
        prev_flat = act_prev.reshape(-1, self.hidden_dim)

        i_indices = self.connection_indices[0]
        j_indices = self.connection_indices[1]

        correlation = (curr_flat[:, i_indices] * prev_flat[:, j_indices]).mean(dim=0)
        self.connection_values.data += lr * correlation - decay * self.connection_values.data
