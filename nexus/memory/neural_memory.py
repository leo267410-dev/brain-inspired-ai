"""External neural memory with differentiable read/write."""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from nexus.config import NexusOmegaConfig


class NeuralMemory(nn.Module):
    """
    External neural memory bank with differentiable read/write.
    Stores key-value pairs for long-term knowledge retrieval.
    """

    def __init__(self, config: NexusOmegaConfig):
        super().__init__()
        self.num_entries = config.memory_num_entries
        self.key_dim = config.memory_key_dim
        self.hidden_dim = config.hidden_dim
        self.top_k = config.memory_top_k

        # Memory banks (not trained via backprop — updated via write operations)
        self.register_buffer(
            "keys", torch.randn(config.memory_num_entries, config.memory_key_dim) * 0.01,
        )
        self.register_buffer(
            "values", torch.zeros(config.memory_num_entries, config.hidden_dim),
        )
        self.register_buffer(
            "usage_count", torch.zeros(config.memory_num_entries),
        )

        # Projection layers
        self.query_proj = nn.Linear(config.hidden_dim, config.memory_key_dim)
        self.key_proj = nn.Linear(config.hidden_dim, config.memory_key_dim)
        self.output_proj = nn.Linear(config.hidden_dim, config.hidden_dim)

        # Gate for blending memory with hidden state
        self.memory_gate = nn.Linear(config.hidden_dim * 2, 1)

    def read(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Read from memory using content-based addressing.

        Args:
            query: (batch, seq_len, hidden_dim)

        Returns:
            Tuple of (memory_output, attention_weights):
                memory_output: (batch, seq_len, hidden_dim)
                attention_weights: (batch, seq_len, top_k)
        """
        B, L, D = query.shape

        # Project query to key space
        q = self.query_proj(query)  # (B, L, key_dim)

        # Compute similarity with all keys
        q_flat = q.reshape(B * L, self.key_dim)  # (B*L, key_dim)
        sim = torch.matmul(q_flat, self.keys.T)  # (B*L, num_entries)

        # Top-k retrieval
        top_k_sim, top_k_idx = torch.topk(sim, self.top_k, dim=-1)  # (B*L, top_k)
        top_k_weights = F.softmax(top_k_sim, dim=-1)  # (B*L, top_k)

        # Gather top-k values
        top_k_values = self.values[top_k_idx]  # (B*L, top_k, hidden_dim)

        # Weighted sum of values
        memory_out = (top_k_weights.unsqueeze(-1) * top_k_values).sum(dim=1)  # (B*L, D)
        memory_out = memory_out.view(B, L, D)
        memory_out = self.output_proj(memory_out)

        # Update usage counts
        if self.training:
            with torch.no_grad():
                unique_idx = top_k_idx.unique()
                self.usage_count[unique_idx] += 1

        return memory_out, top_k_weights.view(B, L, self.top_k)

    @torch.no_grad()
    def write(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        """
        Write to memory at specific indices.

        Args:
            keys: (N, key_dim) keys to write.
            values: (N, hidden_dim) values to write.
            indices: (N,) memory indices to write to.
        """
        self.keys[indices] = keys
        self.values[indices] = values

    @torch.no_grad()
    def write_lru(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        """
        Write to memory, replacing least-recently-used entries.

        Args:
            keys: (N, key_dim) keys to write.
            values: (N, hidden_dim) values to write.
        """
        N = keys.shape[0]
        _, lru_indices = torch.topk(self.usage_count, N, largest=False)
        self.keys[lru_indices] = keys
        self.values[lru_indices] = values
        self.usage_count[lru_indices] = 0

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Read from memory and blend with hidden state.

        Args:
            hidden: (batch, seq_len, hidden_dim)

        Returns:
            (batch, seq_len, hidden_dim) memory-augmented hidden states.
        """
        memory_out, _ = self.read(hidden)
        gate = torch.sigmoid(
            self.memory_gate(torch.cat([hidden, memory_out], dim=-1)),
        )
        return hidden + gate * memory_out
