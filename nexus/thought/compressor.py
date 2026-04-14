"""Thought sequence compression to reduce context length."""

from __future__ import annotations

import torch
import torch.nn as nn


class ThoughtCompressor(nn.Module):
    """
    Compresses a sequence of thought tokens into a shorter representation.
    Uses learned pooling with attention-based aggregation.
    """

    def __init__(
        self,
        hidden_dim: int,
        max_thought_tokens: int = 1024,
        compressed_len: int = 16,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.compressed_len = compressed_len

        # Learnable query vectors for compression
        self.compress_queries = nn.Parameter(
            torch.randn(compressed_len, hidden_dim) * 0.02,
        )

        # Cross-attention for compression
        self.compress_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=4, dropout=0.1, batch_first=True,
        )
        self.compress_norm = nn.LayerNorm(hidden_dim)
        self.compress_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)

    def forward(self, thoughts: torch.Tensor) -> torch.Tensor:
        """
        Compress thought tokens into a shorter representation.

        Args:
            thoughts: (batch, num_thoughts, hidden_dim)

        Returns:
            (batch, compressed_len, hidden_dim) compressed thoughts.
        """
        B = thoughts.shape[0]

        # Expand queries for batch
        queries = self.compress_queries.unsqueeze(0).expand(B, -1, -1)

        # Cross-attend from queries to thoughts
        compressed, _ = self.compress_attn(queries, thoughts, thoughts)
        compressed = self.compress_norm(queries + compressed)
        compressed = self.ffn_norm(compressed + self.compress_ffn(compressed))

        return compressed
