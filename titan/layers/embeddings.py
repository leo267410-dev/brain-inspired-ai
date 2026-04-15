"""Positional embeddings — RoPE (Rotary Position Embedding).

RoPE encodes position through rotation in 2D subspaces of the embedding.
Proven superior to learned absolute position embeddings for length generalization.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """Precomputes cos/sin tables for Rotary Position Embedding."""

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, seq_len: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        # (1, 1, seq_len, head_dim) for broadcasting over batch and heads
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dimension."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply rotary position embedding to tensor x."""
    return (x * cos) + (rotate_half(x) * sin)
