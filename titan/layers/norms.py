"""Normalization layers."""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Cheaper than LayerNorm (no mean subtraction), same quality for transformers.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(norm + self.eps) * self.weight
