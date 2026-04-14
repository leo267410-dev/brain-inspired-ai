"""Mamba-style State Space Model blocks and SSM-Attention hybrid."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from nexus.config import NexusOmegaConfig


class MambaSSMBlock(nn.Module):
    """
    Simplified Mamba-style selective state space model.
    O(n) complexity for sequence processing.
    """

    def __init__(
        self,
        hidden_dim: int,
        state_dim: int = 16,
        conv_dim: int = 4,
        expand_factor: int = 2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        inner_dim = expand_factor * hidden_dim
        self.inner_dim = inner_dim

        # Input projection: hidden_dim -> inner_dim * 2 (for x and z gate)
        self.in_proj = nn.Linear(hidden_dim, inner_dim * 2)

        # 1D convolution for local context
        self.conv1d = nn.Conv1d(
            inner_dim, inner_dim,
            kernel_size=conv_dim, padding=conv_dim - 1, groups=inner_dim,
        )

        # SSM parameters (selective — input-dependent)
        self.dt_proj = nn.Linear(inner_dim, inner_dim)
        self.A_log = nn.Parameter(
            torch.log(torch.randn(inner_dim, state_dim).abs() + 1e-6)
        )
        self.B_proj = nn.Linear(inner_dim, state_dim)
        self.C_proj = nn.Linear(inner_dim, state_dim)
        self.D = nn.Parameter(torch.ones(inner_dim))

        # Output projection
        self.out_proj = nn.Linear(inner_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim)

        Returns:
            (batch, seq_len, hidden_dim)
        """
        B, L, _ = x.shape

        # Project and split into x_part and gate z
        xz = self.in_proj(x)
        x_part, z = xz.chunk(2, dim=-1)

        # Conv for local context
        x_conv = self.conv1d(x_part.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_conv = F.silu(x_conv)

        # Selective SSM parameters
        A = -torch.exp(self.A_log)  # (inner_dim, state_dim)
        dt = F.softplus(self.dt_proj(x_conv))  # (B, L, inner_dim)
        B_input = self.B_proj(x_conv)  # (B, L, state_dim)
        C_input = self.C_proj(x_conv)  # (B, L, state_dim)

        # Selective scan
        y = self.selective_scan(x_conv, dt, A, B_input, C_input, self.D)

        # Gate and project
        y = y * F.silu(z)
        return self.out_proj(y)

    def selective_scan(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sequential selective scan.

        Args:
            x: (batch, seq_len, inner_dim)
            dt: (batch, seq_len, inner_dim) step sizes
            A: (inner_dim, state_dim) state matrix
            B: (batch, seq_len, state_dim) input matrix
            C: (batch, seq_len, state_dim) output matrix
            D: (inner_dim,) skip connection

        Returns:
            (batch, seq_len, inner_dim)
        """
        batch, seq_len, dim = x.shape
        state_dim = A.shape[1]

        h = torch.zeros(batch, dim, state_dim, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(seq_len):
            # Discretize: dA = exp(dt * A), dB = dt * B
            dA = torch.exp(dt[:, t, :, None] * A[None, :, :])  # (B, D, N)
            dB = dt[:, t, :, None] * B[:, t, None, :]  # (B, D, N)

            # State update: h = dA * h + dB * x
            h = dA * h + dB * x[:, t, :, None]

            # Output: y = C * h + D * x
            y = (h * C[:, t, None, :]).sum(dim=-1) + D * x[:, t]
            outputs.append(y)

        return torch.stack(outputs, dim=1)


class HybridSSMAttention(nn.Module):
    """Combines Mamba SSM for local processing with sparse attention for global."""

    def __init__(self, config: NexusOmegaConfig):
        super().__init__()
        from nexus.model.attention import HierarchicalSparseAttention

        self.ssm = MambaSSMBlock(
            config.hidden_dim, config.ssm_state_dim,
            config.ssm_conv_dim, config.ssm_expand_factor,
        )
        self.attention = HierarchicalSparseAttention(config)
        self.gate = nn.Linear(config.hidden_dim, 1)
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim)
            attention_mask: optional attention mask
            kv_cache: optional KV cache tuple

        Returns:
            (batch, seq_len, hidden_dim)
        """
        # SSM path
        ssm_out = self.ssm(self.norm1(x))
        # Attention path
        attn_out = self.attention(self.norm2(x), attention_mask, kv_cache)
        # Blend
        gate = torch.sigmoid(self.gate(x))
        return x + gate * ssm_out + (1 - gate) * attn_out
