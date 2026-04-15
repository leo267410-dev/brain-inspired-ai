"""Recurrent Mixer — lightweight O(1) memory token mixing.

Replaces full attention in non-anchor layers. Combines:
  1. Causal depthwise convolution for local context
  2. Gated recurrent scan for long-range state propagation

This gives every layer some sequential modeling capability without the O(S^2)
cost of full attention. Attention anchors (every 3rd layer) handle global routing.

Design inspired by HelixForge's RecurrentHelixMixer with improvements:
  - Separate decay and inject signals for cleaner state dynamics
  - Gate controls interpolation between direct value and recurrent state
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from titan.config import TitanConfig


class RecurrentMixer(nn.Module):
    """Lightweight recurrent token mixer with causal convolution."""

    def __init__(self, cfg: TitanConfig):
        super().__init__()
        d = cfg.d_model
        # Project to 4 streams: value, gate, decay, inject
        self.proj = nn.Linear(d, d * 4, bias=cfg.use_bias)
        # Causal depthwise conv for local mixing
        self.dw_conv = nn.Conv1d(
            d,
            d,
            kernel_size=cfg.recurrent_kernel_size,
            padding=cfg.recurrent_kernel_size - 1,
            groups=d,
            bias=False,
        )
        self.out = nn.Linear(d, d, bias=cfg.use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape

        # Local mixing via causal depthwise conv
        mixed = self.dw_conv(x.transpose(1, 2))[:, :, :t].transpose(1, 2)

        # Project into 4 streams
        v, gate, decay, inject = self.proj(x + mixed).chunk(4, dim=-1)
        gate = torch.sigmoid(gate)
        decay = torch.sigmoid(decay)
        inject = torch.tanh(inject)

        # Recurrent scan: state accumulates long-range context
        state = torch.zeros(b, d, device=x.device, dtype=x.dtype)
        outs: List[torch.Tensor] = []
        for i in range(t):
            state = decay[:, i] * state + (1.0 - decay[:, i]) * inject[:, i]
            outs.append(gate[:, i] * v[:, i] + (1.0 - gate[:, i]) * state)

        y = torch.stack(outs, dim=1)
        return self.out(y)
