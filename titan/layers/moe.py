"""Mixture of Experts with Oscillatory Expert Banks.

Two key innovations over standard MoE:

1. **Shared Expert Banks**: Expert parameters are shared across all layers.
   Each layer has its own lightweight router but reuses the same bank weights.
   This concentrates parameters into specialized experts without blowing up
   per-layer parameter count. (Inspired by DeepSeek-V3 shared experts.)

2. **Oscillatory Activation**: Instead of SwiGLU (silu(gate) * up), expert banks
   use SRN activation: tanh(mag) * cos(softplus(freq) + phase).
   This gives each expert neuron richer nonlinear compute per parameter — the
   core insight from neuron research that simple activations are informationally
   shallow and require massive scale to compensate.

   When use_oscillatory=False, falls back to standard SwiGLU for comparison.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from titan.layers.norms import RMSNorm

if TYPE_CHECKING:
    from titan.config import TitanConfig


class OscillatoryExpertBank(nn.Module):
    """Shared expert bank pool with oscillatory (SRN) or SwiGLU activation.

    Parameters are shared across all layers. Each bank is an independent expert
    with its own up/gate/down projections.
    """

    def __init__(self, cfg: TitanConfig):
        super().__init__()
        self.use_oscillatory = cfg.use_oscillatory
        d = cfg.d_model
        h = cfg.bank_hidden
        n = cfg.routing_banks

        if self.use_oscillatory:
            # SRN-style: magnitude + frequency projections
            self.bank_mag = nn.Parameter(torch.empty(n, d, h))
            self.bank_freq = nn.Parameter(torch.empty(n, d, h))
            self.bank_phase = nn.Parameter(torch.zeros(n, h))
            self.bank_down = nn.Parameter(torch.empty(n, h, d))
            self._init_oscillatory()
        else:
            # Standard SwiGLU: up + gate + down
            self.bank_up = nn.Parameter(torch.empty(n, d, h))
            self.bank_gate = nn.Parameter(torch.empty(n, d, h))
            self.bank_down = nn.Parameter(torch.empty(n, h, d))
            self._init_swiglu()

    def _init_oscillatory(self) -> None:
        nn.init.kaiming_normal_(self.bank_mag, mode="fan_in", nonlinearity="tanh")
        nn.init.uniform_(self.bank_freq, 0.5, 2.0)
        nn.init.uniform_(self.bank_phase, -0.3, 0.3)
        nn.init.xavier_uniform_(self.bank_down)

    def _init_swiglu(self) -> None:
        nn.init.xavier_uniform_(self.bank_up)
        nn.init.xavier_uniform_(self.bank_gate)
        nn.init.xavier_uniform_(self.bank_down)


class SmartMoEFFN(nn.Module):
    """Per-layer MoE FFN using shared expert banks.

    Each layer has its own router but reuses the shared OscillatoryExpertBank.
    Top-k routing selects which expert banks to activate per token.
    """

    def __init__(self, cfg: TitanConfig, shared_banks: OscillatoryExpertBank):
        super().__init__()
        self.cfg = cfg
        self.shared_banks = shared_banks
        self.router = nn.Linear(cfg.d_model, cfg.routing_banks, bias=False)
        self.router_bias = nn.Parameter(torch.zeros(cfg.routing_banks))
        self.out_norm = RMSNorm(cfg.d_model)
        # Cached router probabilities from last forward pass (for entropy regularization)
        self._cached_router_probs: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        banks = self.shared_banks

        # Router scores
        scores = self.router(x) + self.router_bias
        # Cache full router probs for entropy regularization
        self._cached_router_probs = F.softmax(scores, dim=-1)
        topv, topi = torch.topk(scores, k=self.cfg.routing_topk, dim=-1)
        weights = F.softmax(topv, dim=-1)

        out = torch.zeros_like(x)
        flat_x = x.reshape(b * t, d)
        flat_topi = topi.reshape(b * t, self.cfg.routing_topk)
        flat_weights = weights.reshape(b * t, self.cfg.routing_topk)

        for k_idx in range(self.cfg.routing_topk):
            bank_ids = flat_topi[:, k_idx]
            bank_weight = flat_weights[:, k_idx].unsqueeze(-1)

            if banks.use_oscillatory:
                # SRN activation: tanh(mag) * cos(softplus(freq) + phase)
                mag_w = banks.bank_mag[bank_ids]   # (bt, d, h)
                freq_w = banks.bank_freq[bank_ids]  # (bt, d, h)
                phase = banks.bank_phase[bank_ids]   # (bt, h)
                down = banks.bank_down[bank_ids]     # (bt, h, d)

                mag = torch.bmm(flat_x.unsqueeze(1), mag_w).squeeze(1)   # (bt, h)
                freq = torch.bmm(flat_x.unsqueeze(1), freq_w).squeeze(1)  # (bt, h)
                hidden = torch.tanh(mag) * torch.cos(F.softplus(freq) + 0.1 + phase)
            else:
                # SwiGLU activation: silu(gate) * up
                up = banks.bank_up[bank_ids]
                gate = banks.bank_gate[bank_ids]
                down = banks.bank_down[bank_ids]

                hidden = torch.bmm(flat_x.unsqueeze(1), up).squeeze(1)
                hidden_gate = torch.bmm(flat_x.unsqueeze(1), gate).squeeze(1)
                hidden = F.silu(hidden_gate) * hidden

            bank_out = torch.bmm(hidden.unsqueeze(1), down).squeeze(1)
            out = out + (bank_out.view(b, t, d) * bank_weight.view(b, t, 1))

        return self.out_norm(out)

    def router_entropy_loss(self) -> torch.Tensor:
        """Regularization loss from cached router probs (call after forward)."""
        if self._cached_router_probs is None:
            return torch.tensor(0.0)
        probs = self._cached_router_probs  # (b, t, n_banks)
        # Average probabilities across tokens
        avg_probs = probs.mean(dim=(0, 1))  # (n_banks,)
        # Target: uniform distribution → maximize entropy
        entropy = -(avg_probs * torch.log(avg_probs + 1e-8)).sum()
        max_entropy = torch.log(torch.tensor(
            float(self.cfg.routing_banks), device=probs.device
        ))
        return max_entropy - entropy
