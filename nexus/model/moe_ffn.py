"""Shared Expert Pool with Mixture-of-Experts feed-forward network."""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from nexus.config import NexusOmegaConfig


class Expert(nn.Module):
    """Single expert: 2-layer FFN with SwiGLU activation."""

    def __init__(self, hidden_dim: int, expert_dim: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, expert_dim, bias=False)
        self.w2 = nn.Linear(expert_dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, expert_dim, bias=False)  # gate for SwiGLU

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., hidden_dim)

        Returns:
            (..., hidden_dim)
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Router(nn.Module):
    """Top-k expert router with load balancing loss."""

    def __init__(self, hidden_dim: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self.top_k = top_k
        self.num_experts = num_experts

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch * seq_len, hidden_dim)

        Returns:
            Tuple of (top_k_weights, top_k_indices, load_balance_loss).
        """
        logits = self.gate(x)  # (N, num_experts)
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        # Load balancing loss
        routing_probs = F.softmax(logits, dim=-1)
        expert_usage = routing_probs.mean(dim=0)
        load_balance_loss = self.num_experts * (expert_usage * expert_usage).sum()

        return top_k_weights, top_k_indices, load_balance_loss


class SharedExpertPool(nn.Module):
    """
    64 experts shared across ALL 24 layers.
    Each layer has its own router but draws from the same expert pool.
    This creates 64^24 possible computation paths.
    """

    def __init__(self, config: NexusOmegaConfig):
        super().__init__()
        self.num_experts = config.num_shared_experts
        self.top_k = config.top_k_experts

        self.experts = nn.ModuleList([
            Expert(config.hidden_dim, config.expert_dim)
            for _ in range(config.num_shared_experts)
        ])

        # Each layer gets its own router (populated via register_router)
        self.routers = nn.ModuleDict()

        # Track expert utilization
        self.register_buffer(
            "expert_counts", torch.zeros(config.num_shared_experts),
        )

    def register_router(
        self, layer_idx: int, hidden_dim: int, num_experts: int, top_k: int,
    ) -> None:
        """Register a per-layer router for the shared expert pool."""
        self.routers[str(layer_idx)] = Router(hidden_dim, num_experts, top_k)

    def forward(self, x: torch.Tensor, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch * seq_len, hidden_dim)
            layer_idx: which layer is calling (determines which router to use).

        Returns:
            Tuple of (output, load_balance_loss).
        """
        router = self.routers[str(layer_idx)]
        weights, indices, lb_loss = router(x)

        # Dispatch to experts
        output = torch.zeros_like(x)
        for k in range(router.top_k):
            expert_indices = indices[:, k]  # (N,)
            expert_weights = weights[:, k: k + 1]  # (N, 1)

            for expert_idx in range(len(self.experts)):
                mask = expert_indices == expert_idx
                if mask.any():
                    expert_input = x[mask]
                    expert_output = self.experts[expert_idx](expert_input)
                    output[mask] += expert_weights[mask] * expert_output

                    if self.training:
                        self.expert_counts[expert_idx] += mask.sum().item()

        return output, lb_loss
