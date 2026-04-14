"""Meta-controller for dynamic computation allocation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from nexus.config import NexusOmegaConfig


class MetaController(nn.Module):
    """
    Meta-controller that dynamically allocates computation resources.
    Decides per-token:
    - How many layers to use (early exit)
    - How wide each layer should be (dynamic width)
    - How many recursive loops to run
    - Whether to invoke external memory
    - Whether to activate the thought engine
    """

    def __init__(self, config: NexusOmegaConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim

        # Shared encoder for input difficulty estimation
        self.difficulty_encoder = nn.Sequential(
            nn.Linear(config.hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
        )

        # Decision heads
        self.depth_head = nn.Linear(128, 1)  # predicted optimal depth (0-1 fraction)
        self.width_head = nn.Linear(128, 1)  # predicted width usage (0-1 fraction)
        self.recursion_head = nn.Linear(128, config.max_recursive_loops)  # loop count
        self.memory_head = nn.Linear(128, 1)  # whether to query memory
        self.thought_head = nn.Linear(128, 1)  # whether to activate thought engine

    def forward(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Analyze input difficulty and recommend computation allocation.

        Args:
            hidden: (batch, seq_len, hidden_dim) input hidden states.

        Returns:
            Dict with per-token computation recommendations.
        """
        # Pool over sequence for global difficulty estimate
        pooled = hidden.mean(dim=1)  # (batch, hidden_dim)
        features = self.difficulty_encoder(pooled)  # (batch, 128)

        return {
            "depth_fraction": torch.sigmoid(self.depth_head(features)),
            "width_fraction": torch.sigmoid(self.width_head(features)),
            "recursion_logits": self.recursion_head(features),
            "use_memory": torch.sigmoid(self.memory_head(features)),
            "use_thought": torch.sigmoid(self.thought_head(features)),
        }

    def get_depth_budget(self, hidden: torch.Tensor, total_layers: int) -> int:
        """
        Determine how many layers to use for this input.

        Args:
            hidden: (batch, seq_len, hidden_dim)
            total_layers: total number of layers in the model.

        Returns:
            Number of layers to use.
        """
        with torch.no_grad():
            decisions = self.forward(hidden)
            fraction = decisions["depth_fraction"].mean().item()
            return max(1, int(fraction * total_layers))

    def get_recursion_count(self, hidden: torch.Tensor) -> int:
        """
        Determine how many recursive loops to use.

        Args:
            hidden: (batch, seq_len, hidden_dim)

        Returns:
            Number of recursive loops.
        """
        with torch.no_grad():
            decisions = self.forward(hidden)
            logits = decisions["recursion_logits"]
            return logits.argmax(dim=-1).mode().values.item() + 1

    def should_use_memory(self, hidden: torch.Tensor) -> bool:
        """Check if external memory should be queried."""
        with torch.no_grad():
            decisions = self.forward(hidden)
            return decisions["use_memory"].mean().item() > 0.5

    def should_use_thought(self, hidden: torch.Tensor) -> bool:
        """Check if thought engine should be activated."""
        with torch.no_grad():
            decisions = self.forward(hidden)
            return decisions["use_thought"].mean().item() > 0.5
