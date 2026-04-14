"""Recursive reasoning with scratchpad for deep multi-step inference."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from nexus.config import NexusOmegaConfig


class Scratchpad(nn.Module):
    """
    Persistent scratchpad memory for recursive reasoning.
    Allows the model to read/write intermediate computation results.
    """

    def __init__(self, hidden_dim: int, scratchpad_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.scratchpad_dim = scratchpad_dim

        self.write_gate = nn.Linear(hidden_dim, scratchpad_dim)
        self.write_value = nn.Linear(hidden_dim, scratchpad_dim)
        self.read_proj = nn.Linear(scratchpad_dim, hidden_dim)
        self.blend_gate = nn.Linear(hidden_dim + scratchpad_dim, 1)

    def write(self, hidden: torch.Tensor, scratchpad: torch.Tensor) -> torch.Tensor:
        """
        Write to scratchpad.

        Args:
            hidden: (batch, seq_len, hidden_dim)
            scratchpad: (batch, seq_len, scratchpad_dim)

        Returns:
            Updated (batch, seq_len, scratchpad_dim) scratchpad.
        """
        gate = torch.sigmoid(self.write_gate(hidden))  # (B, L, scratchpad_dim)
        value = torch.tanh(self.write_value(hidden))  # (B, L, scratchpad_dim)
        return scratchpad * (1 - gate) + value * gate

    def read(self, hidden: torch.Tensor, scratchpad: torch.Tensor) -> torch.Tensor:
        """
        Read from scratchpad and blend with hidden state.

        Args:
            hidden: (batch, seq_len, hidden_dim)
            scratchpad: (batch, seq_len, scratchpad_dim)

        Returns:
            (batch, seq_len, hidden_dim) blended output.
        """
        read_val = self.read_proj(scratchpad)  # (B, L, hidden_dim)
        blend = torch.sigmoid(
            self.blend_gate(torch.cat([hidden, scratchpad], dim=-1)),
        )
        return hidden + blend * read_val


class RecursiveReasoningBlock(nn.Module):
    """
    Enables recursive computation by looping through a subset of layers
    multiple times with a scratchpad for intermediate results.
    """

    def __init__(self, config: NexusOmegaConfig):
        super().__init__()
        self.max_loops = config.max_recursive_loops
        self.scratchpad_dim = config.scratchpad_dim
        self.hidden_dim = config.hidden_dim

        self.scratchpad = Scratchpad(config.hidden_dim, config.scratchpad_dim)

        # Loop controller: decides whether to continue looping
        self.loop_controller = nn.Sequential(
            nn.Linear(config.hidden_dim + config.scratchpad_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Iteration embedding (tells the model which iteration it's on)
        self.iteration_embedding = nn.Embedding(config.max_recursive_loops, config.hidden_dim)

    def forward(
        self,
        hidden: torch.Tensor,
        layer_fn: Optional[object] = None,
    ) -> Tuple[torch.Tensor, int]:
        """
        Run recursive reasoning.

        Args:
            hidden: (batch, seq_len, hidden_dim)
            layer_fn: callable that applies one pass of the reasoning layers.

        Returns:
            Tuple of (output hidden, number of loops executed).
        """
        B, L, _ = hidden.shape
        device = hidden.device

        scratchpad_state = torch.zeros(
            B, L, self.scratchpad_dim, device=device, dtype=hidden.dtype,
        )

        num_loops = 0
        for i in range(self.max_loops):
            # Add iteration embedding
            iter_emb = self.iteration_embedding(
                torch.tensor(i, device=device),
            ).unsqueeze(0).unsqueeze(0)
            hidden = hidden + iter_emb

            # Apply reasoning layers if provided
            if layer_fn is not None:
                hidden = layer_fn(hidden)

            # Update scratchpad
            scratchpad_state = self.scratchpad.write(hidden, scratchpad_state)

            # Read from scratchpad
            hidden = self.scratchpad.read(hidden, scratchpad_state)

            num_loops += 1

            # Check if we should continue (during inference)
            if not self.training:
                continue_prob = self.loop_controller(
                    torch.cat([hidden, scratchpad_state], dim=-1),
                ).mean()
                if continue_prob < 0.5:
                    break

        return hidden, num_loops
