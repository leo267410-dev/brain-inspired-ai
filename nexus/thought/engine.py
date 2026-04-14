"""Internal thought engine for chain-of-thought reasoning."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from nexus.config import NexusOmegaConfig


class ThoughtEngine(nn.Module):
    """
    Generates internal chain-of-thought tokens before producing output.
    The thought tokens are generated in a hidden space and compressed
    before being used to condition the output.
    """

    def __init__(self, config: NexusOmegaConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.max_thought_tokens = config.max_thought_tokens
        self.confidence_threshold = config.thought_confidence_threshold

        # Thought token generator (small autoregressive head)
        self.thought_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.thought_norm = nn.LayerNorm(config.hidden_dim)

        # Single-layer transformer for thought generation
        self.thought_attn = nn.MultiheadAttention(
            config.hidden_dim, num_heads=config.num_heads,
            dropout=config.dropout, batch_first=True,
        )
        self.thought_ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
        )
        self.thought_ffn_norm = nn.LayerNorm(config.hidden_dim)

        # Confidence estimator
        from nexus.thought.confidence import ConfidenceEstimator
        self.confidence = ConfidenceEstimator(config.hidden_dim)

        # Compressor
        from nexus.thought.compressor import ThoughtCompressor
        self.compressor = ThoughtCompressor(
            config.hidden_dim, config.max_thought_tokens,
        )

    def generate_thoughts(
        self,
        hidden: torch.Tensor,
        max_steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int]:
        """
        Generate internal thought tokens.

        Args:
            hidden: (batch, seq_len, hidden_dim) input context.
            max_steps: maximum number of thought tokens to generate.

        Returns:
            Tuple of (thought_tokens, num_steps):
                thought_tokens: (batch, num_steps, hidden_dim)
                num_steps: number of thought tokens generated.
        """
        if max_steps is None:
            max_steps = self.max_thought_tokens

        B, L, D = hidden.shape

        # Initialize first thought from pooled context
        context = hidden.mean(dim=1, keepdim=True)  # (B, 1, D)
        thought_token = self.thought_proj(context)  # (B, 1, D)

        thoughts = [thought_token]
        for step in range(1, max_steps):
            # Attend over all previous thoughts + input context
            all_context = torch.cat([hidden] + thoughts, dim=1)
            attended, _ = self.thought_attn(
                thought_token, all_context, all_context,
            )
            thought_token = self.thought_norm(thought_token + attended)
            thought_token = self.thought_ffn_norm(
                thought_token + self.thought_ffn(thought_token),
            )
            thoughts.append(thought_token)

            # Check confidence (stop early if confident enough)
            if not self.training:
                conf = self.confidence(thought_token)
                if conf.mean().item() > self.confidence_threshold:
                    break

        thought_sequence = torch.cat(thoughts, dim=1)  # (B, num_steps, D)
        return thought_sequence, len(thoughts)

    def forward(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate thoughts, compress them, and return the conditioning signal.

        Args:
            hidden: (batch, seq_len, hidden_dim)

        Returns:
            Tuple of (conditioned_hidden, thought_summary):
                conditioned_hidden: (batch, seq_len, hidden_dim)
                thought_summary: (batch, compressed_len, hidden_dim)
        """
        thoughts, num_steps = self.generate_thoughts(hidden)
        thought_summary = self.compressor(thoughts)
        # Condition the hidden states with thought summary via cross-attention
        conditioned, _ = self.thought_attn(hidden, thought_summary, thought_summary)
        conditioned = self.thought_norm(hidden + conditioned)
        return conditioned, thought_summary
