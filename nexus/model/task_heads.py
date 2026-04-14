"""Task-specific output heads for language modeling and code generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from nexus.config import NexusOmegaConfig


class LanguageModelHead(nn.Module):
    """Standard language model head projecting to vocabulary logits."""

    def __init__(self, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: (batch, seq_len, hidden_dim)

        Returns:
            (batch, seq_len, vocab_size) logits.
        """
        return self.proj(self.norm(hidden))

    def tie_weights(self, embedding_weight: nn.Parameter) -> None:
        """Tie projection weights to token embedding weights."""
        self.proj.weight = embedding_weight


class CodeCompletionHead(nn.Module):
    """Head specialized for code completion with syntax awareness.

    Uses a small syntax adapter that modifies hidden states before
    projecting through the (optionally shared) LM projection.
    """

    def __init__(self, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.syntax_adapter = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, hidden_dim),
        )
        self.proj = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: (batch, seq_len, hidden_dim)

        Returns:
            (batch, seq_len, vocab_size) logits.
        """
        normed = self.norm(hidden)
        syntax_offset = self.syntax_adapter(normed)
        return self.proj(normed + syntax_offset)

    def tie_weights(self, embedding_weight: nn.Parameter) -> None:
        """Tie projection weights to token embedding weights."""
        self.proj.weight = embedding_weight


class ClassificationHead(nn.Module):
    """Classification head for sequence classification tasks."""

    def __init__(self, hidden_dim: int, num_classes: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: (batch, seq_len, hidden_dim) — uses first token for CLS.

        Returns:
            (batch, num_classes) logits.
        """
        cls_hidden = self.norm(hidden[:, 0])
        return self.classifier(cls_hidden)


class TaskHeadRouter(nn.Module):
    """Routes hidden states to the appropriate task head."""

    def __init__(self, config: NexusOmegaConfig):
        super().__init__()
        self.lm_head = LanguageModelHead(config.hidden_dim, config.vocab_size)
        self.code_head = CodeCompletionHead(config.hidden_dim, config.vocab_size)

    def tie_weights(self, embedding_weight: nn.Parameter) -> None:
        """Tie both head projections to the token embedding weights."""
        self.lm_head.tie_weights(embedding_weight)
        self.code_head.tie_weights(embedding_weight)

    def forward(
        self,
        hidden: torch.Tensor,
        task: str = "lm",
    ) -> torch.Tensor:
        """
        Args:
            hidden: (batch, seq_len, hidden_dim)
            task: one of "lm", "code".

        Returns:
            (batch, seq_len, vocab_size) logits.
        """
        if task == "code":
            return self.code_head(hidden)
        return self.lm_head(hidden)
