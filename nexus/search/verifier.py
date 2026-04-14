"""Code verifier for validating generated code solutions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn


@dataclass
class VerificationResult:
    """Result of code verification."""

    is_valid: bool
    confidence: float
    errors: List[str]
    suggestions: List[str]


class CodeVerifier(nn.Module):
    """
    Neural code verifier that checks generated code for correctness.
    Uses learned representations to detect common errors.
    """

    def __init__(self, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Syntax check head
        self.syntax_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        # Semantic check head
        self.semantic_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        # Error type classifier
        self.error_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 10),  # 10 error categories
        )

        # Overall quality score
        self.quality_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, hidden: torch.Tensor) -> dict:
        """
        Verify code from hidden states.

        Args:
            hidden: (batch, seq_len, hidden_dim) code representations.

        Returns:
            Dict with verification scores.
        """
        # Pool over sequence
        pooled = hidden.mean(dim=1)  # (batch, hidden_dim)

        syntax_score = self.syntax_head(pooled).squeeze(-1)
        semantic_score = self.semantic_head(pooled).squeeze(-1)
        error_logits = self.error_classifier(pooled)
        quality_score = self.quality_head(pooled).squeeze(-1)

        return {
            "syntax_score": syntax_score,
            "semantic_score": semantic_score,
            "error_logits": error_logits,
            "quality_score": quality_score,
        }

    def verify(self, hidden: torch.Tensor) -> VerificationResult:
        """
        Run full verification and return structured result.

        Args:
            hidden: (1, seq_len, hidden_dim) single code sample.

        Returns:
            VerificationResult with detailed information.
        """
        with torch.no_grad():
            result = self.forward(hidden)

        syntax_ok = result["syntax_score"].item() > 0.5
        semantic_ok = result["semantic_score"].item() > 0.5
        quality = result["quality_score"].item()

        errors = []
        if not syntax_ok:
            errors.append("Potential syntax error detected")
        if not semantic_ok:
            errors.append("Potential semantic error detected")

        suggestions = []
        if quality < 0.5:
            suggestions.append("Consider regenerating with higher temperature")
        if quality < 0.3:
            suggestions.append("Code quality is low — try more reasoning steps")

        return VerificationResult(
            is_valid=syntax_ok and semantic_ok,
            confidence=quality,
            errors=errors,
            suggestions=suggestions,
        )
