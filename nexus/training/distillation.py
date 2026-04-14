"""Knowledge distillation from larger teacher models."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """
    Knowledge distillation loss combining:
    - Soft target loss (KL divergence from teacher)
    - Hard target loss (cross-entropy with ground truth)
    - Feature-level distillation (hidden state matching)
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.7,
        feature_weight: float = 0.1,
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.feature_weight = feature_weight

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        student_hidden: Optional[torch.Tensor] = None,
        teacher_hidden: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute distillation loss.

        Args:
            student_logits: (batch, seq_len, vocab_size) student predictions.
            teacher_logits: (batch, seq_len, vocab_size) teacher predictions.
            labels: (batch, seq_len) ground truth token IDs.
            student_hidden: optional student hidden states for feature distillation.
            teacher_hidden: optional teacher hidden states for feature distillation.

        Returns:
            Scalar loss.
        """
        T = self.temperature

        # Soft target loss (KL divergence)
        student_soft = F.log_softmax(student_logits / T, dim=-1)
        teacher_soft = F.softmax(teacher_logits / T, dim=-1)
        soft_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (T * T)

        # Hard target loss
        hard_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss

        # Feature-level distillation
        if student_hidden is not None and teacher_hidden is not None:
            # Project teacher hidden to student dimension if needed
            if student_hidden.shape[-1] != teacher_hidden.shape[-1]:
                proj = nn.Linear(
                    teacher_hidden.shape[-1], student_hidden.shape[-1],
                ).to(student_hidden.device)
                teacher_hidden = proj(teacher_hidden)

            feature_loss = F.mse_loss(student_hidden, teacher_hidden.detach())
            total_loss += self.feature_weight * feature_loss

        return total_loss


class ProgressiveDistillation:
    """
    Progressive distillation strategy where the student is trained
    in stages, gradually reducing the teacher influence.
    """

    def __init__(
        self,
        initial_alpha: float = 0.9,
        final_alpha: float = 0.1,
        warmup_steps: int = 10_000,
        total_steps: int = 100_000,
    ):
        self.initial_alpha = initial_alpha
        self.final_alpha = final_alpha
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def get_alpha(self, step: int) -> float:
        """Get the current distillation weight."""
        if step < self.warmup_steps:
            return self.initial_alpha

        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = min(1.0, progress)
        return self.initial_alpha + (self.final_alpha - self.initial_alpha) * progress
