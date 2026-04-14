"""Confidence estimation for thought engine and self-verification."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConfidenceEstimator(nn.Module):
    """Estimates confidence that the model's reasoning is complete/correct."""

    def __init__(self, hidden_dim: int, num_buckets: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.num_buckets = num_buckets

        # Calibration: track predicted vs actual accuracy per bucket
        self.register_buffer("bucket_correct", torch.zeros(num_buckets))
        self.register_buffer("bucket_total", torch.zeros(num_buckets))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: (..., hidden_dim)

        Returns:
            (...) confidence scores in [0, 1].
        """
        return self.net(hidden).squeeze(-1)

    @torch.no_grad()
    def update_calibration(
        self, predicted_conf: torch.Tensor, correct: torch.Tensor,
    ) -> None:
        """
        Update calibration statistics.

        Args:
            predicted_conf: (N,) predicted confidence scores.
            correct: (N,) binary correctness labels.
        """
        bucket_ids = (predicted_conf * self.num_buckets).long().clamp(
            0, self.num_buckets - 1,
        )
        for b in range(self.num_buckets):
            mask = bucket_ids == b
            if mask.any():
                self.bucket_correct[b] += correct[mask].sum()
                self.bucket_total[b] += mask.sum()

    def get_calibration_error(self) -> float:
        """Compute Expected Calibration Error (ECE)."""
        total = self.bucket_total.sum().item()
        if total == 0:
            return 0.0

        ece = 0.0
        for b in range(self.num_buckets):
            if self.bucket_total[b] > 0:
                avg_conf = (b + 0.5) / self.num_buckets
                avg_acc = (self.bucket_correct[b] / self.bucket_total[b]).item()
                weight = self.bucket_total[b].item() / total
                ece += weight * abs(avg_conf - avg_acc)
        return ece
