"""Evaluation metrics for NEXUS-Ω."""

from __future__ import annotations

import math
from typing import Dict, List

import torch


def compute_perplexity(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute perplexity from logits and labels.

    Args:
        logits: (batch, seq_len, vocab_size) model predictions.
        labels: (batch, seq_len) ground truth token IDs.

    Returns:
        Perplexity score.
    """
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
        reduction="mean",
    )
    return math.exp(loss.item())


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute top-1 accuracy.

    Args:
        logits: (batch, seq_len, vocab_size) model predictions.
        labels: (batch, seq_len) ground truth.

    Returns:
        Accuracy in [0, 1].
    """
    predictions = logits.argmax(dim=-1)
    mask = labels != -100
    correct = (predictions == labels) & mask
    return correct.sum().item() / max(mask.sum().item(), 1)


def compute_topk_accuracy(
    logits: torch.Tensor, labels: torch.Tensor, k: int = 5,
) -> float:
    """
    Compute top-k accuracy.

    Args:
        logits: (batch, seq_len, vocab_size) model predictions.
        labels: (batch, seq_len) ground truth.
        k: number of top predictions to consider.

    Returns:
        Top-k accuracy in [0, 1].
    """
    mask = labels != -100
    top_k_preds = logits.topk(k, dim=-1).indices  # (B, L, k)
    labels_expanded = labels.unsqueeze(-1).expand_as(top_k_preds)
    correct = (top_k_preds == labels_expanded).any(dim=-1) & mask
    return correct.sum().item() / max(mask.sum().item(), 1)


def compute_moe_metrics(model: torch.nn.Module) -> Dict[str, float]:
    """
    Compute MoE-specific metrics (expert utilization, load balance).

    Args:
        model: model with SharedExpertPool.

    Returns:
        Dict with MoE metrics.
    """
    metrics: Dict[str, float] = {}

    for name, module in model.named_modules():
        if hasattr(module, "expert_counts"):
            counts = module.expert_counts
            if counts.sum() > 0:
                normalized = counts / counts.sum()
                entropy = -(normalized * (normalized + 1e-10).log()).sum().item()
                max_entropy = math.log(len(counts))
                metrics[f"{name}_utilization_entropy"] = entropy
                metrics[f"{name}_load_balance"] = entropy / max_entropy if max_entropy > 0 else 0.0
                metrics[f"{name}_max_utilization"] = normalized.max().item()
                metrics[f"{name}_min_utilization"] = normalized.min().item()

    return metrics


def compute_sparsity_metrics(model: torch.nn.Module) -> Dict[str, float]:
    """
    Compute neuron sparsity metrics.

    Args:
        model: model with SmartNeuronLayers.

    Returns:
        Dict with sparsity metrics.
    """
    metrics: Dict[str, float] = {}

    for name, module in model.named_modules():
        if hasattr(module, "get_utilization"):
            utilization = module.get_utilization()
            metrics[f"{name}_avg_activation"] = utilization.mean().item()
            metrics[f"{name}_active_fraction"] = (utilization > 0).float().mean().item()

    return metrics


class MetricTracker:
    """Tracks and aggregates metrics over training/evaluation."""

    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}

    def update(self, key: str, value: float) -> None:
        """Add a metric value."""
        if key not in self.metrics:
            self.metrics[key] = []
        self.metrics[key].append(value)

    def get_average(self, key: str) -> float:
        """Get the average of a metric."""
        values = self.metrics.get(key, [])
        return sum(values) / len(values) if values else 0.0

    def get_last(self, key: str) -> float:
        """Get the most recent value of a metric."""
        values = self.metrics.get(key, [])
        return values[-1] if values else 0.0

    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics.clear()

    def summary(self) -> Dict[str, float]:
        """Get summary of all tracked metrics."""
        return {k: self.get_average(k) for k in self.metrics}
