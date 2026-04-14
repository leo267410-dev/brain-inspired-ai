"""Loss functions for NEXUS-Ω training."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class NexusLoss(nn.Module):
    """
    Composite loss for NEXUS-Ω combining:
    - Language modeling cross-entropy
    - MoE load balancing
    - Early exit auxiliary loss
    - Thought engine consistency loss
    - Smart neuron sparsity regularization
    """

    def __init__(
        self,
        moe_balance_weight: float = 0.01,
        early_exit_weight: float = 0.1,
        thought_weight: float = 0.05,
        sparsity_weight: float = 0.01,
    ):
        super().__init__()
        self.moe_balance_weight = moe_balance_weight
        self.early_exit_weight = early_exit_weight
        self.thought_weight = thought_weight
        self.sparsity_weight = sparsity_weight

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        moe_loss: Optional[torch.Tensor] = None,
        early_exit_losses: Optional[list[torch.Tensor]] = None,
        thought_loss: Optional[torch.Tensor] = None,
        sparsity_loss: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute composite loss.

        Args:
            logits: (batch, seq_len, vocab_size) model predictions.
            labels: (batch, seq_len) ground truth token IDs.
            moe_loss: scalar MoE load balancing loss.
            early_exit_losses: list of per-layer early exit losses.
            thought_loss: scalar thought engine consistency loss.
            sparsity_loss: scalar neuron sparsity regularization.

        Returns:
            Dict with individual and total losses.
        """
        # Primary LM loss
        lm_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

        total_loss = lm_loss
        loss_dict: Dict[str, torch.Tensor] = {"lm_loss": lm_loss}

        # MoE load balancing
        if moe_loss is not None:
            weighted_moe = self.moe_balance_weight * moe_loss
            total_loss = total_loss + weighted_moe
            loss_dict["moe_loss"] = weighted_moe

        # Early exit auxiliary losses
        if early_exit_losses:
            exit_loss = sum(early_exit_losses) / len(early_exit_losses)
            weighted_exit = self.early_exit_weight * exit_loss
            total_loss = total_loss + weighted_exit
            loss_dict["early_exit_loss"] = weighted_exit

        # Thought engine consistency
        if thought_loss is not None:
            weighted_thought = self.thought_weight * thought_loss
            total_loss = total_loss + weighted_thought
            loss_dict["thought_loss"] = weighted_thought

        # Sparsity regularization
        if sparsity_loss is not None:
            weighted_sparsity = self.sparsity_weight * sparsity_loss
            total_loss = total_loss + weighted_sparsity
            loss_dict["sparsity_loss"] = weighted_sparsity

        loss_dict["total_loss"] = total_loss
        return loss_dict


class ContrastiveLoss(nn.Module):
    """Contrastive loss for representation learning."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self, anchor: torch.Tensor, positive: torch.Tensor, negatives: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute InfoNCE contrastive loss.

        Args:
            anchor: (batch, hidden_dim) anchor representations.
            positive: (batch, hidden_dim) positive examples.
            negatives: (batch, num_neg, hidden_dim) negative examples.

        Returns:
            Scalar loss.
        """
        # Normalize
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negatives = F.normalize(negatives, dim=-1)

        # Positive similarity
        pos_sim = (anchor * positive).sum(dim=-1, keepdim=True) / self.temperature

        # Negative similarities
        neg_sim = torch.bmm(negatives, anchor.unsqueeze(-1)).squeeze(-1) / self.temperature

        # InfoNCE
        logits = torch.cat([pos_sim, neg_sim], dim=-1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

        return F.cross_entropy(logits, labels)
