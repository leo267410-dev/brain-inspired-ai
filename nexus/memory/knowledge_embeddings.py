"""Trainable knowledge embedding vectors for factual knowledge compression."""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from nexus.config import NexusOmegaConfig


class KnowledgeEmbeddings(nn.Module):
    """
    Trainable knowledge vectors that store compressed factual knowledge.
    These are retrieved via attention and injected into the hidden state.
    Each vector represents a learned knowledge "slot" that captures
    distributional patterns from training data.
    """

    def __init__(self, config: NexusOmegaConfig):
        super().__init__()
        self.num_vectors = config.num_knowledge_vectors
        self.hidden_dim = config.hidden_dim

        # Knowledge vectors
        self.knowledge = nn.Parameter(
            torch.randn(config.num_knowledge_vectors, config.hidden_dim) * 0.02,
        )

        # Query projection for retrieval
        self.query_proj = nn.Linear(config.hidden_dim, config.hidden_dim)

        # Output projection and gate
        self.output_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.gate = nn.Linear(config.hidden_dim * 2, 1)

    def retrieve(
        self, query: torch.Tensor, top_k: int = 32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve relevant knowledge vectors.

        Args:
            query: (batch, seq_len, hidden_dim)
            top_k: number of knowledge vectors to retrieve.

        Returns:
            Tuple of (retrieved_knowledge, attention_weights).
        """
        B, L, D = query.shape
        q = self.query_proj(query)  # (B, L, D)

        # Compute attention over knowledge vectors
        # (B, L, D) @ (D, K) -> (B, L, K)
        attn_logits = torch.matmul(q, self.knowledge.T) / (D ** 0.5)

        # Sparse top-k attention
        top_k_logits, top_k_idx = torch.topk(attn_logits, top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)  # (B, L, top_k)

        # Gather knowledge vectors
        top_k_knowledge = self.knowledge[top_k_idx.reshape(-1)].reshape(
            B, L, top_k, D,
        )

        # Weighted sum
        retrieved = (top_k_weights.unsqueeze(-1) * top_k_knowledge).sum(dim=2)
        retrieved = self.output_proj(retrieved)

        return retrieved, top_k_weights

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Retrieve and inject knowledge into hidden states.

        Args:
            hidden: (batch, seq_len, hidden_dim)

        Returns:
            (batch, seq_len, hidden_dim) knowledge-augmented states.
        """
        knowledge, _ = self.retrieve(hidden)
        gate = torch.sigmoid(
            self.gate(torch.cat([hidden, knowledge], dim=-1)),
        )
        return hidden + gate * knowledge
