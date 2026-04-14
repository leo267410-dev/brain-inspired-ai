"""Approximate nearest neighbor index for fast memory retrieval."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


class MemoryIndex:
    """
    Approximate nearest neighbor index using locality-sensitive hashing.
    Provides sub-linear retrieval from the external memory bank.
    """

    def __init__(
        self,
        key_dim: int,
        num_tables: int = 8,
        num_buckets: int = 256,
    ):
        self.key_dim = key_dim
        self.num_tables = num_tables
        self.num_buckets = num_buckets

        # Random projection vectors for LSH
        self.projections = torch.randn(num_tables, key_dim, num_buckets // 2)
        self.projections = self.projections / self.projections.norm(dim=1, keepdim=True)

        # Hash tables: table_idx -> bucket_idx -> list of memory indices
        self.tables: list[dict[int, list[int]]] = [
            {} for _ in range(num_tables)
        ]

    def _hash(self, keys: torch.Tensor) -> torch.Tensor:
        """
        Compute LSH hashes for a batch of keys.

        Args:
            keys: (N, key_dim)

        Returns:
            (N, num_tables) bucket indices.
        """
        projections = self.projections.to(keys.device)
        # Project keys onto random hyperplanes: (N, key_dim) @ (T, key_dim, B//2) -> (N, T, B//2)
        projected = torch.einsum("nk,tkb->ntb", keys, projections)
        # Binary hash: sign of projections
        binary = (projected > 0).long()
        # Convert to bucket index
        powers = 2 ** torch.arange(binary.shape[-1], device=keys.device)
        bucket_ids = (binary * powers).sum(dim=-1) % self.num_buckets  # (N, T)
        return bucket_ids

    def build(self, keys: torch.Tensor) -> None:
        """
        Build the index from a set of keys.

        Args:
            keys: (N, key_dim)
        """
        self.tables = [{} for _ in range(self.num_tables)]
        bucket_ids = self._hash(keys)  # (N, T)

        for idx in range(keys.shape[0]):
            for t in range(self.num_tables):
                bucket = bucket_ids[idx, t].item()
                if bucket not in self.tables[t]:
                    self.tables[t][bucket] = []
                self.tables[t][bucket].append(idx)

    def query(
        self,
        query_keys: torch.Tensor,
        all_keys: torch.Tensor,
        top_k: int = 32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Query the index for approximate nearest neighbors.

        Args:
            query_keys: (Q, key_dim) query vectors.
            all_keys: (N, key_dim) all stored keys (for re-ranking).
            top_k: number of results to return.

        Returns:
            Tuple of (indices, similarities) each (Q, top_k).
        """
        Q = query_keys.shape[0]
        bucket_ids = self._hash(query_keys)  # (Q, T)

        all_indices = []
        for q in range(Q):
            candidates: set[int] = set()
            for t in range(self.num_tables):
                bucket = bucket_ids[q, t].item()
                if bucket in self.tables[t]:
                    candidates.update(self.tables[t][bucket])

            if len(candidates) == 0:
                # Fallback to random indices
                candidates = set(range(min(top_k, all_keys.shape[0])))

            candidate_list = list(candidates)
            candidate_tensor = torch.tensor(candidate_list, device=query_keys.device)
            candidate_keys = all_keys[candidate_tensor]

            # Re-rank by exact similarity
            sim = F.cosine_similarity(
                query_keys[q].unsqueeze(0), candidate_keys, dim=-1,
            )
            k = min(top_k, len(candidate_list))
            top_sim, top_local_idx = torch.topk(sim, k)
            top_global_idx = candidate_tensor[top_local_idx]

            # Pad if needed
            if k < top_k:
                pad_idx = torch.zeros(
                    top_k - k, dtype=torch.long, device=query_keys.device,
                )
                pad_sim = torch.zeros(top_k - k, device=query_keys.device)
                top_global_idx = torch.cat([top_global_idx, pad_idx])
                top_sim = torch.cat([top_sim, pad_sim])

            all_indices.append((top_global_idx, top_sim))

        indices = torch.stack([x[0] for x in all_indices])
        similarities = torch.stack([x[1] for x in all_indices])
        return indices, similarities
