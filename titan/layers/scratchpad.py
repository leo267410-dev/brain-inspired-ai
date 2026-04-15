"""Adaptive Compute — Difficulty-Gated Scratchpad Refiner.

The key insight: not all tokens need the same amount of compute.
Common words like "the", "is", "a" can be predicted with minimal processing.
Complex reasoning tokens (variable names, logic connectives, numbers) need more.

The DifficultyGate produces a per-token scalar [0,1] indicating how much
extra compute to allocate. The ScratchpadRefiner then applies iterative
latent refinement only to the extent indicated by the difficulty score.

This is "thinking before answering" — the model can do internal reasoning
steps without producing output tokens. At inference, easy tokens fly through
with near-zero overhead; hard tokens get multiple refinement passes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from titan.config import TitanConfig


class DifficultyGate(nn.Module):
    """Per-token difficulty estimator. Output in [0, 1]."""

    def __init__(self, cfg: TitanConfig):
        super().__init__()
        self.proj = nn.Linear(cfg.d_model, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.proj(x))  # (b, t, 1)


class ScratchpadRefiner(nn.Module):
    """Iterative latent refinement module.

    Maintains a bank of learnable "scratch slots" — latent vectors that the
    model can read from and write to during refinement. Each step:
      1. Score which slots are relevant to current hidden state
      2. Read a weighted combination of slots
      3. Mix with current state through a bottleneck
      4. Gate the update by difficulty score

    Number of effective refinement steps is controlled by the difficulty gate:
    easy tokens (difficulty ≈ 0) pass through unchanged; hard tokens
    (difficulty ≈ 1) get full refinement.
    """

    def __init__(self, cfg: TitanConfig):
        super().__init__()
        d = cfg.d_model
        self.steps = cfg.scratch_steps
        self.slots = nn.Parameter(
            torch.randn(cfg.scratch_slots, d) / (d**0.5)
        )
        self.score = nn.Linear(d, cfg.scratch_slots, bias=False)
        self.mix = nn.Linear(d * 2, cfg.scratch_bottleneck, bias=False)
        self.update = nn.Linear(cfg.scratch_bottleneck, d * 2, bias=False)

    def forward(
        self, x: torch.Tensor, difficulty: torch.Tensor
    ) -> torch.Tensor:
        if self.steps <= 0:
            return x

        batch, seq, dim = x.shape
        refined = x
        # Expand slots for broadcasting: (1, 1, n_slots, d)
        slot_bank = self.slots.unsqueeze(0).unsqueeze(0).expand(
            batch, seq, -1, -1
        )

        for _ in range(self.steps):
            # Score relevance of each slot
            weights = torch.softmax(self.score(refined), dim=-1)
            # Weighted read from slots
            latent = torch.sum(weights.unsqueeze(-1) * slot_bank, dim=-2)
            # Mix current state with slot readout
            h = torch.tanh(self.mix(torch.cat([refined, latent], dim=-1)))
            # Produce gated update
            gate, val = self.update(h).chunk(2, dim=-1)
            gate = torch.sigmoid(gate) * difficulty
            refined = refined + gate * torch.tanh(val)

        return refined
