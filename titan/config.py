"""TITAN model configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class TitanConfig:
    """Configuration for the TITAN architecture.

    Default config: ~142M parameters, fits comfortably under 150M budget.
    Hybrid recurrent-attention with oscillatory MoE experts.
    """

    # --- Vocabulary & Embedding ---
    vocab_size: int = 32768
    d_model: int = 768
    max_seq_len: int = 8192
    rope_base: float = 10000.0
    dropout: float = 0.0

    # --- Layer Stack ---
    n_layers: int = 22
    anchor_every: int = 3  # place GQA attention every N layers

    # --- Attention (anchor layers only) ---
    n_heads: int = 12
    n_kv_heads: int = 2  # grouped query attention
    use_conformal: bool = True  # conformal distance-based scoring

    # --- Recurrent Mixer (all layers) ---
    recurrent_kernel_size: int = 4

    # --- Adaptive Compute ---
    scratch_slots: int = 8
    scratch_steps: int = 2
    scratch_bottleneck: int = 192

    # --- MoE with Oscillatory Experts ---
    routing_banks: int = 8
    routing_topk: int = 2
    bank_hidden: int = 1280
    use_oscillatory: bool = True  # SRN activation in expert banks

    # --- Multi-Token Prediction ---
    mtp_heads: int = 4

    # --- Training ---
    tie_embeddings: bool = True
    use_bias: bool = False
    gradient_checkpointing: bool = True

    # --- Derived ---
    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads

    @property
    def kv_dim(self) -> int:
        return self.head_dim * self.n_kv_heads

    @property
    def n_anchor_layers(self) -> int:
        return sum(
            1 for i in range(self.n_layers) if (i + 1) % self.anchor_every == 0
        )

    def anchor_layer_indices(self) -> List[int]:
        return [i for i in range(self.n_layers) if (i + 1) % self.anchor_every == 0]
