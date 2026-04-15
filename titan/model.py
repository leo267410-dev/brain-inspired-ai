"""TITAN — Tiled Interleaved Transformer with Adaptive Neurons.

Full model combining all validated innovations:

Architecture (default ~142M params):
  TokenEmbed(32768, 768) + RoPE
    → 22 × TitanCell:
        - RecurrentMixer (all layers) — O(1) local context
        - ConformalGQA (every 3rd layer) — global attention with distance scoring
        - DifficultyGate + ScratchpadRefiner — adaptive compute
        - SmartMoEFFN (shared oscillatory banks) — richer per-parameter compute
    → FinalNorm → LMHead (tied) + MultiTokenHeads (4)

Design principles:
  1. Massive parallelism: all core ops are matmul-based, MoE parallelizes across banks
  2. Parameter efficiency: shared banks, GQA (2 KV heads), weight tying, hybrid recurrent
  3. Richer primitives: oscillatory neurons do more per parameter than standard activations
  4. Adaptive compute: easy tokens skip refinement, hard tokens get iterative scratchpad
  5. Multi-scale: local recurrent + global attention at different layer frequencies
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from titan.config import TitanConfig
from titan.layers.attention import ConformalGQA
from titan.layers.moe import OscillatoryExpertBank, SmartMoEFFN
from titan.layers.norms import RMSNorm
from titan.layers.recurrent import RecurrentMixer
from titan.layers.scratchpad import DifficultyGate, ScratchpadRefiner


class TitanCell(nn.Module):
    """Single TITAN layer.

    Every layer has: recurrent mixer + difficulty gate + scratchpad + MoE FFN.
    Anchor layers additionally have: conformal GQA attention.
    """

    def __init__(
        self,
        cfg: TitanConfig,
        shared_banks: OscillatoryExpertBank,
        use_attention: bool,
    ):
        super().__init__()
        self.use_attention = use_attention
        d = cfg.d_model

        # All layers: recurrent mixer
        self.norm1 = RMSNorm(d)
        self.recurrent = RecurrentMixer(cfg)

        # Anchor layers: conformal GQA
        self.norm2 = RMSNorm(d) if use_attention else None
        self.attention = ConformalGQA(cfg) if use_attention else None

        # All layers: adaptive compute
        self.diff_gate = DifficultyGate(cfg)
        self.scratch = ScratchpadRefiner(cfg)

        # All layers: MoE FFN
        self.norm3 = RMSNorm(d)
        self.ffn = SmartMoEFFN(cfg, shared_banks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Recurrent mixer (local context)
        h = x + self.recurrent(self.norm1(x))

        # Attention anchor (global context)
        if self.use_attention:
            h = h + self.attention(self.norm2(h))

        # Adaptive compute: difficulty-gated scratchpad
        difficulty = self.diff_gate(h)
        h = self.scratch(h, difficulty)

        # MoE FFN (oscillatory expert banks)
        h = h + self.ffn(self.norm3(h))

        return h


class MultiTokenHead(nn.Module):
    """Auxiliary heads for multi-token prediction.

    Each head predicts a future token offset using a small adapter,
    reusing the main LM head weights. Benefits:
      1. Training: provides richer gradient signal (predicts multiple positions)
      2. Inference: enables speculative decoding for faster generation
    """

    def __init__(self, cfg: TitanConfig):
        super().__init__()
        self.adapters = nn.ModuleList([
            nn.Sequential(
                RMSNorm(cfg.d_model),
                nn.Linear(cfg.d_model, cfg.d_model, bias=False),
                nn.GELU(),
            )
            for _ in range(cfg.mtp_heads)
        ])

    def forward(
        self, x: torch.Tensor, lm_head: nn.Linear
    ) -> List[torch.Tensor]:
        return [lm_head(adapter(x)) for adapter in self.adapters]


class Titan(nn.Module):
    """TITAN — Tiled Interleaved Transformer with Adaptive Neurons.

    Sub-150M parameter language model with massive GPU parallelism.
    """

    def __init__(self, cfg: TitanConfig):
        super().__init__()
        self.cfg = cfg

        # Token embeddings
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # Shared oscillatory expert banks (reused across all layers)
        self.shared_banks = OscillatoryExpertBank(cfg)

        # Layer stack: hybrid recurrent-attention
        anchor_indices = set(cfg.anchor_layer_indices())
        self.layers = nn.ModuleList([
            TitanCell(
                cfg,
                self.shared_banks,
                use_attention=(i in anchor_indices),
            )
            for i in range(cfg.n_layers)
        ])

        # Output
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying: embedding = LM head (saves ~25M params)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_embeddings.weight

        # Multi-token prediction heads
        self.mtp = MultiTokenHead(cfg)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Weight initialization tuned for deep hybrid architectures."""
        for name, p in self.named_parameters():
            if p.dim() < 2:
                continue
            if "o_proj" in name or "out." in name:
                # Scale down output projections for stable residuals
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * self.cfg.n_layers))
            elif "tok_embeddings" in name:
                nn.init.normal_(p, std=0.02)
            elif "W_mag" in name or "bank_mag" in name:
                pass  # Already initialized in OscillatoryActivation / OscillatoryExpertBank
            elif "W_freq" in name or "bank_freq" in name:
                pass  # Already initialized
            elif "bank_phase" in name or "phase" in name:
                pass  # Already initialized with specific range for oscillatory diversity
            elif p.dim() >= 2:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        input_ids: torch.Tensor,
        return_mtp: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """Forward pass.

        Args:
            input_ids: (batch, seq_len) token indices
            return_mtp: if True, also return multi-token prediction logits

        Returns:
            logits: (batch, seq_len, vocab_size) next-token logits
            mtp_logits: list of (batch, seq_len, vocab_size) for future offsets,
                        or None if return_mtp=False
        """
        x = self.tok_embeddings(input_ids)

        for layer in self.layers:
            if self.cfg.gradient_checkpointing and self.training:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        mtp_logits = self.mtp(x, self.lm_head) if return_mtp else None
        return logits, mtp_logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Autoregressive generation with top-k sampling."""
        self.eval()
        out = input_ids
        for _ in range(max_new_tokens):
            idx = out[:, -self.cfg.max_seq_len:]
            logits, _ = self(idx, return_mtp=False)
            next_logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k > 0:
                values, _ = torch.topk(
                    next_logits, k=min(top_k, next_logits.size(-1))
                )
                next_logits = torch.where(
                    next_logits < values[:, [-1]],
                    torch.full_like(next_logits, float("-inf")),
                    next_logits,
                )
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            out = torch.cat([out, next_id], dim=1)
        return out

    def param_count(self) -> dict:
        """Report parameter counts by category."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Categorize
        embedding_params = self.tok_embeddings.weight.numel()
        bank_params = sum(
            p.numel() for n, p in self.named_parameters() if "shared_banks" in n
        )
        attention_params = sum(
            p.numel() for n, p in self.named_parameters()
            if "attention" in n and "shared_banks" not in n
        )
        recurrent_params = sum(
            p.numel() for n, p in self.named_parameters() if "recurrent" in n
        )
        mtp_params = sum(
            p.numel() for n, p in self.named_parameters() if "mtp" in n
        )

        return {
            "total": total,
            "total_millions": round(total / 1e6, 2),
            "trainable": trainable,
            "under_150m": total < 150_000_000,
            "breakdown": {
                "embedding": embedding_params,
                "shared_expert_banks": bank_params,
                "attention_anchors": attention_params,
                "recurrent_mixer": recurrent_params,
                "multi_token_heads": mtp_params,
                "other": total - embedding_params - bank_params
                    - attention_params - recurrent_params - mtp_params,
            },
        }
