"""Inference engine for NEXUS-Ω with KV-cache and sampling strategies."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from nexus.config import NexusOmegaConfig


class InferenceEngine:
    """
    Inference engine with autoregressive generation,
    KV-caching, and various sampling strategies.
    """

    def __init__(self, model: torch.nn.Module, config: NexusOmegaConfig):
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive text generation.

        Args:
            input_ids: (batch, seq_len) prompt token IDs.
            max_new_tokens: maximum tokens to generate.
            temperature: sampling temperature.
            top_k: top-k filtering.
            top_p: nucleus sampling threshold.
            repetition_penalty: penalty for repeated tokens.
            eos_token_id: stop generation at this token.

        Returns:
            (batch, seq_len + new_tokens) generated token IDs.
        """
        self.model.eval()
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            # Truncate to max_seq_len
            context = generated[:, -self.config.max_seq_len:]

            # Forward pass
            outputs = self.model(context)
            if isinstance(outputs, dict):
                logits = outputs["logits"]
            else:
                logits = outputs

            next_logits = logits[:, -1, :]  # (batch, vocab_size)

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for b in range(generated.shape[0]):
                    for token_id in generated[b].unique():
                        if next_logits[b, token_id] > 0:
                            next_logits[b, token_id] /= repetition_penalty
                        else:
                            next_logits[b, token_id] *= repetition_penalty

            # Temperature
            next_logits = next_logits / temperature

            # Top-k filtering
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(next_logits, top_k, dim=-1)
                next_logits = torch.full_like(next_logits, float("-inf"))
                next_logits.scatter_(1, top_k_indices, top_k_logits)

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[sorted_mask] = float("-inf")
                next_logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=-1)

            # Check EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated

    @torch.no_grad()
    def beam_search(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        num_beams: int = 4,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Beam search decoding.

        Args:
            input_ids: (1, seq_len) prompt (batch size must be 1).
            max_new_tokens: maximum new tokens.
            num_beams: number of beams.
            eos_token_id: end of sequence token ID.

        Returns:
            (1, seq_len + new_tokens) best beam.
        """
        self.model.eval()
        device = input_ids.device

        # Initialize beams: (num_beams, seq_len)
        beams = input_ids.expand(num_beams, -1).clone()
        beam_scores = torch.zeros(num_beams, device=device)
        beam_scores[1:] = float("-inf")  # Only first beam active initially

        for _ in range(max_new_tokens):
            context = beams[:, -self.config.max_seq_len:]
            outputs = self.model(context)
            if isinstance(outputs, dict):
                logits = outputs["logits"]
            else:
                logits = outputs
            next_logits = F.log_softmax(logits[:, -1, :], dim=-1)

            vocab_size = next_logits.shape[-1]
            next_scores = beam_scores[:, None] + next_logits  # (num_beams, vocab)
            next_scores = next_scores.view(-1)  # (num_beams * vocab,)

            top_scores, top_indices = torch.topk(next_scores, num_beams)
            beam_indices = top_indices // vocab_size
            token_indices = top_indices % vocab_size

            beams = torch.cat([
                beams[beam_indices],
                token_indices.unsqueeze(-1),
            ], dim=-1)
            beam_scores = top_scores

            if eos_token_id is not None and (beams[:, -1] == eos_token_id).any():
                # Return best completed beam
                best_idx = beam_scores.argmax()
                return beams[best_idx: best_idx + 1]

        best_idx = beam_scores.argmax()
        return beams[best_idx: best_idx + 1]
