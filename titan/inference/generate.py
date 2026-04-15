"""TITAN inference with multi-token speculative decoding.

The multi-token prediction heads trained during training can be used at inference
to speculatively predict multiple future tokens in parallel, then verify them
with a single forward pass. This gives up to Nx speedup (where N = number of
correctly predicted tokens) with zero quality loss.

Speculative decoding flow:
  1. Run model forward → get main logits + MTP head logits
  2. Sample next token from main logits
  3. Speculatively sample future tokens from MTP heads
  4. On next forward pass, verify speculative tokens
  5. Accept all correct speculations, reject from first mismatch
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from titan.model import Titan


@torch.no_grad()
def speculative_generate(
    model: Titan,
    input_ids: torch.Tensor,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 50,
    use_speculation: bool = True,
) -> torch.Tensor:
    """Generate tokens with optional multi-token speculative decoding.

    Args:
        model: trained TITAN model
        input_ids: (batch, seq) prompt token ids
        max_new_tokens: maximum tokens to generate
        temperature: sampling temperature
        top_k: top-k filtering
        use_speculation: if True, use MTP heads for speculative decoding

    Returns:
        (batch, seq + generated) full sequence including prompt
    """
    model.eval()
    out = input_ids
    tokens_generated = 0

    while tokens_generated < max_new_tokens:
        idx = out[:, -model.cfg.max_seq_len:]
        logits, mtp_logits = model(idx, return_mtp=use_speculation)

        # Sample next token from main head
        next_token = _sample(logits[:, -1, :], temperature, top_k)
        out = torch.cat([out, next_token], dim=1)
        tokens_generated += 1

        if tokens_generated >= max_new_tokens:
            break

        # Speculative: sample from MTP heads and accept greedily
        if use_speculation and mtp_logits is not None:
            spec_tokens = []
            for mtp_l in mtp_logits:
                spec = _sample(mtp_l[:, -1, :], temperature, top_k)
                spec_tokens.append(spec)

            # Verify speculative tokens with a single forward pass
            spec_seq = torch.cat([out[:, -1:]] + spec_tokens, dim=1)
            verify_logits, _ = model(
                torch.cat([idx, spec_seq], dim=1)[:, -model.cfg.max_seq_len:],
                return_mtp=False,
            )

            # Accept tokens that match greedy verification
            n_accepted = 0
            for i, spec_tok in enumerate(spec_tokens):
                if tokens_generated >= max_new_tokens:
                    break
                verify_pos = -(len(spec_tokens) + 1 - i)
                verified = _sample(
                    verify_logits[:, verify_pos, :], temperature, top_k
                )
                if torch.equal(verified, spec_tok):
                    out = torch.cat([out, spec_tok], dim=1)
                    tokens_generated += 1
                    n_accepted += 1
                else:
                    # Mismatch: use verified token instead and stop speculation
                    out = torch.cat([out, verified], dim=1)
                    tokens_generated += 1
                    break

    return out


def _sample(
    logits: torch.Tensor,
    temperature: float = 0.8,
    top_k: int = 50,
) -> torch.Tensor:
    """Sample from logits with temperature and top-k filtering."""
    logits = logits / max(temperature, 1e-5)
    if top_k > 0:
        values, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
        logits = torch.where(
            logits < values[:, [-1]],
            torch.full_like(logits, float("-inf")),
            logits,
        )
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
