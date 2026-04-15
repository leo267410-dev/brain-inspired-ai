"""TITAN training losses.

Multi-objective training with:
  1. Next-token prediction (standard cross-entropy)
  2. Multi-token prediction (auxiliary heads predict future offsets)
  3. Router entropy regularization (prevent expert collapse)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


def titan_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mtp_logits: Optional[List[torch.Tensor]] = None,
    mtp_weight: float = 0.15,
    router_entropy_losses: Optional[List[torch.Tensor]] = None,
    router_weight: float = 0.01,
) -> Dict[str, torch.Tensor]:
    """Compute combined TITAN training loss.

    Args:
        logits: (batch, seq, vocab) main next-token logits
        targets: (batch, seq) target token ids
        mtp_logits: list of (batch, seq, vocab) for future offsets
        mtp_weight: weight for multi-token prediction loss
        router_entropy_losses: list of entropy regularization losses from MoE layers
        router_weight: weight for router entropy regularization

    Returns:
        dict with 'total', 'ntp', 'mtp', 'router' loss components
    """
    batch, seq, vocab = logits.shape

    # 1. Next-token prediction loss (shift by 1)
    ntp_loss = F.cross_entropy(
        logits[:, :-1].contiguous().view(-1, vocab),
        targets[:, 1:].contiguous().view(-1),
        ignore_index=-100,
    )

    result: Dict[str, torch.Tensor] = {"ntp": ntp_loss}
    total = ntp_loss

    # 2. Multi-token prediction losses (each head predicts offset +i)
    mtp_loss = torch.tensor(0.0, device=logits.device)
    if mtp_logits is not None:
        n_valid = 0
        for i, mtp_l in enumerate(mtp_logits):
            offset = i + 2  # head 0 predicts +2, head 1 predicts +3, etc.
            if seq <= offset:
                continue
            mtp_loss = mtp_loss + F.cross_entropy(
                mtp_l[:, :-offset].contiguous().view(-1, vocab),
                targets[:, offset:].contiguous().view(-1),
                ignore_index=-100,
            )
            n_valid += 1
        if n_valid > 0:
            mtp_loss = mtp_loss / n_valid
            total = total + mtp_weight * mtp_loss
    result["mtp"] = mtp_loss

    # 3. Router entropy regularization
    router_loss = torch.tensor(0.0, device=logits.device)
    if router_entropy_losses:
        router_loss = torch.stack(router_entropy_losses).mean()
        total = total + router_weight * router_loss
    result["router"] = router_loss

    result["total"] = total
    return result
