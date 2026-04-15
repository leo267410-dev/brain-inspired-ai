"""TITAN optimizer and learning rate schedule.

Parameter group separation:
  - Embeddings: lower LR (structure-aware initialization shouldn't be overwritten)
  - Oscillatory params (phase, freq): no weight decay (they encode frequencies)
  - Everything else: standard AdamW with cosine schedule
"""

from __future__ import annotations

import math

import torch

from titan.model import Titan


def build_optimizer(
    model: Titan,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
) -> torch.optim.AdamW:
    """Build AdamW optimizer with parameter group separation."""
    no_decay_keywords = {"phase", "freq", "bias", "norm", "slots"}
    emb_keywords = {"tok_embeddings"}

    groups = [
        # Group 1: embeddings (lower LR, no decay)
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(e in n for e in emb_keywords) and p.requires_grad
            ],
            "lr": lr * 0.3,
            "weight_decay": 0.0,
        },
        # Group 2: oscillatory / normalization params (no decay)
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay_keywords)
                and not any(e in n for e in emb_keywords)
                and p.requires_grad
            ],
            "lr": lr,
            "weight_decay": 0.0,
        },
        # Group 3: everything else (full decay)
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay_keywords)
                and not any(e in n for e in emb_keywords)
                and p.requires_grad
            ],
            "lr": lr,
            "weight_decay": weight_decay,
        },
    ]

    # Filter out empty groups
    groups = [g for g in groups if g["params"]]

    return torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8)


def cosine_warmup_schedule(
    optimizer: torch.optim.Optimizer,
    step: int,
    warmup: int,
    total: int,
    lr_min: float = 1e-5,
) -> None:
    """Update learning rate with cosine schedule and linear warmup."""
    for group in optimizer.param_groups:
        peak = group.get("_lr_peak", group["lr"])
        group.setdefault("_lr_peak", group["lr"])

        if step < warmup:
            group["lr"] = peak * step / max(warmup, 1)
        else:
            progress = (step - warmup) / max(total - warmup, 1)
            group["lr"] = lr_min + 0.5 * (peak - lr_min) * (
                1 + math.cos(math.pi * progress)
            )
