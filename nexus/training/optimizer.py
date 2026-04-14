"""Custom optimizers and learning rate schedulers for NEXUS-Ω."""

from __future__ import annotations

import math
from typing import Iterable, Optional, Tuple

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class NexusOptimizer(Optimizer):
    """
    Custom AdamW variant with:
    - Gradient centralization
    - Adaptive gradient clipping
    - Separate learning rates for different parameter groups
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 3e-4,
        betas: Tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
        grad_centralization: bool = True,
        agc_clip: float = 0.01,
    ):
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            grad_centralization=grad_centralization, agc_clip=agc_clip,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[object] = None) -> Optional[float]:
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("NexusOptimizer does not support sparse gradients")

                # Gradient centralization (for weight matrices)
                if group["grad_centralization"] and grad.dim() > 1:
                    grad = grad - grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True)

                # Adaptive gradient clipping
                if group["agc_clip"] > 0:
                    param_norm = p.data.norm(2)
                    grad_norm = grad.norm(2)
                    if grad_norm > 0:
                        clip_ratio = group["agc_clip"] * param_norm / (grad_norm + 1e-8)
                        if clip_ratio < 1.0:
                            grad = grad * clip_ratio

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                state["step"] += 1

                # Decoupled weight decay
                p.data.mul_(1 - group["lr"] * group["weight_decay"])

                # Adam update
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bc1 = 1 - beta1 ** state["step"]
                bc2 = 1 - beta2 ** state["step"]
                step_size = group["lr"] / bc1

                denom = (exp_avg_sq.sqrt() / math.sqrt(bc2)).add_(group["eps"])
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


def create_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """
    Create a cosine annealing schedule with linear warmup.

    Args:
        optimizer: the optimizer.
        warmup_steps: number of warmup steps.
        total_steps: total number of training steps.
        min_lr_ratio: minimum LR as fraction of peak LR.

    Returns:
        LambdaLR scheduler.
    """

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (
            1 + math.cos(math.pi * progress)
        )

    return LambdaLR(optimizer, lr_lambda)
