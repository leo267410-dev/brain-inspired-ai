"""TITAN training loop.

Supports:
  - Mixed precision (bf16/fp16)
  - Gradient accumulation
  - Multi-token prediction training
  - Router entropy regularization
  - Cosine warmup LR schedule
  - Gradient clipping
  - Periodic logging
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from titan.model import Titan
from titan.training.losses import titan_loss
from titan.training.optimizer import build_optimizer, cosine_warmup_schedule


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 500
    total_steps: int = 10000
    batch_size: int = 8
    grad_accum_steps: int = 4
    grad_clip: float = 1.0
    precision: str = "bf16"  # "fp32", "fp16", "bf16"
    mtp_weight: float = 0.15
    router_weight: float = 0.01
    log_every: int = 10
    save_every: int = 1000
    save_path: Optional[str] = None


class TitanTrainer:
    """Training orchestrator for TITAN models."""

    def __init__(
        self,
        model: Titan,
        train_cfg: TrainConfig,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.cfg = train_cfg
        self.device = device
        self.optimizer = build_optimizer(
            model, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
        )

        # Mixed precision
        self.use_amp = train_cfg.precision in ("fp16", "bf16")
        if train_cfg.precision == "bf16":
            self.amp_dtype = torch.bfloat16
        elif train_cfg.precision == "fp16":
            self.amp_dtype = torch.float16
        else:
            self.amp_dtype = torch.float32

        self.scaler = (
            torch.amp.GradScaler("cuda")
            if train_cfg.precision == "fp16" and device == "cuda"
            else None
        )

        self.step = 0
        self.metrics_history: List[Dict[str, float]] = []

    def train_step(self, input_ids: torch.Tensor) -> Dict[str, float]:
        """Single training step with gradient accumulation."""
        self.model.train()
        total_loss_accum = 0.0
        ntp_accum = 0.0
        mtp_accum = 0.0

        for micro_step in range(self.cfg.grad_accum_steps):
            with torch.amp.autocast(
                device_type=self.device.split(":")[0],
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                logits, mtp_logits = self.model(
                    input_ids, return_mtp=True
                )

                # Collect router entropy losses from MoE layers
                router_losses = []
                for layer in self.model.layers:
                    with torch.amp.autocast(
                        device_type=self.device.split(":")[0],
                        dtype=self.amp_dtype,
                        enabled=self.use_amp,
                    ):
                        rl = layer.ffn.router_entropy_loss(
                            layer.norm3(
                                torch.zeros(
                                    1, 1, self.model.cfg.d_model,
                                    device=input_ids.device,
                                )
                            )
                        )
                    router_losses.append(rl)

                losses = titan_loss(
                    logits=logits,
                    targets=input_ids,
                    mtp_logits=mtp_logits,
                    mtp_weight=self.cfg.mtp_weight,
                    router_entropy_losses=router_losses,
                    router_weight=self.cfg.router_weight,
                )

                loss = losses["total"] / self.cfg.grad_accum_steps

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss_accum += losses["total"].item()
            ntp_accum += losses["ntp"].item()
            mtp_accum += losses["mtp"].item()

        # Gradient clipping + optimizer step
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.grad_clip
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.grad_clip
            )
            self.optimizer.step()

        self.optimizer.zero_grad(set_to_none=True)

        # LR schedule
        cosine_warmup_schedule(
            self.optimizer,
            self.step,
            self.cfg.warmup_steps,
            self.cfg.total_steps,
        )

        self.step += 1
        avg_factor = self.cfg.grad_accum_steps
        metrics = {
            "step": self.step,
            "loss": total_loss_accum / avg_factor,
            "ntp_loss": ntp_accum / avg_factor,
            "mtp_loss": mtp_accum / avg_factor,
            "lr": self.optimizer.param_groups[-1]["lr"],
        }
        self.metrics_history.append(metrics)
        return metrics

    def save_checkpoint(self, path: str) -> None:
        """Save model + optimizer state."""
        torch.save(
            {
                "step": self.step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.model.cfg,
                "metrics_history": self.metrics_history,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        """Load model + optimizer state."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.step = ckpt["step"]
        self.metrics_history = ckpt.get("metrics_history", [])
