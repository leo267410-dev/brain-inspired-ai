"""Main training loop for NEXUS-Ω."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    pass


@dataclass
class TrainingArgs:
    """Training hyperparameters."""

    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_epochs: int = 100
    warmup_steps: int = 2000
    max_steps: int = 500_000
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    save_interval: int = 5000
    eval_interval: int = 1000
    log_interval: int = 100
    output_dir: str = "checkpoints"
    fp16: bool = False
    bf16: bool = True
    seed: int = 42


class NexusTrainer:
    """
    Trainer for NEXUS-Ω with curriculum learning, distillation,
    and Hebbian lateral updates.
    """

    def __init__(
        self,
        model: nn.Module,
        args: TrainingArgs,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[object] = None,
    ):
        self.model = model
        self.args = args
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.global_step = 0
        self.epoch = 0

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Setup optimizer
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = self._create_optimizer()

        self.scheduler = scheduler

        # Setup mixed precision
        self.scaler = torch.amp.GradScaler("cuda") if args.fp16 else None
        self.autocast_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)

        # Metrics tracking
        self.train_losses: list[float] = []

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create AdamW optimizer with weight decay."""
        # Separate params that should/shouldn't have weight decay
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name or "embedding" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        return torch.optim.AdamW([
            {"params": decay_params, "weight_decay": self.args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ], lr=self.args.learning_rate)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Execute a single training step."""
        self.model.train()

        input_ids = batch["input_ids"].to(self.device)
        labels = batch.get("labels", input_ids[:, 1:]).to(self.device)

        amp_device = "cuda" if self.device.type == "cuda" else "cpu"
        use_amp = self.autocast_dtype != torch.float32 and self.device.type == "cuda"

        with torch.amp.autocast(amp_device, dtype=self.autocast_dtype, enabled=use_amp):
            outputs = self.model(input_ids)

            if isinstance(outputs, dict):
                logits = outputs["logits"]
                # Shift logits and labels for next-token prediction
                shift_logits = logits[:, :-1].contiguous()
                shift_labels = labels[:, :shift_logits.size(1)].contiguous()
                loss = nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
                # Add MoE load balancing loss
                moe_loss = outputs.get("moe_loss", torch.tensor(0.0, device=self.device))
                loss = loss + 0.01 * moe_loss
            else:
                logits = outputs[:, :-1].contiguous()
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels[:, :logits.size(1)].contiguous().view(-1),
                    ignore_index=-100,
                )

        loss = loss / self.args.gradient_accumulation_steps

        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss.item() * self.args.gradient_accumulation_steps

    def train(self) -> Dict[str, float]:
        """Run the full training loop."""
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        start_time = time.time()
        accumulated_loss = 0.0

        for epoch in range(self.args.max_epochs):
            self.epoch = epoch

            for step, batch in enumerate(self.train_dataloader):
                loss = self.train_step(batch)
                accumulated_loss += loss

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.args.max_grad_norm,
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.args.max_grad_norm,
                        )
                        self.optimizer.step()

                    if self.scheduler is not None:
                        self.scheduler.step()

                    self.optimizer.zero_grad()
                    self.global_step += 1

                    avg_loss = accumulated_loss / self.args.gradient_accumulation_steps
                    self.train_losses.append(avg_loss)
                    accumulated_loss = 0.0

                    if self.global_step % self.args.log_interval == 0:
                        elapsed = time.time() - start_time
                        print(
                            f"Step {self.global_step} | Loss: {avg_loss:.4f} | "
                            f"Time: {elapsed:.1f}s",
                        )

                    if self.global_step % self.args.save_interval == 0:
                        self.save_checkpoint()

                    if self.global_step >= self.args.max_steps:
                        break

            if self.global_step >= self.args.max_steps:
                break

        self.save_checkpoint()
        return {"final_loss": self.train_losses[-1] if self.train_losses else 0.0}

    def save_checkpoint(self, path: Optional[str] = None) -> None:
        """Save model checkpoint."""
        if path is None:
            path = f"{self.args.output_dir}/checkpoint-{self.global_step}"

        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "train_losses": self.train_losses,
        }, f"{path}/model.pt")
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(f"{path}/model.pt", map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]
        self.train_losses = checkpoint.get("train_losses", [])
        print(f"Loaded checkpoint from {path} (step {self.global_step})")
