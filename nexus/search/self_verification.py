"""Self-verification loop: generate, verify, revise."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Tuple

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from nexus.config import NexusOmegaConfig
    from nexus.search.verifier import CodeVerifier


class SelfVerificationLoop(nn.Module):
    """
    Iterative self-verification: the model generates output, verifies it,
    and revises if the confidence is below a threshold.
    """

    def __init__(self, config: NexusOmegaConfig):
        super().__init__()
        self.max_revisions = config.max_revisions
        self.confidence_threshold = config.verification_confidence_threshold
        self.hidden_dim = config.hidden_dim

        # Revision network: takes hidden + verification feedback → revised hidden
        self.revision_net = nn.Sequential(
            nn.Linear(config.hidden_dim + 1, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

        self.revision_norm = nn.LayerNorm(config.hidden_dim)

    def forward(
        self,
        hidden: torch.Tensor,
        verifier: CodeVerifier,
        generate_fn: Optional[Callable] = None,
    ) -> Tuple[torch.Tensor, int, float]:
        """
        Run self-verification loop.

        Args:
            hidden: (batch, seq_len, hidden_dim) initial hidden states.
            verifier: CodeVerifier module.
            generate_fn: optional function to regenerate from revised hidden.

        Returns:
            Tuple of (revised_hidden, num_revisions, final_confidence).
        """
        current_hidden = hidden
        num_revisions = 0
        final_confidence = 0.0

        for _ in range(self.max_revisions):
            # Verify current output
            verification = verifier(current_hidden)
            confidence = verification["quality_score"].mean().item()
            final_confidence = confidence

            if confidence > self.confidence_threshold:
                break

            # Create revision signal
            conf_tensor = verification["quality_score"].unsqueeze(-1).unsqueeze(-1).expand_as(
                current_hidden[:, :, :1],
            )  # (B, L, 1)

            revision_input = torch.cat([current_hidden, conf_tensor], dim=-1)
            revision = self.revision_net(revision_input)
            current_hidden = self.revision_norm(current_hidden + revision)

            num_revisions += 1

        return current_hidden, num_revisions, final_confidence


class TestTimeTrainer:
    """
    Test-Time Training (TTT): fine-tune a small set of parameters
    on the current input at inference time.
    Uses LoRA-style low-rank adaptations.
    """

    def __init__(self, config: NexusOmegaConfig):
        self.lora_rank = config.ttt_lora_rank
        self.num_steps = config.ttt_num_steps
        self.lr = config.ttt_lr

    def create_lora_params(
        self, model: nn.Module,
    ) -> list[Tuple[nn.Parameter, nn.Parameter]]:
        """
        Create LoRA adapters for attention layers.

        Args:
            model: the model to adapt.

        Returns:
            List of (lora_A, lora_B) parameter pairs.
        """
        lora_params = []
        for name, param in model.named_parameters():
            if "q_proj" in name or "v_proj" in name:
                if param.dim() == 2:
                    d_out, d_in = param.shape
                    lora_a = nn.Parameter(
                        torch.randn(d_out, self.lora_rank, device=param.device) * 0.01,
                    )
                    lora_b = nn.Parameter(
                        torch.zeros(self.lora_rank, d_in, device=param.device),
                    )
                    lora_params.append((lora_a, lora_b))
        return lora_params

    def adapt(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        loss_fn: Callable,
    ) -> None:
        """
        Perform test-time training steps.

        Args:
            model: the model to adapt (in-place).
            input_ids: (batch, seq_len) current input.
            loss_fn: function that computes loss from model output.
        """
        lora_params = self.create_lora_params(model)
        all_params = [p for pair in lora_params for p in pair]

        if not all_params:
            return

        optimizer = torch.optim.Adam(all_params, lr=self.lr)

        for _ in range(self.num_steps):
            optimizer.zero_grad()
            output = model(input_ids)
            loss = loss_fn(output)
            loss.backward()
            optimizer.step()
