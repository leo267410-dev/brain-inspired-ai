"""Main NEXUS-Ω model assembling all components."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from nexus.config import NexusOmegaConfig
from nexus.model.attention import HierarchicalSparseAttention
from nexus.model.early_exit import DynamicWidthGate, EarlyExitManager
from nexus.model.embeddings import MultiResolutionEmbedding
from nexus.model.lateral import CrossLayerLateral, HebbianLateralConnections
from nexus.model.meta_controller import MetaController
from nexus.model.moe_ffn import SharedExpertPool
from nexus.model.recursive_reasoning import RecursiveReasoningBlock
from nexus.model.smart_neuron import SmartNeuronLayer
from nexus.model.ssm import HybridSSMAttention, MambaSSMBlock
from nexus.model.task_heads import TaskHeadRouter


class NexusBlock(nn.Module):
    """
    A single NEXUS-Ω transformer block.
    Depending on the layer index, it uses:
    - Pure SSM (layers 0-7)
    - Hybrid SSM + Attention (layers 8-15)
    - Full attention (layers 16-23)

    Each block includes MoE FFN, smart neuron gating,
    lateral connections, and dynamic width gating.
    """

    def __init__(
        self,
        config: NexusOmegaConfig,
        layer_idx: int,
        shared_expert_pool: SharedExpertPool,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_dim = config.hidden_dim

        # Determine block type based on layer index
        if layer_idx < config.num_ssm_layers:
            self.block_type = "ssm"
            self.core = MambaSSMBlock(
                config.hidden_dim, config.ssm_state_dim,
                config.ssm_conv_dim, config.ssm_expand_factor,
            )
        elif layer_idx < config.num_ssm_layers + config.num_hybrid_layers:
            self.block_type = "hybrid"
            self.core = HybridSSMAttention(config)
        else:
            self.block_type = "attention"
            self.core = HierarchicalSparseAttention(config)

        # Pre-norm
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)

        # Shared MoE FFN
        self.shared_expert_pool = shared_expert_pool
        shared_expert_pool.register_router(
            layer_idx, config.hidden_dim,
            config.num_shared_experts, config.top_k_experts,
        )

        # Smart neuron gating
        self.smart_neuron = SmartNeuronLayer(config.hidden_dim, config.neuron_gate_threshold)

        # Lateral connections
        self.lateral = HebbianLateralConnections(
            config.hidden_dim, config.lateral_connectivity, config.lateral_alpha,
        )

        # Dynamic width
        self.width_gate = DynamicWidthGate(config.hidden_dim, config.dynamic_width_threshold)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the block.

        Args:
            x: (batch, seq_len, hidden_dim)
            attention_mask: optional attention mask
            kv_cache: optional KV cache

        Returns:
            Tuple of (output, moe_loss).
        """
        B, L, D = x.shape

        # Core block (SSM, hybrid, or attention)
        normed = self.norm1(x)
        if self.block_type == "ssm":
            core_out = self.core(normed)
        elif self.block_type == "hybrid":
            core_out = self.core(normed, attention_mask, kv_cache)
            core_out = core_out - normed  # Remove residual added inside hybrid block
        else:
            core_out = self.core(normed, attention_mask, kv_cache)

        x = x + self.dropout(core_out)

        # MoE FFN
        normed_ffn = self.norm2(x)
        flat = normed_ffn.view(B * L, D)
        moe_out, moe_loss = self.shared_expert_pool(flat, self.layer_idx)
        moe_out = moe_out.view(B, L, D)
        x = x + self.dropout(moe_out)

        # Smart neuron gating
        x = self.smart_neuron(x)

        # Lateral connections
        x = self.lateral(x)

        # Dynamic width gating
        x = self.width_gate(x)

        return x, moe_loss


class NexusOmegaModel(nn.Module):
    """
    NEXUS-Ω: A sub-200M parameter language model with:
    - Shared MoE across all layers (64 experts, top-2 routing)
    - SSM-Attention hybrid architecture
    - Thought engine for internal chain-of-thought
    - External neural memory
    - MCTS code search
    - Recursive reasoning with scratchpad
    - Smart neuron gating with evolutionary pruning
    - Hebbian lateral connections
    - Dynamic depth/width
    - Self-verification loop
    """

    def __init__(self, config: Optional[NexusOmegaConfig] = None):
        super().__init__()
        if config is None:
            config = NexusOmegaConfig()
        self.config = config

        # Embeddings
        self.embeddings = MultiResolutionEmbedding(
            config.vocab_size, config.hidden_dim, config.max_seq_len,
        )

        # Shared expert pool (one instance shared across all layers)
        self.shared_expert_pool = SharedExpertPool(config)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            NexusBlock(config, i, self.shared_expert_pool)
            for i in range(config.num_layers)
        ])

        # Cross-layer lateral connections
        self.cross_laterals = nn.ModuleList([
            CrossLayerLateral(config.hidden_dim, config.lateral_connectivity, config.lateral_alpha)
            for _ in range(config.num_layers - 1)
        ])

        # Early exit manager
        self.early_exit = EarlyExitManager(
            config.num_layers, config.hidden_dim, config.early_exit_threshold,
        )

        # Recursive reasoning (for deeper layers)
        self.recursive_reasoning = RecursiveReasoningBlock(config)

        # Meta controller
        self.meta_controller = MetaController(config)

        # Task heads
        self.task_heads = TaskHeadRouter(config)

        # Final layer norm
        self.final_norm = nn.LayerNorm(config.hidden_dim)

        # Initialize weights
        self.apply(self._init_weights)

        # Tie word embeddings to task head projections
        if config.tie_word_embeddings:
            self.task_heads.tie_weights(self.embeddings.token_embedding.weight)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights with small values."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        segment_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        task: str = "lm",
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the full model.

        Args:
            input_ids: (batch, seq_len) token IDs.
            segment_ids: (batch, seq_len) segment IDs.
            attention_mask: optional attention mask.
            task: task type for head selection ("lm" or "code").

        Returns:
            Dict with 'logits', 'loss' (if applicable), and auxiliary outputs.
        """
        B, L = input_ids.shape

        # Embeddings
        hidden = self.embeddings(input_ids, segment_ids)

        # Process through blocks
        total_moe_loss = torch.tensor(0.0, device=hidden.device)
        prev_hidden = hidden

        for i, block in enumerate(self.blocks):
            hidden, moe_loss = block(hidden, attention_mask)
            total_moe_loss = total_moe_loss + moe_loss

            # Cross-layer lateral connections
            if i > 0:
                hidden = self.cross_laterals[i - 1](hidden, prev_hidden)

            prev_hidden = hidden

        # Recursive reasoning on final hidden states
        hidden, num_loops = self.recursive_reasoning(hidden)

        # Final norm
        hidden = self.final_norm(hidden)

        # Task head
        logits = self.task_heads(hidden, task)

        result: Dict[str, torch.Tensor] = {
            "logits": logits,
            "hidden_states": hidden,
            "moe_loss": total_moe_loss,
        }

        return result

    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter counts by component."""
        from nexus.utils.profiler import count_parameters
        return count_parameters(self)
