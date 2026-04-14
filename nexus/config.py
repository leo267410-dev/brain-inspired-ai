"""Configuration system for NEXUS-Ω using dataclasses."""

from dataclasses import dataclass


@dataclass
class NexusOmegaConfig:
    """Master configuration for the NEXUS-Ω model."""

    # Model dimensions
    vocab_size: int = 48000
    hidden_dim: int = 768
    num_layers: int = 24  # total: 8 SSM + 8 hybrid + 8 full attention
    num_ssm_layers: int = 8
    num_hybrid_layers: int = 8
    num_attention_layers: int = 8
    num_heads: int = 12
    head_dim: int = 64  # hidden_dim // num_heads

    # MoE
    num_shared_experts: int = 32
    expert_dim: int = 512
    tie_word_embeddings: bool = True
    top_k_experts: int = 2
    moe_load_balance_weight: float = 0.01

    # Sequence
    max_seq_len: int = 8192
    local_window_size: int = 256
    global_landmark_interval: int = 64

    # Smart Neurons
    neuron_gate_threshold: float = 0.3
    neuron_prune_percentile: float = 5.0
    evolutionary_step_interval: int = 1000

    # Dynamic Depth/Width
    early_exit_threshold: float = 0.8
    dynamic_width_threshold: float = 0.3

    # Recursive Reasoning
    max_recursive_loops: int = 3
    recursive_start_layer: int = 18  # layers 18-24 (0-indexed: 16-23)
    scratchpad_dim: int = 256

    # Thought Engine
    max_thought_tokens: int = 1024
    thought_confidence_threshold: float = 0.85

    # External Memory
    memory_num_entries: int = 1_000_000
    memory_key_dim: int = 128
    memory_top_k: int = 32

    # Knowledge Embeddings
    num_knowledge_vectors: int = 4096

    # MCTS
    mcts_num_simulations: int = 200
    mcts_num_candidates: int = 8
    mcts_exploration_weight: float = 1.414

    # TTT (Test-Time Training)
    ttt_lora_rank: int = 16
    ttt_num_steps: int = 10
    ttt_lr: float = 1e-4

    # Self-Verification
    max_revisions: int = 3
    verification_confidence_threshold: float = 0.3

    # SSM (Mamba)
    ssm_state_dim: int = 16
    ssm_conv_dim: int = 4
    ssm_expand_factor: int = 1

    # Training
    dropout: float = 0.1
    use_bfloat16: bool = True

    # Lateral connections
    lateral_connectivity: float = 0.1  # 10% sparse connectivity
    lateral_lr: float = 0.01
    lateral_alpha: float = 0.01
    lateral_decay: float = 0.001


def nexus_omega_base() -> NexusOmegaConfig:
    """Default base configuration for NEXUS-Ω."""
    return NexusOmegaConfig()


def nexus_omega_small() -> NexusOmegaConfig:
    """Smaller config for testing."""
    return NexusOmegaConfig(
        hidden_dim=256,
        num_layers=6,
        num_ssm_layers=2,
        num_hybrid_layers=2,
        num_attention_layers=2,
        num_heads=4,
        head_dim=64,
        num_shared_experts=8,
        expert_dim=512,
        max_seq_len=512,
        num_knowledge_vectors=256,
        memory_num_entries=1000,
        max_thought_tokens=64,
    )


def nexus_omega_code() -> NexusOmegaConfig:
    """Code-optimized config."""
    return NexusOmegaConfig(
        mcts_num_simulations=400,
        max_thought_tokens=1024,
        max_recursive_loops=5,
        neuron_gate_threshold=0.25,
        tie_word_embeddings=True,
    )


def nexus_omega_language() -> NexusOmegaConfig:
    """Language-optimized config."""
    return NexusOmegaConfig(
        mcts_num_simulations=50,
        max_thought_tokens=512,
        max_recursive_loops=2,
        neuron_gate_threshold=0.35,
        tie_word_embeddings=True,
    )
