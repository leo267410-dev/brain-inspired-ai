#!/usr/bin/env python3
"""Demo script showing NEXUS-Ω model instantiation and forward pass."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nexus.config import NexusOmegaConfig, nexus_omega_small
from nexus.model.nexus_model import NexusOmegaModel
from nexus.utils.profiler import count_parameters, get_memory_usage


def main() -> None:
    print("=" * 60)
    print("NEXUS-Ω (Omega) Demo")
    print("A sub-200M parameter language model with radical innovations")
    print("=" * 60)

    # Use small config for demo
    config = nexus_omega_small()
    print(f"\nConfig: hidden_dim={config.hidden_dim}, layers={config.num_layers}")
    print(f"  SSM layers: {config.num_ssm_layers}")
    print(f"  Hybrid layers: {config.num_hybrid_layers}")
    print(f"  Attention layers: {config.num_attention_layers}")
    print(f"  Shared experts: {config.num_shared_experts}")
    print(f"  Max seq len: {config.max_seq_len}")

    # Create model
    print("\nInstantiating model...")
    model = NexusOmegaModel(config)
    model.eval()

    # Parameter count
    params = count_parameters(model)
    total = params.pop("_total", 0)
    print(f"\nTotal parameters: {total:,}")
    print("Breakdown:")
    for name, count in sorted(params.items(), key=lambda x: -x[1]):
        pct = 100 * count / total if total > 0 else 0
        print(f"  {name}: {count:,} ({pct:.1f}%)")

    # Memory
    mem = get_memory_usage(model)
    print(f"\nModel memory: {mem['total_memory_mb']:.1f} MB")

    # Forward pass with random data
    print("\nRunning forward pass...")
    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        outputs = model(input_ids)

    logits = outputs["logits"]
    print(f"  Input shape:  ({batch_size}, {seq_len})")
    print(f"  Output shape: {tuple(logits.shape)}")
    print(f"  Hidden states shape: {tuple(outputs['hidden_states'].shape)}")
    print(f"  MoE loss: {outputs['moe_loss'].item():.4f}")

    # Verify output is valid
    assert logits.shape == (batch_size, seq_len, config.vocab_size), \
        f"Unexpected output shape: {logits.shape}"
    assert not torch.isnan(logits).any(), "NaN in output!"
    assert not torch.isinf(logits).any(), "Inf in output!"

    print("\n✓ Forward pass successful — model is fully functional!")

    # Also show base config param count
    print("\n--- Base config parameter count ---")
    base_config = NexusOmegaConfig()
    base_model = NexusOmegaModel(base_config)
    base_total = sum(p.numel() for p in base_model.parameters())
    print(f"Base model parameters: {base_total:,}")
    print(f"Under 200M: {'YES' if base_total < 200_000_000 else 'NO'}")

    print("\n" + "=" * 60)
    print("Demo complete!")


if __name__ == "__main__":
    main()
