#!/usr/bin/env python3
"""Benchmark script for NEXUS-Ω performance profiling."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nexus.config import nexus_omega_base, nexus_omega_small
from nexus.inference.quantize import estimate_model_size
from nexus.model.nexus_model import NexusOmegaModel
from nexus.utils.profiler import (
    benchmark_throughput,
    count_parameters,
    get_memory_usage,
)


def main() -> None:
    print("=" * 60)
    print("NEXUS-Ω Benchmark Suite")
    print("=" * 60)

    # Small config for quick benchmarking
    print("\n--- Small Config ---")
    config = nexus_omega_small()
    model = NexusOmegaModel(config)
    model.eval()

    # Parameter count
    param_counts = count_parameters(model)
    print("\nParameter Counts:")
    for name, count in sorted(param_counts.items()):
        print(f"  {name}: {count:,}")

    # Memory usage
    mem = get_memory_usage(model)
    print("\nMemory Usage:")
    for key, val in mem.items():
        print(f"  {key}: {val:.2f} MB")

    # Size estimates
    sizes = estimate_model_size(model)
    print("\nSize Estimates:")
    for key, val in sizes.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.2f} MB")
        else:
            print(f"  {key}: {val:,}")

    # Throughput benchmark
    print("\nThroughput Benchmark (CPU):")
    throughput = benchmark_throughput(
        model, batch_size=1, seq_len=64, vocab_size=config.vocab_size,
        num_warmup=1, num_iterations=3, device="cpu",
    )
    for key, val in throughput.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.2f}")
        else:
            print(f"  {key}: {val}")

    # Base config parameter count check
    print("\n--- Base Config (parameter count only) ---")
    base_config = nexus_omega_base()
    base_model = NexusOmegaModel(base_config)
    total_base = sum(p.numel() for p in base_model.parameters())
    print(f"Total parameters: {total_base:,}")
    print(f"Under 200M: {'YES' if total_base < 200_000_000 else 'NO'}")

    print("\n" + "=" * 60)
    print("Benchmark complete!")


if __name__ == "__main__":
    main()
