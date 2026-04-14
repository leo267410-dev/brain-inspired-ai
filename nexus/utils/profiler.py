"""Model profiling utilities for NEXUS-Ω."""

from __future__ import annotations

import time
from typing import Dict

import torch
import torch.nn as nn


def count_parameters(model: nn.Module, trainable_only: bool = True) -> Dict[str, int]:
    """
    Count unique parameters in the model, broken down by component.

    Correctly handles shared/tied weights by tracking parameter identity
    so that shared expert pools and tied embeddings are not double-counted.

    Args:
        model: the model to profile.
        trainable_only: whether to count only trainable parameters.

    Returns:
        Dict mapping component names to parameter counts.
    """
    counts: Dict[str, int] = {}
    total = 0
    seen_ids: set = set()

    for name, param in model.named_parameters():
        if trainable_only and not param.requires_grad:
            continue

        # Skip already-counted parameters (shared/tied weights)
        param_id = param.data_ptr()
        if param_id in seen_ids:
            continue
        seen_ids.add(param_id)

        # Extract top-level module name
        parts = name.split(".")
        component = parts[0] if parts else "other"

        if component not in counts:
            counts[component] = 0
        counts[component] += param.numel()
        total += param.numel()

    counts["_total"] = total
    return counts


def estimate_flops(
    model: nn.Module, input_shape: tuple, device: str = "cpu",
) -> Dict[str, float]:
    """
    Estimate FLOPs for a forward pass (approximate).

    Args:
        model: the model.
        input_shape: (batch, seq_len) input shape.
        device: device to run on.

    Returns:
        Dict with FLOP estimates.
    """
    from torch.utils.flop_counter import FlopCounterMode

    x = torch.randint(0, 100, input_shape, device=device)

    flop_counter = FlopCounterMode(display=False)
    with flop_counter:
        model(x)

    total_flops = flop_counter.get_total_flops()
    return {
        "total_flops": total_flops,
        "gflops": total_flops / 1e9,
    }


def benchmark_throughput(
    model: nn.Module,
    batch_size: int = 1,
    seq_len: int = 512,
    vocab_size: int = 48000,
    num_warmup: int = 3,
    num_iterations: int = 10,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Benchmark model throughput.

    Args:
        model: the model.
        batch_size: batch size.
        seq_len: sequence length.
        vocab_size: vocabulary size.
        num_warmup: warmup iterations.
        num_iterations: timing iterations.
        device: device to run on.

    Returns:
        Dict with throughput metrics.
    """
    model = model.to(device)
    model.eval()

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            model(input_ids)

    # Benchmark
    if device == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iterations):
            model(input_ids)

    if device == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start_time
    avg_time = elapsed / num_iterations
    tokens_per_second = (batch_size * seq_len) / avg_time

    return {
        "avg_latency_ms": avg_time * 1000,
        "tokens_per_second": tokens_per_second,
        "batch_size": batch_size,
        "seq_len": seq_len,
    }


def get_memory_usage(model: nn.Module) -> Dict[str, float]:
    """
    Get memory usage of the model.

    Returns:
        Dict with memory stats in MB.
    """
    param_mem = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
    buffer_mem = sum(b.numel() * b.element_size() for b in model.buffers()) / (1024 ** 2)

    result = {
        "param_memory_mb": param_mem,
        "buffer_memory_mb": buffer_mem,
        "total_memory_mb": param_mem + buffer_mem,
    }

    if torch.cuda.is_available():
        result["cuda_allocated_mb"] = torch.cuda.memory_allocated() / (1024 ** 2)
        result["cuda_reserved_mb"] = torch.cuda.memory_reserved() / (1024 ** 2)

    return result
