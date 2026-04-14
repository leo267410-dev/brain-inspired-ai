"""Post-training quantization for efficient inference."""

from __future__ import annotations

import torch
import torch.nn as nn


class DynamicQuantizer:
    """
    Dynamic quantization wrapper for NEXUS-Ω.
    Supports INT8 and INT4 quantization of linear layers.
    """

    @staticmethod
    def quantize_dynamic(
        model: nn.Module, dtype: torch.dtype = torch.qint8,
    ) -> nn.Module:
        """
        Apply PyTorch dynamic quantization to linear layers.

        Args:
            model: model to quantize.
            dtype: quantization dtype (qint8 or quint8).

        Returns:
            Quantized model.
        """
        return torch.ao.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=dtype,
        )


class WeightOnlyQuantizer:
    """
    Weight-only quantization that keeps activations in full precision
    but stores weights in lower precision.
    """

    @staticmethod
    def quantize_weights_int8(module: nn.Linear) -> dict:
        """
        Quantize a linear layer's weights to INT8.

        Args:
            module: linear layer to quantize.

        Returns:
            Dict with quantized weight, scale, and zero_point.
        """
        weight = module.weight.data.float()
        w_min = weight.min()
        w_max = weight.max()

        # Symmetric quantization
        abs_max = max(abs(w_min.item()), abs(w_max.item()))
        scale = abs_max / 127.0
        zero_point = 0

        quantized_weight = torch.round(weight / scale).clamp(-128, 127).to(torch.int8)

        return {
            "quantized_weight": quantized_weight,
            "scale": scale,
            "zero_point": zero_point,
        }

    @staticmethod
    def dequantize_int8(quantized_weight: torch.Tensor, scale: float) -> torch.Tensor:
        """Dequantize INT8 weights back to float."""
        return quantized_weight.float() * scale


def estimate_model_size(model: nn.Module, bits: int = 32) -> dict:
    """
    Estimate model size at different quantization levels.

    Args:
        model: the model.
        bits: target bits per parameter.

    Returns:
        Dict with size estimates.
    """
    total_params = sum(p.numel() for p in model.parameters())
    size_fp32 = total_params * 4 / (1024 ** 2)  # MB
    size_fp16 = total_params * 2 / (1024 ** 2)
    size_int8 = total_params * 1 / (1024 ** 2)
    size_int4 = total_params * 0.5 / (1024 ** 2)

    return {
        "total_params": total_params,
        "size_fp32_mb": size_fp32,
        "size_fp16_mb": size_fp16,
        "size_int8_mb": size_int8,
        "size_int4_mb": size_int4,
    }
