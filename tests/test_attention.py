"""Tests for attention modules."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nexus.config import nexus_omega_small
from nexus.model.attention import (
    GlobalLandmarkAttention,
    HierarchicalSparseAttention,
    LocalWindowAttention,
)


def test_local_window_attention():
    """Test local window attention produces correct output shape."""
    B, L, H, D = 2, 128, 4, 64
    lwa = LocalWindowAttention(window_size=32)
    q = torch.randn(B, L, H, D)
    k = torch.randn(B, L, H, D)
    v = torch.randn(B, L, H, D)
    out = lwa(q, k, v)
    assert out.shape == (B, L, H, D), f"Expected {(B, L, H, D)}, got {out.shape}"
    assert not torch.isnan(out).any(), "NaN in output"


def test_local_window_attention_non_divisible():
    """Test with sequence length not divisible by window size."""
    B, L, H, D = 2, 100, 4, 64
    lwa = LocalWindowAttention(window_size=32)
    q = torch.randn(B, L, H, D)
    k = torch.randn(B, L, H, D)
    v = torch.randn(B, L, H, D)
    out = lwa(q, k, v)
    assert out.shape == (B, L, H, D)


def test_global_landmark_attention():
    """Test global landmark attention."""
    B, L, H, D = 2, 256, 4, 64
    gla = GlobalLandmarkAttention(landmark_interval=64)
    q = torch.randn(B, L, H, D)
    k = torch.randn(B, L, H, D)
    v = torch.randn(B, L, H, D)
    out = gla(q, k, v)
    assert out.shape == (B, L, H, D), f"Expected {(B, L, H, D)}, got {out.shape}"
    assert not torch.isnan(out).any(), "NaN in output"


def test_hierarchical_sparse_attention():
    """Test full hierarchical sparse attention."""
    config = nexus_omega_small()
    hsa = HierarchicalSparseAttention(config)
    B, L = 2, 128
    x = torch.randn(B, L, config.hidden_dim)
    out = hsa(x)
    assert out.shape == (B, L, config.hidden_dim), f"Unexpected shape: {out.shape}"
    assert not torch.isnan(out).any(), "NaN in output"


def test_hierarchical_sparse_attention_with_mask():
    """Test attention with a mask."""
    config = nexus_omega_small()
    hsa = HierarchicalSparseAttention(config)
    B, L = 2, 64
    x = torch.randn(B, L, config.hidden_dim)
    mask = torch.ones(B, L, dtype=torch.bool)
    mask[:, L // 2:] = False
    out = hsa(x, attention_mask=mask)
    assert out.shape == (B, L, config.hidden_dim)


if __name__ == "__main__":
    test_local_window_attention()
    test_local_window_attention_non_divisible()
    test_global_landmark_attention()
    test_hierarchical_sparse_attention()
    test_hierarchical_sparse_attention_with_mask()
    print("All attention tests passed!")
