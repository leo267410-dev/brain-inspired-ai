"""Tests for SSM modules."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nexus.config import nexus_omega_small
from nexus.model.ssm import HybridSSMAttention, MambaSSMBlock


def test_mamba_ssm_block():
    """Test Mamba SSM block output shape and no NaNs."""
    block = MambaSSMBlock(hidden_dim=256, state_dim=16, conv_dim=4, expand_factor=2)
    B, L, D = 2, 64, 256
    x = torch.randn(B, L, D)
    out = block(x)
    assert out.shape == (B, L, D), f"Expected {(B, L, D)}, got {out.shape}"
    assert not torch.isnan(out).any(), "NaN in output"


def test_mamba_ssm_block_short_seq():
    """Test with very short sequence."""
    block = MambaSSMBlock(hidden_dim=128, state_dim=8, conv_dim=4, expand_factor=2)
    x = torch.randn(1, 4, 128)
    out = block(x)
    assert out.shape == (1, 4, 128)
    assert not torch.isnan(out).any()


def test_hybrid_ssm_attention():
    """Test hybrid SSM + attention block."""
    config = nexus_omega_small()
    block = HybridSSMAttention(config)
    B, L = 2, 64
    x = torch.randn(B, L, config.hidden_dim)
    out = block(x)
    assert out.shape == (B, L, config.hidden_dim)
    assert not torch.isnan(out).any(), "NaN in hybrid output"


def test_mamba_gradient_flow():
    """Test that gradients flow through the SSM block."""
    block = MambaSSMBlock(hidden_dim=64, state_dim=8, conv_dim=4, expand_factor=2)
    x = torch.randn(1, 16, 64, requires_grad=True)
    out = block(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "No gradient on input"
    assert not torch.isnan(x.grad).any(), "NaN in gradients"


if __name__ == "__main__":
    test_mamba_ssm_block()
    test_mamba_ssm_block_short_seq()
    test_hybrid_ssm_attention()
    test_mamba_gradient_flow()
    print("All SSM tests passed!")
