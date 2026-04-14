"""Tests for MoE modules."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nexus.config import nexus_omega_small
from nexus.model.moe_ffn import Expert, Router, SharedExpertPool


def test_expert():
    """Test single expert forward pass."""
    expert = Expert(hidden_dim=256, expert_dim=512)
    x = torch.randn(10, 256)
    out = expert(x)
    assert out.shape == (10, 256), f"Expected (10, 256), got {out.shape}"
    assert not torch.isnan(out).any()


def test_router():
    """Test router produces valid routing weights."""
    router = Router(hidden_dim=256, num_experts=8, top_k=2)
    x = torch.randn(32, 256)
    weights, indices, lb_loss = router(x)
    assert weights.shape == (32, 2), f"Expected (32, 2), got {weights.shape}"
    assert indices.shape == (32, 2)
    assert (weights >= 0).all(), "Negative routing weights"
    assert torch.allclose(weights.sum(dim=-1), torch.ones(32), atol=1e-5)
    assert lb_loss.item() >= 0


def test_shared_expert_pool():
    """Test shared expert pool with registered routers."""
    config = nexus_omega_small()
    pool = SharedExpertPool(config)

    # Register routers for two layers
    pool.register_router(0, config.hidden_dim, config.num_shared_experts, config.top_k_experts)
    pool.register_router(1, config.hidden_dim, config.num_shared_experts, config.top_k_experts)

    x = torch.randn(16, config.hidden_dim)

    out0, loss0 = pool(x, layer_idx=0)
    assert out0.shape == x.shape
    assert not torch.isnan(out0).any()

    out1, loss1 = pool(x, layer_idx=1)
    assert out1.shape == x.shape


def test_shared_expert_pool_gradient():
    """Test gradient flow through shared expert pool."""
    config = nexus_omega_small()
    pool = SharedExpertPool(config)
    pool.register_router(0, config.hidden_dim, config.num_shared_experts, config.top_k_experts)

    x = torch.randn(8, config.hidden_dim, requires_grad=True)
    out, loss = pool(x, layer_idx=0)
    total = out.sum() + loss
    total.backward()
    assert x.grad is not None


if __name__ == "__main__":
    test_expert()
    test_router()
    test_shared_expert_pool()
    test_shared_expert_pool_gradient()
    print("All MoE tests passed!")
