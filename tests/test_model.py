"""Tests for the full NEXUS-Ω model."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nexus.config import nexus_omega_base, nexus_omega_small
from nexus.model.nexus_model import NexusOmegaModel


def test_model_instantiation():
    """Test model can be instantiated with default config."""
    config = nexus_omega_small()
    model = NexusOmegaModel(config)
    assert model is not None


def test_model_forward_pass():
    """Test forward pass produces correct output shapes."""
    config = nexus_omega_small()
    model = NexusOmegaModel(config)
    model.eval()

    B, L = 2, 64
    input_ids = torch.randint(0, config.vocab_size, (B, L))

    with torch.no_grad():
        outputs = model(input_ids)

    assert "logits" in outputs
    assert "hidden_states" in outputs
    assert "moe_loss" in outputs
    assert outputs["logits"].shape == (B, L, config.vocab_size)
    assert outputs["hidden_states"].shape == (B, L, config.hidden_dim)


def test_model_no_nan():
    """Test model output has no NaN values."""
    config = nexus_omega_small()
    model = NexusOmegaModel(config)
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (1, 32))
    with torch.no_grad():
        outputs = model(input_ids)

    assert not torch.isnan(outputs["logits"]).any(), "NaN in logits"
    assert not torch.isnan(outputs["hidden_states"]).any(), "NaN in hidden states"


def test_model_gradient_flow():
    """Test that gradients flow through the entire model."""
    config = nexus_omega_small()
    model = NexusOmegaModel(config)
    model.train()

    input_ids = torch.randint(0, config.vocab_size, (1, 32))
    outputs = model(input_ids)
    loss = outputs["logits"].sum() + outputs["moe_loss"]
    loss.backward()

    # Check that at least some parameters have gradients
    has_grad = False
    for param in model.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break
    assert has_grad, "No gradients found in model parameters"


def test_model_with_segment_ids():
    """Test model with segment IDs."""
    config = nexus_omega_small()
    model = NexusOmegaModel(config)
    model.eval()

    B, L = 2, 64
    input_ids = torch.randint(0, config.vocab_size, (B, L))
    segment_ids = torch.randint(0, 3, (B, L))

    with torch.no_grad():
        outputs = model(input_ids, segment_ids=segment_ids)

    assert outputs["logits"].shape == (B, L, config.vocab_size)


def test_model_code_task():
    """Test model with code task head."""
    config = nexus_omega_small()
    model = NexusOmegaModel(config)
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (1, 32))
    with torch.no_grad():
        outputs = model(input_ids, task="code")

    assert outputs["logits"].shape == (1, 32, config.vocab_size)


def test_parameter_count_under_200m():
    """Verify base model has < 200M parameters."""
    config = nexus_omega_base()
    model = NexusOmegaModel(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Base model parameters: {total_params:,}")
    assert total_params < 200_000_000, f"Model has {total_params:,} params, exceeds 200M"


def test_model_deterministic():
    """Test model produces deterministic output with same input."""
    config = nexus_omega_small()
    model = NexusOmegaModel(config)
    model.eval()

    torch.manual_seed(42)
    input_ids = torch.randint(0, config.vocab_size, (1, 32))

    with torch.no_grad():
        out1 = model(input_ids)["logits"]
        out2 = model(input_ids)["logits"]

    assert torch.allclose(out1, out2, atol=1e-5), "Model is not deterministic"


if __name__ == "__main__":
    test_model_instantiation()
    test_model_forward_pass()
    test_model_no_nan()
    test_model_gradient_flow()
    test_model_with_segment_ids()
    test_model_code_task()
    test_parameter_count_under_200m()
    test_model_deterministic()
    print("All model tests passed!")
