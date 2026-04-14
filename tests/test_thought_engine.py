"""Tests for thought engine modules."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nexus.config import nexus_omega_small
from nexus.thought.compressor import ThoughtCompressor
from nexus.thought.confidence import ConfidenceEstimator
from nexus.thought.engine import ThoughtEngine


def test_confidence_estimator():
    """Test confidence estimator output range."""
    estimator = ConfidenceEstimator(hidden_dim=256)
    x = torch.randn(4, 32, 256)
    conf = estimator(x)
    assert conf.shape == (4, 32), f"Expected (4, 32), got {conf.shape}"
    assert (conf >= 0).all() and (conf <= 1).all(), "Confidence out of [0, 1]"


def test_confidence_calibration():
    """Test calibration tracking."""
    estimator = ConfidenceEstimator(hidden_dim=64)
    predicted = torch.tensor([0.1, 0.3, 0.7, 0.9])
    correct = torch.tensor([0.0, 0.0, 1.0, 1.0])
    estimator.update_calibration(predicted, correct)
    ece = estimator.get_calibration_error()
    assert isinstance(ece, float)
    assert ece >= 0


def test_thought_compressor():
    """Test thought compressor reduces sequence length."""
    compressor = ThoughtCompressor(hidden_dim=256, compressed_len=8)
    thoughts = torch.randn(2, 64, 256)
    compressed = compressor(thoughts)
    assert compressed.shape == (2, 8, 256), f"Expected (2, 8, 256), got {compressed.shape}"
    assert not torch.isnan(compressed).any()


def test_thought_engine():
    """Test thought engine generates and compresses thoughts."""
    config = nexus_omega_small()
    engine = ThoughtEngine(config)
    engine.eval()
    B, L = 2, 32
    hidden = torch.randn(B, L, config.hidden_dim)

    with torch.no_grad():
        conditioned, summary = engine(hidden)

    assert conditioned.shape == (B, L, config.hidden_dim)
    assert not torch.isnan(conditioned).any()
    assert summary.shape[0] == B
    assert summary.shape[2] == config.hidden_dim


def test_thought_engine_training():
    """Test thought engine in training mode."""
    config = nexus_omega_small()
    engine = ThoughtEngine(config)
    engine.train()
    hidden = torch.randn(1, 16, config.hidden_dim, requires_grad=True)
    conditioned, summary = engine(hidden)
    loss = conditioned.sum()
    loss.backward()
    assert hidden.grad is not None


if __name__ == "__main__":
    test_confidence_estimator()
    test_confidence_calibration()
    test_thought_compressor()
    test_thought_engine()
    test_thought_engine_training()
    print("All thought engine tests passed!")
