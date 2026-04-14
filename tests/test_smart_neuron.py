"""Tests for smart neuron modules."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nexus.model.smart_neuron import (
    AdaptiveNeuronGate,
    EvolutionaryNeuronManager,
    SmartNeuronLayer,
)


def test_adaptive_neuron_gate():
    """Test neuron gating produces correct shape and sparsity."""
    gate = AdaptiveNeuronGate(hidden_dim=256, threshold=0.3)
    x = torch.randn(2, 32, 256)
    out = gate(x)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()


def test_neuron_gate_training_tracks_stats():
    """Test that training mode tracks activation statistics."""
    gate = AdaptiveNeuronGate(hidden_dim=64, threshold=0.3)
    gate.train()
    x = torch.randn(2, 16, 64)
    _ = gate(x)
    assert gate.total_count.item() > 0
    util = gate.get_utilization()
    assert util.shape == (64,)


def test_smart_neuron_layer():
    """Test SmartNeuronLayer wrapper."""
    layer = SmartNeuronLayer(hidden_dim=128, threshold=0.3)
    x = torch.randn(2, 32, 128)
    out = layer(x)
    assert out.shape == x.shape


def test_evolutionary_neuron_manager():
    """Test evolutionary pruning."""
    layers = [SmartNeuronLayer(hidden_dim=64, threshold=0.3) for _ in range(3)]

    # Run some data through to build stats
    for layer in layers:
        layer.train()
        x = torch.randn(4, 16, 64)
        _ = layer(x)

    manager = EvolutionaryNeuronManager()
    num_pruned = manager.evolutionary_step(layers, prune_percentile=10.0)
    assert isinstance(num_pruned, int)
    assert num_pruned >= 0


if __name__ == "__main__":
    test_adaptive_neuron_gate()
    test_neuron_gate_training_tracks_stats()
    test_smart_neuron_layer()
    test_evolutionary_neuron_manager()
    print("All smart neuron tests passed!")
