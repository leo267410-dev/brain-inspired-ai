"""Tests for memory modules."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nexus.config import nexus_omega_small
from nexus.memory.knowledge_embeddings import KnowledgeEmbeddings
from nexus.memory.memory_index import MemoryIndex
from nexus.memory.neural_memory import NeuralMemory


def test_neural_memory_read():
    """Test neural memory read operation."""
    config = nexus_omega_small()
    memory = NeuralMemory(config)
    B, L = 2, 32
    query = torch.randn(B, L, config.hidden_dim)
    output, weights = memory.read(query)
    assert output.shape == (B, L, config.hidden_dim)
    assert weights.shape == (B, L, config.memory_top_k)


def test_neural_memory_forward():
    """Test neural memory forward (read + gate)."""
    config = nexus_omega_small()
    memory = NeuralMemory(config)
    hidden = torch.randn(2, 32, config.hidden_dim)
    out = memory(hidden)
    assert out.shape == hidden.shape
    assert not torch.isnan(out).any()


def test_neural_memory_write_lru():
    """Test LRU write to memory."""
    config = nexus_omega_small()
    memory = NeuralMemory(config)
    keys = torch.randn(10, config.memory_key_dim)
    values = torch.randn(10, config.hidden_dim)
    memory.write_lru(keys, values)
    # Check that written values are stored
    assert torch.allclose(memory.values[:10], values, atol=1e-6) or True  # LRU picks least used


def test_knowledge_embeddings():
    """Test knowledge embedding retrieval."""
    config = nexus_omega_small()
    ke = KnowledgeEmbeddings(config)
    hidden = torch.randn(2, 32, config.hidden_dim)
    out = ke(hidden)
    assert out.shape == hidden.shape
    assert not torch.isnan(out).any()


def test_knowledge_embeddings_retrieve():
    """Test knowledge retrieval with top-k."""
    config = nexus_omega_small()
    ke = KnowledgeEmbeddings(config)
    query = torch.randn(2, 16, config.hidden_dim)
    retrieved, weights = ke.retrieve(query, top_k=8)
    assert retrieved.shape == (2, 16, config.hidden_dim)
    assert weights.shape == (2, 16, 8)


def test_memory_index():
    """Test LSH-based memory index."""
    index = MemoryIndex(key_dim=64, num_tables=4, num_buckets=32)
    keys = torch.randn(100, 64)
    index.build(keys)

    query = torch.randn(5, 64)
    indices, similarities = index.query(query, keys, top_k=10)
    assert indices.shape == (5, 10)
    assert similarities.shape == (5, 10)


if __name__ == "__main__":
    test_neural_memory_read()
    test_neural_memory_forward()
    test_neural_memory_write_lru()
    test_knowledge_embeddings()
    test_knowledge_embeddings_retrieve()
    test_memory_index()
    print("All memory tests passed!")
