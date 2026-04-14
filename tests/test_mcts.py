"""Tests for MCTS code search."""

import sys
from pathlib import Path
from typing import Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nexus.config import nexus_omega_small
from nexus.search.mcts import MCTSCodeSearch, MCTSNode


def test_mcts_node_ucb():
    """Test UCB score computation."""
    parent = MCTSNode(state=torch.randn(64), visit_count=10)
    child = MCTSNode(state=torch.randn(64), parent=parent, visit_count=5, total_value=3.0, prior=0.5)
    score = child.ucb_score(exploration_weight=1.414)
    assert isinstance(score, float)
    assert score > 0


def test_mcts_node_unvisited():
    """Test unvisited node has infinite UCB."""
    node = MCTSNode(state=torch.randn(64), visit_count=0, prior=0.5)
    assert node.ucb_score() == float("inf")


def test_mcts_search():
    """Test MCTS search returns valid results."""
    config = nexus_omega_small()
    mcts = MCTSCodeSearch(config)

    # Mock policy: returns random logits
    def policy_fn(state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = state.shape[0] if state.dim() > 1 else 1
        logits = torch.randn(B, 1, config.vocab_size)
        values = torch.randn(B, 1)
        return logits, values

    # Mock value function
    def value_fn(state: torch.Tensor) -> float:
        return torch.randn(1).item()

    root_state = torch.randn(32, config.hidden_dim)
    results = mcts.search(root_state, policy_fn, value_fn, num_simulations=10)

    assert isinstance(results, list)
    if results:
        token_id, visit_frac = results[0]
        assert isinstance(token_id, int)
        assert 0 <= visit_frac <= 1


def test_mcts_expand():
    """Test MCTS expansion creates children."""
    config = nexus_omega_small()
    mcts = MCTSCodeSearch(config)

    def policy_fn(state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = state.shape[0] if state.dim() > 1 else 1
        return torch.randn(B, 1, config.vocab_size), torch.randn(B, 1)

    root = MCTSNode(state=torch.randn(16, config.hidden_dim))
    children = mcts.expand(root, policy_fn)
    assert len(children) == config.mcts_num_candidates
    assert len(root.children) == config.mcts_num_candidates


if __name__ == "__main__":
    test_mcts_node_ucb()
    test_mcts_node_unvisited()
    test_mcts_search()
    test_mcts_expand()
    print("All MCTS tests passed!")
