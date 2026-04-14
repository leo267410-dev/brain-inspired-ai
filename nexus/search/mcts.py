"""Monte Carlo Tree Search for code generation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from nexus.config import NexusOmegaConfig


@dataclass
class MCTSNode:
    """A node in the MCTS search tree."""

    state: torch.Tensor  # (hidden_dim,) or (seq_len, hidden_dim) hidden state
    token_id: int = -1  # token that led to this node
    parent: Optional["MCTSNode"] = None
    children: List["MCTSNode"] = field(default_factory=list)
    visit_count: int = 0
    total_value: float = 0.0
    prior: float = 0.0

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def ucb_score(self, exploration_weight: float = 1.414) -> float:
        """Upper Confidence Bound score."""
        if self.visit_count == 0:
            return float("inf")
        parent_visits = self.parent.visit_count if self.parent else 1
        exploitation = self.value
        exploration = exploration_weight * self.prior * math.sqrt(
            math.log(parent_visits) / self.visit_count
        )
        return exploitation + exploration


class MCTSCodeSearch:
    """
    Monte Carlo Tree Search for exploring code generation paths.
    Uses a value function and policy to guide search.
    """

    def __init__(self, config: NexusOmegaConfig):
        self.num_simulations = config.mcts_num_simulations
        self.num_candidates = config.mcts_num_candidates
        self.exploration_weight = config.mcts_exploration_weight

    def select(self, node: MCTSNode) -> MCTSNode:
        """Select the best child using UCB."""
        while node.children:
            node = max(
                node.children,
                key=lambda c: c.ucb_score(self.exploration_weight),
            )
        return node

    def expand(
        self,
        node: MCTSNode,
        policy_fn: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    ) -> List[MCTSNode]:
        """
        Expand a leaf node by generating candidate children.

        Args:
            node: leaf node to expand.
            policy_fn: function that takes state and returns (logits, values).

        Returns:
            List of new child nodes.
        """
        logits, _ = policy_fn(node.state.unsqueeze(0))
        probs = F.softmax(logits[0, -1], dim=-1)  # last token's distribution

        # Take top-k candidates
        top_probs, top_indices = torch.topk(probs, self.num_candidates)

        children = []
        for prob, idx in zip(top_probs, top_indices):
            child = MCTSNode(
                state=node.state,  # Will be updated during simulation
                token_id=idx.item(),
                parent=node,
                prior=prob.item(),
            )
            node.children.append(child)
            children.append(child)

        return children

    def simulate(
        self,
        node: MCTSNode,
        value_fn: Callable[[torch.Tensor], float],
    ) -> float:
        """
        Simulate from a node to estimate its value.

        Args:
            node: node to simulate from.
            value_fn: function that estimates the value of a state.

        Returns:
            Estimated value.
        """
        return value_fn(node.state.unsqueeze(0))

    def backpropagate(self, node: MCTSNode, value: float) -> None:
        """Backpropagate the value up the tree."""
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            node = node.parent

    def search(
        self,
        root_state: torch.Tensor,
        policy_fn: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
        value_fn: Callable[[torch.Tensor], float],
        num_simulations: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """
        Run MCTS search from a root state.

        Args:
            root_state: (seq_len, hidden_dim) or (hidden_dim,) initial state.
            policy_fn: policy function.
            value_fn: value function.
            num_simulations: override number of simulations.

        Returns:
            List of (token_id, visit_fraction) sorted by visits.
        """
        if num_simulations is None:
            num_simulations = self.num_simulations

        root = MCTSNode(state=root_state)

        for _ in range(num_simulations):
            # Select
            leaf = self.select(root)

            # Expand (if not already expanded)
            if not leaf.children:
                self.expand(leaf, policy_fn)

            # Simulate
            if leaf.children:
                sim_node = leaf.children[0]  # simulate from first child
            else:
                sim_node = leaf
            value = self.simulate(sim_node, value_fn)

            # Backpropagate
            self.backpropagate(sim_node, value)

        # Return action distribution based on visit counts
        if not root.children:
            return []

        total_visits = sum(c.visit_count for c in root.children)
        results = [
            (c.token_id, c.visit_count / max(total_visits, 1))
            for c in root.children
        ]
        return sorted(results, key=lambda x: x[1], reverse=True)
