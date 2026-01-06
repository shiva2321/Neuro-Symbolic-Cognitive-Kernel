"""
Graph Traversal Engine
Implements various traversal algorithms for text generation and graph exploration.
"""

import numpy as np
from typing import List, Dict, Optional, Callable, Set, Tuple
from dataclasses import dataclass
import random
from enum import Enum

from core.graph_network import GraphNetwork, Node


class TraversalStrategy(Enum):
    """Different traversal strategies"""
    GREEDY = "greedy"  # Always pick highest weight
    PROBABILISTIC = "probabilistic"  # Sample based on weights
    BEAM_SEARCH = "beam_search"  # Keep top-k paths
    TEMPERATURE = "temperature"  # Temperature-based sampling


class StopCriterion(Enum):
    """Stop criteria for traversal"""
    MAX_TOKENS = "max_tokens"
    PROBABILITY_THRESHOLD = "probability_threshold"
    PUNCTUATION = "punctuation"
    COHERENCE_SCORE = "coherence_score"
    COMBINED = "combined"


@dataclass
class TraversalConfig:
    """Configuration for graph traversal"""
    strategy: TraversalStrategy = TraversalStrategy.PROBABILISTIC
    max_tokens: int = 100
    min_probability: float = 0.01
    temperature: float = 1.0
    beam_width: int = 5
    stop_criteria: List[StopCriterion] = None
    repetition_penalty: float = 1.2
    coherence_threshold: float = 0.3

    def __post_init__(self):
        if self.stop_criteria is None:
            self.stop_criteria = [
                StopCriterion.MAX_TOKENS,
                StopCriterion.PUNCTUATION,
                StopCriterion.PROBABILITY_THRESHOLD
            ]


@dataclass
class TraversalPath:
    """Represents a path through the graph"""
    nodes: List[str]
    tokens: List[str]
    weights: List[float]
    total_score: float

    def add_step(self, node_id: str, token: str, weight: float):
        """Add a step to the path"""
        self.nodes.append(node_id)
        self.tokens.append(token)
        self.weights.append(weight)
        self.total_score += np.log(weight + 1e-10)  # Log probability


class TraversalEngine:
    """
    Engine for traversing the graph network to generate text.
    Implements multiple traversal strategies and stop criteria.
    """

    def __init__(self, graph: GraphNetwork, config: Optional[TraversalConfig] = None):
        self.graph = graph
        self.config = config or TraversalConfig()
        self.visited_nodes: Set[str] = set()
        self.repetition_count: Dict[str, int] = {}

    def reset(self):
        """Reset traversal state"""
        self.visited_nodes.clear()
        self.repetition_count.clear()

    def traverse(self, start_node_id: str, context: Optional[List[str]] = None) -> TraversalPath:
        """
        Traverse the graph starting from a given node.

        Args:
            start_node_id: Starting node ID
            context: Optional context tokens to guide traversal

        Returns:
            TraversalPath containing the generated sequence
        """
        self.reset()

        if start_node_id not in self.graph.nodes:
            raise ValueError(f"Start node {start_node_id} not found in graph")

        if self.config.strategy == TraversalStrategy.BEAM_SEARCH:
            return self._beam_search_traverse(start_node_id, context)
        else:
            return self._single_path_traverse(start_node_id, context)

    def _single_path_traverse(self, start_node_id: str, context: Optional[List[str]] = None) -> TraversalPath:
        """Single path traversal (greedy, probabilistic, or temperature-based)"""
        path = TraversalPath(
            nodes=[start_node_id],
            tokens=[self.graph.get_node(start_node_id).value],
            weights=[1.0],
            total_score=0.0
        )

        current_node_id = start_node_id

        while not self._should_stop(path, current_node_id):
            # Get next node
            next_node_id, weight = self._select_next_node(current_node_id, context)

            if next_node_id is None:
                break

            # Add to path
            next_node = self.graph.get_node(next_node_id)
            path.add_step(next_node_id, next_node.value, weight)

            # Update state
            self.visited_nodes.add(next_node_id)
            self.repetition_count[next_node_id] = self.repetition_count.get(next_node_id, 0) + 1
            current_node_id = next_node_id

        return path

    def _beam_search_traverse(self, start_node_id: str, context: Optional[List[str]] = None) -> TraversalPath:
        """Beam search traversal to maintain multiple candidate paths"""
        # Initialize beam with start node
        beam = [TraversalPath(
            nodes=[start_node_id],
            tokens=[self.graph.get_node(start_node_id).value],
            weights=[1.0],
            total_score=0.0
        )]

        for step in range(self.config.max_tokens):
            candidates = []

            # Expand each path in the beam
            for path in beam:
                current_node_id = path.nodes[-1]

                # Check if this path should stop
                if self._should_stop(path, current_node_id):
                    candidates.append((path, path.total_score))
                    continue

                # Get top-k next nodes
                neighbors = self.graph.get_weighted_neighbors(current_node_id)

                for next_node_id, weight in neighbors[:self.config.beam_width]:
                    # Create new path
                    new_path = TraversalPath(
                        nodes=path.nodes.copy(),
                        tokens=path.tokens.copy(),
                        weights=path.weights.copy(),
                        total_score=path.total_score
                    )

                    next_node = self.graph.get_node(next_node_id)
                    adjusted_weight = self._apply_repetition_penalty(
                        next_node_id, weight, path.nodes
                    )
                    new_path.add_step(next_node_id, next_node.value, adjusted_weight)

                    candidates.append((new_path, new_path.total_score))

            if not candidates:
                break

            # Keep top-k paths
            candidates.sort(key=lambda x: x[1], reverse=True)
            beam = [path for path, score in candidates[:self.config.beam_width]]

            # Check if all beams have stopped
            if all(self._should_stop(path, path.nodes[-1]) for path in beam):
                break

        # Return best path
        return max(beam, key=lambda p: p.total_score)

    def _select_next_node(self, current_node_id: str, context: Optional[List[str]] = None) -> Tuple[Optional[str], float]:
        """
        Select the next node based on the traversal strategy.

        Returns:
            Tuple of (next_node_id, weight) or (None, 0.0) if no valid next node
        """
        neighbors = self.graph.get_weighted_neighbors(current_node_id)

        if not neighbors:
            return None, 0.0

        # Apply repetition penalty
        adjusted_neighbors = [
            (node_id, self._apply_repetition_penalty(node_id, weight, [current_node_id]))
            for node_id, weight in neighbors
        ]

        # Filter by minimum probability
        adjusted_neighbors = [
            (node_id, weight) for node_id, weight in adjusted_neighbors
            if weight >= self.config.min_probability
        ]

        if not adjusted_neighbors:
            return None, 0.0

        if self.config.strategy == TraversalStrategy.GREEDY:
            return adjusted_neighbors[0]  # Already sorted by weight

        elif self.config.strategy == TraversalStrategy.PROBABILISTIC:
            return self._probabilistic_selection(adjusted_neighbors)

        elif self.config.strategy == TraversalStrategy.TEMPERATURE:
            return self._temperature_selection(adjusted_neighbors)

        return adjusted_neighbors[0]

    def _probabilistic_selection(self, neighbors: List[Tuple[str, float]]) -> Tuple[str, float]:
        """Select next node probabilistically based on weights"""
        node_ids, weights = zip(*neighbors)
        weights = np.array(weights)

        # Normalize weights to probabilities
        probabilities = weights / weights.sum()

        # Sample
        selected_idx = np.random.choice(len(node_ids), p=probabilities)
        return node_ids[selected_idx], weights[selected_idx]

    def _temperature_selection(self, neighbors: List[Tuple[str, float]]) -> Tuple[str, float]:
        """Select next node using temperature-based sampling"""
        node_ids, weights = zip(*neighbors)
        weights = np.array(weights)

        # Apply temperature
        logits = np.log(weights + 1e-10) / self.config.temperature

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / exp_logits.sum()

        # Sample
        selected_idx = np.random.choice(len(node_ids), p=probabilities)
        return node_ids[selected_idx], weights[selected_idx]

    def _apply_repetition_penalty(self, node_id: str, weight: float, path_nodes: List[str]) -> float:
        """Apply penalty for repeated nodes"""
        repetition_count = path_nodes.count(node_id)
        if repetition_count > 0:
            penalty = self.config.repetition_penalty ** repetition_count
            weight = weight / penalty
        return max(weight, 0.0)

    def _should_stop(self, path: TraversalPath, current_node_id: str) -> bool:
        """
        Check if traversal should stop based on configured criteria.

        Args:
            path: Current traversal path
            current_node_id: Current node ID

        Returns:
            True if should stop, False otherwise
        """
        for criterion in self.config.stop_criteria:
            if criterion == StopCriterion.MAX_TOKENS:
                if len(path.tokens) >= self.config.max_tokens:
                    return True

            elif criterion == StopCriterion.PROBABILITY_THRESHOLD:
                if path.weights and path.weights[-1] < self.config.min_probability:
                    return True

            elif criterion == StopCriterion.PUNCTUATION:
                if self._is_sentence_end(path):
                    return True

            elif criterion == StopCriterion.COHERENCE_SCORE:
                if not self._is_coherent(path):
                    return True

        return False

    def _is_sentence_end(self, path: TraversalPath) -> bool:
        """Check if path ends with sentence-ending punctuation"""
        if len(path.tokens) < 5:  # Minimum sentence length
            return False

        if path.tokens:
            last_token = path.tokens[-1].strip()
            return last_token in {'.', '!', '?', '..."', '."', '!"', '?"'}
        return False

    def _is_coherent(self, path: TraversalPath) -> bool:
        """
        Check if the path maintains coherence.
        Uses average weight as a simple coherence metric.
        """
        if len(path.weights) < 3:
            return True

        recent_weights = path.weights[-5:]  # Check last 5 steps
        avg_weight = np.mean(recent_weights)

        return avg_weight >= self.config.coherence_threshold

    def generate_text(self, start_token: str, config: Optional[TraversalConfig] = None) -> str:
        """
        Generate text starting from a given token.

        Args:
            start_token: Starting token/word
            config: Optional traversal configuration

        Returns:
            Generated text as a string
        """
        if config:
            original_config = self.config
            self.config = config

        # Find node with matching value
        start_node_id = None
        for node_id, node in self.graph.nodes.items():
            if node.value.lower() == start_token.lower():
                start_node_id = node_id
                break

        if start_node_id is None:
            # Find closest match
            for node_id, node in self.graph.nodes.items():
                if start_token.lower() in node.value.lower():
                    start_node_id = node_id
                    break

        if start_node_id is None:
            return f"[Could not find starting token: {start_token}]"

        path = self.traverse(start_node_id)

        if config:
            self.config = original_config

        # Reconstruct text from tokens
        text = self._reconstruct_text(path.tokens)
        return text

    def _reconstruct_text(self, tokens: List[str]) -> str:
        """Reconstruct natural text from tokens"""
        if not tokens:
            return ""

        result = []
        for i, token in enumerate(tokens):
            if i == 0:
                result.append(token)
            elif token in {',', '.', '!', '?', ':', ';', "'s", "'t", "'re", "'ve", "'ll", "'d"}:
                result.append(token)
            elif token in {'(', '[', '{'}:
                result.append(' ' + token)
            elif result and result[-1] in {'(', '[', '{', '"', "'"}:
                result.append(token)
            else:
                result.append(' ' + token)

        return ''.join(result).strip()

    def get_multiple_completions(self, start_token: str, num_completions: int = 3,
                                 config: Optional[TraversalConfig] = None) -> List[str]:
        """
        Generate multiple different completions from the same starting token.

        Args:
            start_token: Starting token
            num_completions: Number of different completions to generate
            config: Optional traversal configuration

        Returns:
            List of generated text completions
        """
        completions = []

        for _ in range(num_completions):
            self.reset()
            completion = self.generate_text(start_token, config)
            completions.append(completion)

        return completions

