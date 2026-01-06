"""
Training Engine
Implements training algorithms and weight update mechanisms for the graph network.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
import time
from collections import defaultdict

from core.graph_network import GraphNetwork, Node, Edge


@dataclass
class TrainingConfig:
    """Configuration for training"""
    learning_rate: float = 0.1
    batch_size: int = 32
    num_epochs: int = 10
    weight_decay: float = 0.0001
    momentum: float = 0.9
    use_reinforcement: bool = True
    reward_discount: float = 0.95
    exploration_rate: float = 0.1
    min_edge_weight: float = 0.01


@dataclass
class TrainingMetrics:
    """Training metrics and statistics"""
    epoch: int
    loss: float
    avg_weight: float
    num_updates: int
    time_elapsed: float

    def __repr__(self):
        return (f"Epoch {self.epoch}: Loss={self.loss:.4f}, "
                f"AvgWeight={self.avg_weight:.4f}, Updates={self.num_updates}, "
                f"Time={self.time_elapsed:.2f}s")


class TrainingEngine:
    """
    Engine for training and updating the graph network.
    Supports multiple training paradigms including reinforcement learning.
    """

    def __init__(self, graph: GraphNetwork, config: Optional[TrainingConfig] = None):
        self.graph = graph
        self.config = config or TrainingConfig()

        # Training state
        self.velocity = {}  # For momentum
        self.training_history: List[TrainingMetrics] = []
        self.edge_updates = defaultdict(int)

    def train_from_sequences(self, sequences: List[List[str]],
                            validate_sequences: Optional[List[List[str]]] = None) -> List[TrainingMetrics]:
        """
        Train the graph from sequences of tokens.

        Args:
            sequences: List of token sequences for training
            validate_sequences: Optional validation sequences

        Returns:
            List of training metrics per epoch
        """
        print(f"Starting training with {len(sequences)} sequences for {self.config.num_epochs} epochs")

        for epoch in range(self.config.num_epochs):
            start_time = time.time()

            # Shuffle sequences
            np.random.shuffle(sequences)

            # Process in batches
            total_loss = 0.0
            num_updates = 0

            for i in range(0, len(sequences), self.config.batch_size):
                batch = sequences[i:i + self.config.batch_size]
                loss, updates = self._train_batch(batch)
                total_loss += loss
                num_updates += updates

            # Calculate metrics
            avg_loss = total_loss / len(sequences) if sequences else 0
            avg_weight = self._calculate_average_weight()
            time_elapsed = time.time() - start_time

            metrics = TrainingMetrics(
                epoch=epoch + 1,
                loss=avg_loss,
                avg_weight=avg_weight,
                num_updates=num_updates,
                time_elapsed=time_elapsed
            )

            self.training_history.append(metrics)
            print(metrics)

            # Validation
            if validate_sequences and (epoch + 1) % 5 == 0:
                val_loss = self._validate(validate_sequences)
                print(f"  Validation Loss: {val_loss:.4f}")

            # Prune low-weight edges periodically
            if (epoch + 1) % 10 == 0:
                pruned = self.graph.prune_edges(
                    min_weight=self.config.min_edge_weight,
                    min_co_occurrence=2
                )
                print(f"  Pruned {pruned} low-weight edges")

        return self.training_history

    def _train_batch(self, sequences: List[List[str]]) -> Tuple[float, int]:
        """
        Train on a batch of sequences.

        Returns:
            Tuple of (batch_loss, num_updates)
        """
        batch_loss = 0.0
        num_updates = 0

        for sequence in sequences:
            loss, updates = self._train_sequence(sequence)
            batch_loss += loss
            num_updates += updates

        return batch_loss, num_updates

    def _train_sequence(self, sequence: List[str]) -> Tuple[float, int]:
        """
        Train on a single sequence.
        Uses prediction error to update weights.

        Returns:
            Tuple of (sequence_loss, num_updates)
        """
        sequence_loss = 0.0
        num_updates = 0

        # Find node IDs for sequence
        node_ids = []
        for token in sequence:
            node_id = self._find_node_by_value(token)
            if node_id:
                node_ids.append(node_id)

        if len(node_ids) < 2:
            return 0.0, 0

        # Update edges along the sequence
        for i in range(len(node_ids) - 1):
            current_id = node_ids[i]
            next_id = node_ids[i + 1]

            # Get or create edge
            edge = self.graph.get_edge(current_id, next_id)

            if edge:
                # Positive reinforcement for existing edge
                target_weight = min(1.0, edge.weight + self.config.learning_rate)
                loss = self._update_edge_weight(edge, target_weight)
                sequence_loss += loss
                num_updates += 1
            else:
                # Create new edge
                edge = Edge(
                    source=current_id,
                    target=next_id,
                    weight=self.config.learning_rate,
                    edge_type="sequential",
                    co_occurrence=1
                )
                self.graph.add_edge(edge)
                num_updates += 1

            # Update competing edges (negative reinforcement)
            neighbors = self.graph.get_weighted_neighbors(current_id)
            for neighbor_id, weight in neighbors:
                if neighbor_id != next_id:
                    competing_edge = self.graph.get_edge(current_id, neighbor_id)
                    if competing_edge:
                        # Small penalty for not-taken edges
                        penalty = self.config.learning_rate * 0.1
                        competing_edge.update_weight(-penalty)

        return sequence_loss, num_updates

    def _update_edge_weight(self, edge: Edge, target_weight: float) -> float:
        """
        Update edge weight with momentum.

        Returns:
            Loss (difference from target)
        """
        edge_key = (edge.source, edge.target)

        # Calculate gradient
        gradient = target_weight - edge.weight

        # Apply momentum
        if edge_key not in self.velocity:
            self.velocity[edge_key] = 0.0

        self.velocity[edge_key] = (self.config.momentum * self.velocity[edge_key] +
                                   self.config.learning_rate * gradient)

        # Update weight
        edge.update_weight(self.velocity[edge_key])

        # Apply weight decay
        if self.config.weight_decay > 0:
            decay = -self.config.weight_decay * edge.weight
            edge.update_weight(decay)

        # Track update
        self.edge_updates[edge_key] += 1

        # Return loss
        return abs(target_weight - edge.weight)

    def train_with_reinforcement(self, graph_traverser, start_tokens: List[str],
                                 reward_function: callable) -> List[TrainingMetrics]:
        """
        Train using reinforcement learning.

        Args:
            graph_traverser: TraversalEngine instance for generating sequences
            start_tokens: List of starting tokens for generation
            reward_function: Function that takes generated text and returns reward

        Returns:
            List of training metrics
        """
        print("Starting reinforcement learning training")

        for epoch in range(self.config.num_epochs):
            start_time = time.time()

            total_reward = 0.0
            num_updates = 0

            for start_token in start_tokens:
                # Generate sequence
                path = graph_traverser.traverse(
                    self._find_node_by_value(start_token) or start_token
                )

                # Calculate reward
                generated_text = graph_traverser._reconstruct_text(path.tokens)
                reward = reward_function(generated_text)
                total_reward += reward

                # Update weights based on reward
                updates = self._apply_reward_to_path(path, reward)
                num_updates += updates

            avg_reward = total_reward / len(start_tokens) if start_tokens else 0
            avg_weight = self._calculate_average_weight()
            time_elapsed = time.time() - start_time

            metrics = TrainingMetrics(
                epoch=epoch + 1,
                loss=-avg_reward,  # Negative reward as loss
                avg_weight=avg_weight,
                num_updates=num_updates,
                time_elapsed=time_elapsed
            )

            self.training_history.append(metrics)
            print(f"Epoch {epoch + 1}: Avg Reward={avg_reward:.4f}, "
                  f"Updates={num_updates}, Time={time_elapsed:.2f}s")

        return self.training_history

    def _apply_reward_to_path(self, path, reward: float) -> int:
        """
        Apply reward to edges in a traversal path.
        Uses discounted rewards for earlier steps.

        Returns:
            Number of updates
        """
        num_updates = 0

        # Apply discounted rewards backward through the path
        for i in range(len(path.nodes) - 1):
            source_id = path.nodes[i]
            target_id = path.nodes[i + 1]

            edge = self.graph.get_edge(source_id, target_id)
            if edge:
                # Calculate discounted reward
                discount = self.config.reward_discount ** (len(path.nodes) - i - 1)
                discounted_reward = reward * discount

                # Update weight
                weight_delta = self.config.learning_rate * discounted_reward
                edge.update_weight(weight_delta)
                num_updates += 1

        return num_updates

    def _validate(self, sequences: List[List[str]]) -> float:
        """Calculate validation loss"""
        total_loss = 0.0

        for sequence in sequences:
            loss, _ = self._train_sequence(sequence)
            total_loss += loss

        return total_loss / len(sequences) if sequences else 0

    def _calculate_average_weight(self) -> float:
        """Calculate average edge weight in the graph"""
        if not self.graph.edges:
            return 0.0

        total_weight = sum(edge.weight for edge in self.graph.edges.values())
        return total_weight / len(self.graph.edges)

    def _find_node_by_value(self, value: str) -> Optional[str]:
        """Find node ID by its value"""
        value_lower = value.lower()
        for node_id, node in self.graph.nodes.items():
            if node.value.lower() == value_lower:
                return node_id
        return None

    def update_context_vectors(self, embedding_model=None):
        """
        Update context vectors for all nodes.
        Can use external embedding model or compute from graph structure.

        Args:
            embedding_model: Optional embedding model (e.g., word2vec, BERT)
        """
        print("Updating context vectors...")

        for node_id, node in self.graph.nodes.items():
            if embedding_model:
                # Use external embeddings
                try:
                    vector = embedding_model.encode(node.value)
                    node.context_vector = vector
                except:
                    pass
            else:
                # Compute from graph structure (simple approach)
                vector = self._compute_structural_embedding(node_id)
                node.context_vector = vector

        print("Context vectors updated")

    def _compute_structural_embedding(self, node_id: str, depth: int = 2) -> np.ndarray:
        """
        Compute node embedding based on graph structure.
        Uses neighbor information up to specified depth.
        """
        embedding = np.zeros(128)

        # Add node's own frequency
        node = self.graph.get_node(node_id)
        embedding[0] = np.log1p(node.frequency)

        # Add neighbor information
        neighbors = self.graph.get_weighted_neighbors(node_id)
        for i, (neighbor_id, weight) in enumerate(neighbors[:10]):
            if i < 64:
                embedding[i + 1] = weight

        # Add predecessor information
        predecessors = self.graph.get_predecessors(node_id)
        for i, pred_id in enumerate(predecessors[:10]):
            edge = self.graph.get_edge(pred_id, node_id)
            if edge and i < 64:
                embedding[i + 65] = edge.weight

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def get_training_summary(self) -> Dict:
        """Get summary of training process"""
        if not self.training_history:
            return {}

        return {
            'total_epochs': len(self.training_history),
            'final_loss': self.training_history[-1].loss,
            'final_avg_weight': self.training_history[-1].avg_weight,
            'total_time': sum(m.time_elapsed for m in self.training_history),
            'total_updates': sum(m.num_updates for m in self.training_history),
            'most_updated_edges': sorted(
                self.edge_updates.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }

