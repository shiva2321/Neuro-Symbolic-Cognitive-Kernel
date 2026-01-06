"""
Core Graph Network Data Structures
Implements Node, Edge, and Graph classes for the neural network architecture.
"""

import numpy as np
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import pickle
from collections import defaultdict


class NodeType(Enum):
    """Types of nodes in the graph"""
    WORD = "word"
    PHRASE = "phrase"
    CONCEPT = "concept"
    PUNCTUATION = "punctuation"
    SPECIAL = "special"


class DomainType(Enum):
    """Domain specialization for nodes"""
    GENERAL = "general"
    TEXT = "text"
    MATH = "math"
    CODE = "code"
    MIXED = "mixed"


@dataclass
class Node:
    """
    Represents a node in the graph network.

    Attributes:
        id: Unique identifier for the node
        value: The text/content of the node
        node_type: Type of node (word, phrase, concept, etc.)
        domain: Domain specialization
        frequency: Number of times this node appears in training data
        context_vector: Embedding/feature vector for the node
        metadata: Additional information about the node
    """
    id: str
    value: str
    node_type: NodeType = NodeType.WORD
    domain: DomainType = DomainType.GENERAL
    frequency: int = 0
    context_vector: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.context_vector is None:
            # Initialize with zero vector (will be updated during training)
            self.context_vector = np.zeros(128)  # 128-dimensional embedding

    def update_frequency(self, count: int = 1):
        """Increment the frequency count"""
        self.frequency += count

    def update_context_vector(self, vector: np.ndarray, learning_rate: float = 0.1):
        """Update the context vector using exponential moving average"""
        self.context_vector = (1 - learning_rate) * self.context_vector + learning_rate * vector

    def to_dict(self) -> Dict:
        """Convert node to dictionary for serialization"""
        return {
            'id': self.id,
            'value': self.value,
            'node_type': self.node_type.value,
            'domain': self.domain.value,
            'frequency': self.frequency,
            'context_vector': self.context_vector.tolist() if self.context_vector is not None else None,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Node':
        """Create node from dictionary"""
        node = cls(
            id=data['id'],
            value=data['value'],
            node_type=NodeType(data['node_type']),
            domain=DomainType(data['domain']),
            frequency=data['frequency'],
            metadata=data.get('metadata', {})
        )
        if data.get('context_vector'):
            node.context_vector = np.array(data['context_vector'])
        return node


@dataclass
class Edge:
    """
    Represents a weighted edge between two nodes.

    Attributes:
        source: Source node ID
        target: Target node ID
        weight: Connection strength (0.0 to 1.0)
        edge_type: Type of relationship (sequential, semantic, syntactic)
        co_occurrence: Number of times these nodes appear together
        context_strength: Contextual relevance score
        metadata: Additional edge information
    """
    source: str
    target: str
    weight: float = 0.5
    edge_type: str = "sequential"
    co_occurrence: int = 0
    context_strength: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_weight(self, delta: float, min_weight: float = 0.0, max_weight: float = 1.0):
        """Update edge weight with bounds checking"""
        self.weight = np.clip(self.weight + delta, min_weight, max_weight)

    def update_co_occurrence(self, count: int = 1):
        """Increment co-occurrence count"""
        self.co_occurrence += count
        # Update weight based on co-occurrence (logarithmic scaling)
        self.weight = min(1.0, 0.1 + 0.2 * np.log1p(self.co_occurrence))

    def to_dict(self) -> Dict:
        """Convert edge to dictionary"""
        return {
            'source': self.source,
            'target': self.target,
            'weight': float(self.weight),
            'edge_type': self.edge_type,
            'co_occurrence': self.co_occurrence,
            'context_strength': float(self.context_strength),
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Edge':
        """Create edge from dictionary"""
        return cls(
            source=data['source'],
            target=data['target'],
            weight=data['weight'],
            edge_type=data.get('edge_type', 'sequential'),
            co_occurrence=data.get('co_occurrence', 0),
            context_strength=data.get('context_strength', 0.0),
            metadata=data.get('metadata', {})
        )


class GraphNetwork:
    """
    Main graph network structure for the neural network.
    Manages nodes, edges, and provides graph operations.
    """

    def __init__(self, name: str = "main_graph"):
        self.name = name
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[Tuple[str, str], Edge] = {}
        self.adjacency_list: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_adjacency_list: Dict[str, Set[str]] = defaultdict(set)
        self.domain_nodes: Dict[DomainType, Set[str]] = defaultdict(set)

    def add_node(self, node: Node) -> bool:
        """Add a node to the graph"""
        if node.id in self.nodes:
            # Update existing node
            self.nodes[node.id].update_frequency()
            return False

        self.nodes[node.id] = node
        self.domain_nodes[node.domain].add(node.id)
        return True

    def add_edge(self, edge: Edge) -> bool:
        """Add an edge to the graph"""
        # Ensure both nodes exist
        if edge.source not in self.nodes or edge.target not in self.nodes:
            return False

        edge_key = (edge.source, edge.target)

        if edge_key in self.edges:
            # Update existing edge
            self.edges[edge_key].update_co_occurrence()
            return False

        self.edges[edge_key] = edge
        self.adjacency_list[edge.source].add(edge.target)
        self.reverse_adjacency_list[edge.target].add(edge.source)
        return True

    def get_node(self, node_id: str) -> Optional[Node]:
        """Retrieve a node by ID"""
        return self.nodes.get(node_id)

    def get_edge(self, source: str, target: str) -> Optional[Edge]:
        """Retrieve an edge between two nodes"""
        return self.edges.get((source, target))

    def get_neighbors(self, node_id: str) -> List[str]:
        """Get all neighboring nodes (outgoing edges)"""
        return list(self.adjacency_list.get(node_id, set()))

    def get_predecessors(self, node_id: str) -> List[str]:
        """Get all predecessor nodes (incoming edges)"""
        return list(self.reverse_adjacency_list.get(node_id, set()))

    def get_weighted_neighbors(self, node_id: str) -> List[Tuple[str, float]]:
        """Get neighbors with their edge weights"""
        neighbors = []
        for neighbor_id in self.get_neighbors(node_id):
            edge = self.get_edge(node_id, neighbor_id)
            if edge:
                neighbors.append((neighbor_id, edge.weight))
        return sorted(neighbors, key=lambda x: x[1], reverse=True)

    def get_nodes_by_domain(self, domain: DomainType) -> List[Node]:
        """Get all nodes belonging to a specific domain"""
        return [self.nodes[node_id] for node_id in self.domain_nodes[domain]]

    def prune_edges(self, min_weight: float = 0.01, min_co_occurrence: int = 2):
        """
        Remove low-weight edges to optimize graph size and performance.

        Args:
            min_weight: Minimum weight threshold
            min_co_occurrence: Minimum co-occurrence count
        """
        edges_to_remove = []

        for edge_key, edge in self.edges.items():
            if edge.weight < min_weight or edge.co_occurrence < min_co_occurrence:
                edges_to_remove.append(edge_key)

        for edge_key in edges_to_remove:
            edge = self.edges[edge_key]
            self.adjacency_list[edge.source].discard(edge.target)
            self.reverse_adjacency_list[edge.target].discard(edge.source)
            del self.edges[edge_key]

        return len(edges_to_remove)

    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics"""
        total_weight = sum(edge.weight for edge in self.edges.values())
        avg_weight = total_weight / len(self.edges) if self.edges else 0

        node_degrees = [len(self.adjacency_list[node_id]) for node_id in self.nodes]

        return {
            'num_nodes': len(self.nodes),
            'num_edges': len(self.edges),
            'avg_edge_weight': avg_weight,
            'avg_node_degree': np.mean(node_degrees) if node_degrees else 0,
            'max_node_degree': max(node_degrees) if node_degrees else 0,
            'domain_distribution': {
                domain.value: len(nodes)
                for domain, nodes in self.domain_nodes.items()
            }
        }

    def save(self, filepath: str, format: str = 'pickle'):
        """
        Save graph to disk.

        Args:
            filepath: Path to save the graph
            format: 'pickle' or 'json'
        """
        if format == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
        elif format == 'json':
            data = {
                'name': self.name,
                'nodes': [node.to_dict() for node in self.nodes.values()],
                'edges': [edge.to_dict() for edge in self.edges.values()]
            }
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str, format: str = 'pickle') -> 'GraphNetwork':
        """
        Load graph from disk.

        Args:
            filepath: Path to load the graph from
            format: 'pickle' or 'json'
        """
        if format == 'pickle':
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        elif format == 'json':
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            graph = cls(name=data['name'])

            # Load nodes
            for node_data in data['nodes']:
                node = Node.from_dict(node_data)
                graph.add_node(node)

            # Load edges
            for edge_data in data['edges']:
                edge = Edge.from_dict(edge_data)
                graph.add_edge(edge)

            return graph

    def __len__(self) -> int:
        """Return number of nodes"""
        return len(self.nodes)

    def __repr__(self) -> str:
        return f"GraphNetwork(name='{self.name}', nodes={len(self.nodes)}, edges={len(self.edges)})"

