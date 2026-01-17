"""
GraphStore: Persistent network topology.

Maintains the graph structure (neurons and connections) separate from computation state.
This enables:
- Topology to be serialized/persisted independently
- Swapping computation engines without changing topology
- Clear separation between "how the network is wired" and "how it computes"

Core responsibility: Store and retrieve graph structure only (neurons, edges, types).

Methods:
- add_node, add_edge: Build topology
- get_node, get_edges, get_incoming_edges: Query topology
- persist(): Save topology to file
- restore(): Load topology from file
"""

from typing import Dict, List, Tuple, Optional, Iterable
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class NodeDesc:
    """Description of a neuron node."""
    node_id: int
    node_type: str  # "input", "hidden", "output"
    threshold: float
    refractory_period: int


@dataclass
class EdgeDesc:
    """Description of a synaptic connection."""
    source_id: int
    target_id: int
    weight: float
    is_inhibitory: bool
    synapse_index: int = 0  # Index in target's input list (for multi-synapse targets)


class GraphStore:
    """
    Persistent storage of network topology.

    Maintains:
    - Node list with parameters
    - Edge list (adjacency information)
    - Metadata (creation time, version)

    Does NOT maintain:
    - Membrane potentials
    - Firing states
    - Eligibility traces
    - Weights (those are in synapses)

    All methods are read-only except initialization and reset.

    Public Interface:
    - get_nodes() -> Iterable[NodeDesc]
    - get_edges() -> Iterable[EdgeDesc]
    - get_node(node_id) -> Optional[NodeDesc]
    - get_incoming_edges(node_id) -> Iterable[EdgeDesc]
    - get_outgoing_edges(node_id) -> Iterable[EdgeDesc]
    """

    def __init__(self):
        """Initialize empty graph."""
        self._nodes: Dict[int, NodeDesc] = {}
        self._edges: List[EdgeDesc] = []
        self._incoming: Dict[int, List[EdgeDesc]] = {}  # target_id -> [edges]
        self._outgoing: Dict[int, List[EdgeDesc]] = {}  # source_id -> [edges]

    def add_node(
        self,
        node_id: int,
        node_type: str = "hidden",
        threshold: float = 1.0,
        refractory_period: int = 3
    ) -> NodeDesc:
        """
        Register a neuron node in the graph.

        Args:
            node_id: Unique neuron identifier
            node_type: "input", "hidden", or "output"
            threshold: Firing threshold
            refractory_period: Refractory timesteps after spike

        Returns:
            Created NodeDesc
        """
        node = NodeDesc(node_id, node_type, threshold, refractory_period)
        self._nodes[node_id] = node
        if node_id not in self._incoming:
            self._incoming[node_id] = []
        if node_id not in self._outgoing:
            self._outgoing[node_id] = []
        return node

    def add_edge(
        self,
        source_id: int,
        target_id: int,
        weight: float,
        is_inhibitory: bool = False
    ) -> EdgeDesc:
        """
        Register a synaptic connection.

        Args:
            source_id: Presynaptic neuron
            target_id: Postsynaptic neuron
            weight: Initial synaptic weight
            is_inhibitory: If True, inhibitory synapse

        Returns:
            Created EdgeDesc

        Raises:
            ValueError: If source or target node doesn't exist
        """
        if source_id not in self._nodes:
            raise ValueError(f"Source node {source_id} not in graph")
        if target_id not in self._nodes:
            raise ValueError(f"Target node {target_id} not in graph")

        synapse_index = len(self._incoming[target_id])
        edge = EdgeDesc(source_id, target_id, weight, is_inhibitory, synapse_index)

        self._edges.append(edge)
        self._incoming[target_id].append(edge)
        self._outgoing[source_id].append(edge)

        return edge

    def get_node(self, node_id: int) -> Optional[NodeDesc]:
        """
        Retrieve node descriptor.

        Args:
            node_id: Neuron ID

        Returns:
            NodeDesc or None if not found
        """
        return self._nodes.get(node_id)

    def get_nodes(self) -> Iterable[NodeDesc]:
        """Iterate over all nodes."""
        return iter(self._nodes.values())

    def get_edges(self) -> Iterable[EdgeDesc]:
        """Iterate over all edges."""
        return iter(self._edges)

    def get_incoming_edges(self, node_id: int) -> Iterable[EdgeDesc]:
        """
        Get all edges terminating at a node (inputs to that neuron).

        Args:
            node_id: Target neuron

        Returns:
            Iterable of incoming edges
        """
        return iter(self._incoming.get(node_id, []))

    def get_outgoing_edges(self, node_id: int) -> Iterable[EdgeDesc]:
        """
        Get all edges originating from a node (outputs from that neuron).

        Args:
            node_id: Source neuron

        Returns:
            Iterable of outgoing edges
        """
        return iter(self._outgoing.get(node_id, []))

    def get_node_count(self) -> int:
        """Get total number of neurons."""
        return len(self._nodes)

    def get_edge_count(self) -> int:
        """Get total number of synapses."""
        return len(self._edges)

    def has_node(self, node_id: int) -> bool:
        """Check if neuron exists."""
        return node_id in self._nodes

    def reset(self) -> None:
        """Clear all topology."""
        self._nodes.clear()
        self._edges.clear()
        self._incoming.clear()
        self._outgoing.clear()

    def __repr__(self):
        return f"GraphStore(nodes={self.get_node_count()}, edges={self.get_edge_count()})"

    def persist(self, filepath: str) -> None:
        """
        Save graph topology to JSON file.

        Serializes all nodes and edges. Can be restored with restore().

        Args:
            filepath: Path to save topology file

        Raises:
            IOError: If write fails
        """
        data = {
            "nodes": [
                {
                    "node_id": node.node_id,
                    "node_type": node.node_type,
                    "threshold": node.threshold,
                    "refractory_period": node.refractory_period,
                }
                for node in self._nodes.values()
            ],
            "edges": [
                {
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "weight": edge.weight,
                    "is_inhibitory": edge.is_inhibitory,
                    "synapse_index": edge.synapse_index,
                }
                for edge in self._edges
            ],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def restore(filepath: str) -> "GraphStore":
        """
        Load graph topology from JSON file.

        Reconstructs a GraphStore from persisted data.

        Args:
            filepath: Path to topology file

        Returns:
            New GraphStore with loaded topology

        Raises:
            IOError: If file not found
            ValueError: If data is corrupted
        """
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            raise IOError(f"Topology file not found: {filepath}")
        except json.JSONDecodeError:
            raise ValueError(f"Corrupted topology file: {filepath}")

        graph = GraphStore()

        # Restore nodes
        for node_data in data.get("nodes", []):
            graph.add_node(
                node_id=node_data["node_id"],
                node_type=node_data["node_type"],
                threshold=node_data["threshold"],
                refractory_period=node_data["refractory_period"],
            )

        # Restore edges
        for edge_data in data.get("edges", []):
            graph.add_edge(
                source_id=edge_data["source_id"],
                target_id=edge_data["target_id"],
                weight=edge_data["weight"],
                is_inhibitory=edge_data["is_inhibitory"],
            )

        return graph

