"""
NCGN Memory Module - Core Data Structures

Implements the "Hardware" of the cognitive system:
- ConceptNode: Atomic unit of state (membrane potential)
- Synapse: Unit of memory and association
- GraphMemory: O(1) storage engine with forward/backward indices
- EventSchema: Logic templates for System 2 validation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
import json


class ConceptNode:
    """
    The atomic unit of state in the NCGN.
    
    Represents a concept with membrane potential (energy) that can
    accumulate, decay, and fire when threshold is exceeded.
    """
    __slots__ = [
        'id',                # str: Unique identifier ("dog")
        'energy',            # float: 0.0 to 1.0 (Membrane Potential)
        'resting_potential', # float: 0.0 (Base level after firing)
        'threshold',         # float: 0.75 (Firing threshold)
        'refractory_timer',  # int: Ticks until node can re-fire
        'last_spike_tick',   # int: For STDP learning
        'novelty_score',     # float: 1.0 -> 0.0 (Decays over time)
    ]
    
    def __init__(
        self,
        id: str,
        energy: float = 0.0,
        resting_potential: float = 0.0,
        threshold: float = 0.75,
        refractory_timer: int = 0,
        last_spike_tick: int = -1,
        novelty_score: float = 1.0
    ):
        self.id = id
        self.energy = energy
        self.resting_potential = resting_potential
        self.threshold = threshold
        self.refractory_timer = refractory_timer
        self.last_spike_tick = last_spike_tick
        self.novelty_score = novelty_score
    
    def can_fire(self) -> bool:
        """Check if node can fire (above threshold and not refractory)."""
        return self.energy > self.threshold and self.refractory_timer == 0
    
    def reset_after_fire(self, current_tick: int, refractory_period: int = 3):
        """Reset node state after firing."""
        self.energy = self.resting_potential
        self.refractory_timer = refractory_period
        self.last_spike_tick = current_tick
    
    def __repr__(self) -> str:
        return f"ConceptNode({self.id}, E={self.energy:.3f}, T={self.threshold})"


class Synapse:
    """
    The unit of memory and association.
    
    Connects two ConceptNodes with weight (flow speed) and 
    confidence (epistemic truth value).
    """
    __slots__ = [
        'target_id',    # str: Pointer to post-synaptic node
        'type',         # str: "is_a", "eats", "temporal_next", etc.
        'weight',       # float: 0.0 to 1.0 (Hebbian association strength)
        'confidence',   # float: 0.0 to 1.0 (Epistemic truth value)
        'stdp_trace',   # float: Eligibility trace for delayed learning
        'last_active',  # int: Last tick this synapse transmitted a spike
    ]
    
    def __init__(
        self,
        target_id: str,
        type: str = "associates",
        weight: float = 0.5,
        confidence: float = 0.5,
        stdp_trace: float = 0.0,
        last_active: int = -1
    ):
        self.target_id = target_id
        self.type = type
        self.weight = weight
        self.confidence = confidence
        self.stdp_trace = stdp_trace
        self.last_active = last_active
    
    def transmit(self, spike_magnitude: float = 1.0) -> float:
        """Calculate energy to transmit to target."""
        return self.weight * spike_magnitude
    
    def __repr__(self) -> str:
        return f"Synapse(→{self.target_id}, W={self.weight:.2f}, C={self.confidence:.2f})"


class GraphMemory:
    """
    The optimized storage engine for NCGN.
    
    All lookups are O(1) using dictionary-based indices.
    Provides forward and backward edge traversal for both
    prediction (forward) and explanation (backward) queries.
    """
    
    def __init__(self):
        # Primary Index: O(1) retrieval by node ID
        self.nodes: Dict[str, ConceptNode] = {}
        
        # Forward Index (Adjacency): O(1) retrieval of downstream neighbors
        # { "dog": [Synapse(meat), Synapse(bark)] }
        self.forward_edges: Dict[str, List[Synapse]] = {}
        
        # Reverse Index: O(1) retrieval of upstream causes
        # { "meat": [("dog", Synapse)] }
        self.backward_edges: Dict[str, List[Tuple[str, Synapse]]] = {}
        
        # Active nodes set for sparse iteration
        self._active_nodes: Set[str] = set()
    
    # =====================
    # Node Operations
    # =====================
    
    def add_node(
        self,
        node_id: str,
        energy: float = 0.0,
        threshold: float = 0.75,
        novelty_score: float = 1.0
    ) -> ConceptNode:
        """Add a new node to the graph. O(1)."""
        if node_id in self.nodes:
            return self.nodes[node_id]
        
        node = ConceptNode(
            id=node_id,
            energy=energy,
            threshold=threshold,
            novelty_score=novelty_score
        )
        self.nodes[node_id] = node
        self.forward_edges[node_id] = []
        self.backward_edges[node_id] = []
        
        if energy > 0:
            self._active_nodes.add(node_id)
        
        return node
    
    def get_node(self, node_id: str) -> Optional[ConceptNode]:
        """Retrieve a node by ID. O(1)."""
        return self.nodes.get(node_id)
    
    def has_node(self, node_id: str) -> bool:
        """Check if node exists. O(1)."""
        return node_id in self.nodes
    
    def remove_node(self, node_id: str) -> bool:
        """Remove a node and all its edges. O(E) where E is edge count."""
        if node_id not in self.nodes:
            return False
        
        # Remove forward edges from this node
        del self.forward_edges[node_id]
        
        # Remove backward references to this node
        for source_id, synapse in self.backward_edges.get(node_id, []):
            edges = self.forward_edges.get(source_id, [])
            self.forward_edges[source_id] = [s for s in edges if s.target_id != node_id]
        
        # Remove backward edges from this node
        del self.backward_edges[node_id]
        
        # Remove forward references from other nodes
        for edges in self.forward_edges.values():
            edges[:] = [s for s in edges if s.target_id != node_id]
        
        # Remove from active set and nodes dict
        self._active_nodes.discard(node_id)
        del self.nodes[node_id]
        
        return True
    
    # =====================
    # Synapse Operations
    # =====================
    
    def add_synapse(
        self,
        source_id: str,
        target_id: str,
        type: str = "associates",
        weight: float = 0.5,
        confidence: float = 0.5
    ) -> Optional[Synapse]:
        """Add a synapse between two nodes. O(1)."""
        # Ensure both nodes exist
        if source_id not in self.nodes:
            self.add_node(source_id)
        if target_id not in self.nodes:
            self.add_node(target_id)
        
        synapse = Synapse(
            target_id=target_id,
            type=type,
            weight=weight,
            confidence=confidence
        )
        
        # Add to forward index
        self.forward_edges[source_id].append(synapse)
        
        # Add to backward index
        self.backward_edges[target_id].append((source_id, synapse))
        
        return synapse
    
    def get_outgoing(self, node_id: str) -> List[Synapse]:
        """Get all outgoing synapses from a node. O(1)."""
        return self.forward_edges.get(node_id, [])
    
    def get_incoming(self, node_id: str) -> List[Tuple[str, Synapse]]:
        """Get all incoming synapses to a node. O(1)."""
        return self.backward_edges.get(node_id, [])
    
    def get_synapse(self, source_id: str, target_id: str, type: Optional[str] = None) -> Optional[Synapse]:
        """Find a specific synapse. O(E) where E is outgoing edges from source."""
        for synapse in self.forward_edges.get(source_id, []):
            if synapse.target_id == target_id:
                if type is None or synapse.type == type:
                    return synapse
        return None
    
    # =====================
    # Active Node Management
    # =====================
    
    def mark_active(self, node_id: str):
        """Mark a node as active (has non-zero energy)."""
        self._active_nodes.add(node_id)
    
    def mark_inactive(self, node_id: str):
        """Mark a node as inactive."""
        self._active_nodes.discard(node_id)
    
    def get_active_nodes(self) -> Set[str]:
        """Get the set of currently active nodes."""
        return self._active_nodes.copy()
    
    def update_active_set(self):
        """Refresh the active set based on current energies."""
        self._active_nodes = {
            node_id for node_id, node in self.nodes.items()
            if node.energy > 0
        }
    
    # =====================
    # Utility Methods
    # =====================
    
    def node_count(self) -> int:
        """Total number of nodes."""
        return len(self.nodes)
    
    def edge_count(self) -> int:
        """Total number of synapses."""
        return sum(len(edges) for edges in self.forward_edges.values())
    
    def clear(self):
        """Clear all nodes and edges."""
        self.nodes.clear()
        self.forward_edges.clear()
        self.backward_edges.clear()
        self._active_nodes.clear()


@dataclass
class EventSchema:
    """
    Logic template for System 2 validation.
    
    Defines the expected structure and constraints for actions,
    allowing the system to detect violations like "Dog eats Metal".
    """
    id: str                              # e.g., "schema_eat"
    action: str                          # e.g., "eat"
    confidence: float                    # How much we trust this rule
    roles: Dict[str, str]                # e.g., {"agent": "animate_object", "target": "edible_object"}
    constraints: Dict[str, List[str]] = field(default_factory=dict)  # e.g., {"target": ["is_edible"]}
    
    @classmethod
    def from_json(cls, json_path: str) -> 'EventSchema':
        """Load schema from JSON file."""
        with open(json_path, 'r') as f:
            data = json.load(f)
        return cls(
            id=data['id'],
            action=data['action'],
            confidence=data.get('confidence', 0.5),
            roles=data.get('roles', {}),
            constraints=data.get('constraints', {})
        )
    
    @classmethod
    def from_dict(cls, data: dict) -> 'EventSchema':
        """Load schema from dictionary."""
        return cls(
            id=data['id'],
            action=data['action'],
            confidence=data.get('confidence', 0.5),
            roles=data.get('roles', {}),
            constraints=data.get('constraints', {})
        )
    
    def to_dict(self) -> dict:
        """Convert schema to dictionary."""
        return {
            'id': self.id,
            'action': self.action,
            'confidence': self.confidence,
            'roles': self.roles,
            'constraints': self.constraints
        }
