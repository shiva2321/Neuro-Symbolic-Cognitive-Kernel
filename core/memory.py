"""
NCGN v6.0 Memory Module - Core Data Structures

Implements the graph storage with:
- ConceptNode: Energy, threshold, cluster membership
- Synapse: Weight, trace (eligibility), stability (consolidation)
- GraphMemory: O(1) storage with cluster-aware operations

Key v6.0 Changes:
- Synapse.trace: Eligibility trace for 3-Factor Hebbian learning
- Synapse.stability: Consolidation factor to prevent catastrophic forgetting
- ClusterType: Groups nodes into motor/sensory/hidden for selective inhibition
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum
import math


class ClusterType(Enum):
    """
    Node cluster types for selective inhibition.
    
    Softmax is applied per-cluster to motor nodes (action selection).
    Hidden/Sensory may use different inhibition strategies.
    """
    MOTOR = "motor"          # Action output nodes (Softmax inhibition)
    SENSORY = "sensory"      # Input nodes (egocentric encoding)
    HIDDEN = "hidden"        # Internal processing nodes
    VALUE = "value"          # Value estimation nodes (for TD learning)
    GOAL = "goal"            # Goal/drive nodes


class ConceptNode:
    """
    The atomic unit of state in NCGN v6.0.
    
    Represents a concept with membrane potential (energy) that can
    accumulate, decay, and fire when threshold is exceeded.
    
    Uses __slots__ for memory efficiency.
    """
    
    __slots__ = [
        'id',               # str: Unique identifier
        'energy',           # float: Current membrane potential (0.0 to 1.0)
        'resting_potential',# float: Baseline energy level
        'threshold',        # float: Firing threshold
        'refractory_timer', # int: Ticks until can fire again
        'last_spike_tick',  # int: Tick of last spike
        'novelty_score',    # float: High for new concepts, decays over time
        'cluster',          # ClusterType: Which group this node belongs to
    ]
    
    def __init__(
        self,
        id: str,
        energy: float = 0.0,
        resting_potential: float = 0.0,
        threshold: float = 0.75,
        refractory_timer: int = 0,
        last_spike_tick: int = -1,
        novelty_score: float = 1.0,
        cluster: ClusterType = ClusterType.HIDDEN
    ):
        self.id = id
        self.energy = energy
        self.resting_potential = resting_potential
        self.threshold = threshold
        self.refractory_timer = refractory_timer
        self.last_spike_tick = last_spike_tick
        self.novelty_score = novelty_score
        self.cluster = cluster
    
    def can_fire(self) -> bool:
        """Check if node can fire (above threshold and not refractory)."""
        return self.energy > self.threshold and self.refractory_timer == 0
    
    def reset_after_fire(self, current_tick: int, refractory_period: int = 3):
        """Reset node state after firing."""
        self.energy = 0.0
        self.refractory_timer = refractory_period
        self.last_spike_tick = current_tick
    
    def __repr__(self) -> str:
        return f"ConceptNode({self.id}, E={self.energy:.2f}, cluster={self.cluster.value})"


class Synapse:
    """
    The unit of memory and association in NCGN v6.0.
    
    Connects two ConceptNodes with:
    - weight: Strength of connection (energy flow speed)
    - trace: Eligibility trace for 3-Factor Hebbian learning
    - stability: Consolidation factor that protects against forgetting
    
    CRITICAL v6.0 CHANGES:
    - trace: Set to 1.0 on Pre/Post coincidence, decays exponentially
    - stability: Increases with repeated reinforcement, gates learning rate
    
    Uses __slots__ for memory efficiency.
    """
    
    __slots__ = [
        'target_id',    # str: Pointer to post-synaptic node
        'type',         # str: Semantic type of relationship
        'weight',       # float: Connection strength (0.0 to 1.0)
        'confidence',   # float: Epistemic certainty (for System 2)
        'trace',        # float: Eligibility trace for 3-Factor learning
        'stability',    # float: Consolidation factor (0.0 = labile, 1.0 = rigid)
        'last_active',  # int: Tick of last activation
    ]
    
    def __init__(
        self,
        target_id: str,
        type: str = "associates",
        weight: float = 0.5,
        confidence: float = 0.5,
        trace: float = 0.0,
        stability: float = 0.0,
        last_active: int = -1
    ):
        self.target_id = target_id
        self.type = type
        self.weight = weight
        self.confidence = confidence
        self.trace = trace
        self.stability = stability
        self.last_active = last_active
    
    def transmit(self, spike_magnitude: float = 1.0) -> float:
        """Calculate energy to transmit to target."""
        return self.weight * spike_magnitude
    
    def decay_trace(self, decay_rate: float = 0.95):
        """
        Decay the eligibility trace.
        
        Called each tick. The trace decays exponentially, maintaining
        a memory of recent activity for delayed reward assignment.
        """
        self.trace *= decay_rate
        if self.trace < 0.001:
            self.trace = 0.0
    
    def set_eligible(self):
        """
        Mark synapse as eligible for learning.
        
        Called when Pre-synaptic and Post-synaptic nodes are
        both active (Hebbian coincidence detection).
        """
        self.trace = 1.0
    
    def __repr__(self) -> str:
        return f"Synapse(→{self.target_id}, w={self.weight:.2f}, t={self.trace:.2f}, s={self.stability:.2f})"


class GraphMemory:
    """
    The optimized storage engine for NCGN v6.0.
    
    All lookups are O(1) using dictionary-based indices.
    Provides forward and backward edge traversal, cluster grouping,
    and metabolic (energy) tracking for thermodynamic regulation.
    
    Key v6.0 Features:
    - Cluster-based node grouping for selective inhibition
    - Global energy tracking for seizure prevention
    - Traced synapse tracking for efficient reward broadcast
    """
    
    def __init__(self):
        # Core storage
        self.nodes: Dict[str, ConceptNode] = {}
        
        # Edge indices
        # Forward: source_id -> List[Synapse]
        self._forward_edges: Dict[str, List[Synapse]] = {}
        # Backward: target_id -> List[(source_id, Synapse)]
        self._backward_edges: Dict[str, List[Tuple[str, Synapse]]] = {}
        
        # Activity tracking
        self._active_nodes: Set[str] = set()
        
        # Cluster indices (for selective inhibition)
        self._clusters: Dict[ClusterType, Set[str]] = {
            cluster: set() for cluster in ClusterType
        }
        
        # Traced synapse tracking (for efficient reward broadcast)
        # Contains (source_id, Synapse) pairs with non-zero trace
        self._traced_synapses: Set[Tuple[str, str]] = set()
        
        # Metabolic tracking
        self._total_energy: float = 0.0
        self._energy_cap: float = 10.0  # Maximum total energy (entropy control)
    
    # =====================
    # Node Operations
    # =====================
    
    def add_node(
        self,
        node_id: str,
        energy: float = 0.0,
        threshold: float = 0.75,
        novelty_score: float = 1.0,
        cluster: ClusterType = ClusterType.HIDDEN
    ) -> ConceptNode:
        """Add a new node to the graph. O(1)."""
        if node_id in self.nodes:
            return self.nodes[node_id]
        
        node = ConceptNode(
            id=node_id,
            energy=energy,
            threshold=threshold,
            novelty_score=novelty_score,
            cluster=cluster
        )
        
        self.nodes[node_id] = node
        self._forward_edges[node_id] = []
        self._backward_edges[node_id] = []
        
        # Add to cluster index
        self._clusters[cluster].add(node_id)
        
        if energy > 0:
            self._active_nodes.add(node_id)
            self._total_energy += energy
        
        return node
    
    def get_node(self, node_id: str) -> Optional[ConceptNode]:
        """Retrieve a node by ID. O(1)."""
        return self.nodes.get(node_id)
    
    def has_node(self, node_id: str) -> bool:
        """Check if node exists. O(1)."""
        return node_id in self.nodes
    
    def get_cluster_nodes(self, cluster: ClusterType) -> Set[str]:
        """Get all node IDs in a specific cluster."""
        return self._clusters.get(cluster, set()).copy()
    
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
    ) -> Synapse:
        """Add a synapse between two nodes. O(1)."""
        # Ensure nodes exist
        if source_id not in self.nodes:
            self.add_node(source_id)
        if target_id not in self.nodes:
            self.add_node(target_id)
        
        # Check for existing synapse
        for synapse in self._forward_edges[source_id]:
            if synapse.target_id == target_id and synapse.type == type:
                return synapse
        
        synapse = Synapse(
            target_id=target_id,
            type=type,
            weight=weight,
            confidence=confidence
        )
        
        self._forward_edges[source_id].append(synapse)
        self._backward_edges[target_id].append((source_id, synapse))
        
        return synapse
    
    def get_outgoing(self, node_id: str) -> List[Synapse]:
        """Get all outgoing synapses from a node. O(1)."""
        return self._forward_edges.get(node_id, [])
    
    def get_incoming(self, node_id: str) -> List[Tuple[str, Synapse]]:
        """Get all incoming synapses to a node. O(1)."""
        return self._backward_edges.get(node_id, [])
    
    def get_synapse(
        self, 
        source_id: str, 
        target_id: str, 
        type: Optional[str] = None
    ) -> Optional[Synapse]:
        """Find a specific synapse."""
        for synapse in self._forward_edges.get(source_id, []):
            if synapse.target_id == target_id:
                if type is None or synapse.type == type:
                    return synapse
        return None
    
    # =====================
    # Trace Management
    # =====================
    
    def mark_synapse_traced(self, source_id: str, target_id: str):
        """Mark a synapse as having non-zero trace."""
        self._traced_synapses.add((source_id, target_id))
    
    def unmark_synapse_traced(self, source_id: str, target_id: str):
        """Remove synapse from traced set (when trace decays to 0)."""
        self._traced_synapses.discard((source_id, target_id))
    
    def get_traced_synapses(self) -> List[Tuple[str, Synapse]]:
        """
        Get all synapses with non-zero eligibility trace.
        
        This enables efficient reward broadcast without iterating
        over all synapses in the graph.
        """
        result = []
        for source_id, target_id in self._traced_synapses:
            synapse = self.get_synapse(source_id, target_id)
            if synapse and synapse.trace > 0:
                result.append((source_id, synapse))
        return result
    
    def decay_all_traces(self, decay_rate: float = 0.95):
        """
        Decay all eligibility traces in the graph.
        
        Removes synapses from traced set when trace reaches 0.
        """
        to_remove = []
        for source_id, target_id in self._traced_synapses:
            synapse = self.get_synapse(source_id, target_id)
            if synapse:
                synapse.decay_trace(decay_rate)
                if synapse.trace == 0:
                    to_remove.append((source_id, target_id))
        
        for key in to_remove:
            self._traced_synapses.discard(key)
    
    # =====================
    # Activity Tracking
    # =====================
    
    def mark_active(self, node_id: str):
        """Mark a node as active (has non-zero energy)."""
        node = self.nodes.get(node_id)
        if node:
            self._active_nodes.add(node_id)
    
    def mark_inactive(self, node_id: str):
        """Mark a node as inactive."""
        self._active_nodes.discard(node_id)
    
    def get_active_nodes(self) -> Set[str]:
        """Get the set of currently active nodes."""
        return self._active_nodes.copy()
    
    def is_active(self, node_id: str) -> bool:
        """Check if a node is currently active."""
        return node_id in self._active_nodes
    
    def update_active_set(self):
        """Refresh the active set based on current energies."""
        self._active_nodes.clear()
        self._total_energy = 0.0
        
        for node_id, node in self.nodes.items():
            if node.energy > 0:
                self._active_nodes.add(node_id)
                self._total_energy += node.energy
    
    # =====================
    # Metabolic Tracking
    # =====================
    
    def get_total_energy(self) -> float:
        """Get total system energy (for entropy monitoring)."""
        return self._total_energy
    
    def update_total_energy(self):
        """Recalculate total system energy."""
        self._total_energy = sum(
            node.energy for node in self.nodes.values()
        )
        return self._total_energy
    
    def get_cluster_energy(self, cluster: ClusterType) -> float:
        """Get total energy in a specific cluster."""
        total = 0.0
        for node_id in self._clusters.get(cluster, set()):
            node = self.nodes.get(node_id)
            if node:
                total += node.energy
        return total
    
    # =====================
    # Statistics
    # =====================
    
    @property
    def node_count(self) -> int:
        """Total number of nodes."""
        return len(self.nodes)
    
    @property
    def edge_count(self) -> int:
        """Total number of synapses."""
        return sum(len(edges) for edges in self._forward_edges.values())
    
    @property
    def traced_count(self) -> int:
        """Number of synapses with active traces."""
        return len(self._traced_synapses)
    
    def clear(self):
        """Clear all nodes and edges."""
        self.nodes.clear()
        self._forward_edges.clear()
        self._backward_edges.clear()
        self._active_nodes.clear()
        self._traced_synapses.clear()
        for cluster in self._clusters.values():
            cluster.clear()
        self._total_energy = 0.0
    
    def get_stats(self) -> Dict:
        """Get comprehensive memory statistics."""
        return {
            "nodes": self.node_count,
            "edges": self.edge_count,
            "active": len(self._active_nodes),
            "traced": self.traced_count,
            "total_energy": self._total_energy,
            "clusters": {
                c.value: len(self._clusters[c]) 
                for c in ClusterType
            }
        }


@dataclass
class EventSchema:
    """
    Schema definition for an action (System 2 rule).
    
    Used for validation and expectation generation.
    Added for compatibility with System 2 dialogue intervention.
    """
    action: str
    constraints: Dict[str, List[str]] = field(default_factory=dict)  # role -> [required_properties]
    effects: Dict[str, List[str]] = field(default_factory=dict)      # role -> [added_properties]
    id: str = field(init=False)
    confidence: float = 1.0
    
    def __post_init__(self):
        self.id = f"schema_{self.action}"

    def validate(self, role: str, properties: Set[str]) -> bool:
        """Check if properties satisfy constraints for a role."""
        if role not in self.constraints:
            return True
        required = set(self.constraints[role])
        return required.issubset(properties)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'EventSchema':
        """Create schema from dictionary (v5 compatibility)."""
        schema = cls(
            action=data.get('action', ''),
            constraints=data.get('constraints', {}),
            effects=data.get('effects', {}),
            confidence=data.get('confidence', 1.0)
        )
        return schema

