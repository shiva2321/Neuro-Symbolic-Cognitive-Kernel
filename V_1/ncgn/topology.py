"""
NCGN v7 Topology Module

Rustworkx-based graph topology with integer indexing.
Thread-safe bidirectional label↔index mapping.

CRITICAL: Rustworkx uses integer indices only. The IndexRegistry
maintains perfect synchronization between labels and indices.
"""

import threading
from typing import Dict, List, Tuple, Optional, Set
import numpy as np

try:
    import rustworkx as rx
    RUSTWORKX_AVAILABLE = True
except ImportError:
    RUSTWORKX_AVAILABLE = False
    print("WARNING: rustworkx not available. Install with: pip install rustworkx")


class IndexRegistry:
    """
    Bidirectional mapping: label (str) <-> index (int).
    
    Thread-safe for concurrent System 1/System 2 access.
    
    CRITICAL: This registry MUST stay perfectly synchronized with
    the Rustworkx graph. When nodes are deleted, Rustworkx leaves
    "holes" and reuses those indices for new nodes.
    """
    
    def __init__(self):
        self._label_to_idx: Dict[str, int] = {}
        self._idx_to_label: Dict[int, str] = {}
        self._lock = threading.Lock()
    
    def register(self, label: str, idx: int) -> None:
        """
        Atomically register a new label↔index mapping.
        
        Args:
            label: The concept label (string)
            idx: The Rustworkx node index (integer)
            
        Raises:
            ValueError: If label already exists
        """
        with self._lock:
            if label in self._label_to_idx:
                raise ValueError(f"Label '{label}' already registered at index {self._label_to_idx[label]}")
            self._label_to_idx[label] = idx
            self._idx_to_label[idx] = label
    
    def unregister(self, idx: int) -> Optional[str]:
        """
        Remove mapping by index.
        
        Args:
            idx: The index to unregister
            
        Returns:
            The removed label, or None if not found
        """
        with self._lock:
            if idx not in self._idx_to_label:
                return None
            label = self._idx_to_label.pop(idx)
            del self._label_to_idx[label]
            return label
    
    def unregister_by_label(self, label: str) -> Optional[int]:
        """
        Remove mapping by label.
        
        Args:
            label: The label to unregister
            
        Returns:
            The removed index, or None if not found
        """
        with self._lock:
            if label not in self._label_to_idx:
                return None
            idx = self._label_to_idx.pop(label)
            del self._idx_to_label[idx]
            return idx
    
    def get_index(self, label: str) -> Optional[int]:
        """Get index for a label. O(1)."""
        return self._label_to_idx.get(label)
    
    def get_label(self, idx: int) -> Optional[str]:
        """Get label for an index. O(1)."""
        return self._idx_to_label.get(idx)
    
    def has_label(self, label: str) -> bool:
        """Check if label exists."""
        return label in self._label_to_idx
    
    def has_index(self, idx: int) -> bool:
        """Check if index exists."""
        return idx in self._idx_to_label
    
    def all_labels(self) -> List[str]:
        """Get all registered labels."""
        with self._lock:
            return list(self._label_to_idx.keys())
    
    def all_indices(self) -> List[int]:
        """Get all registered indices."""
        with self._lock:
            return list(self._idx_to_label.keys())
    
    def max_index(self) -> int:
        """Get the maximum registered index, or -1 if empty."""
        with self._lock:
            if not self._idx_to_label:
                return -1
            return max(self._idx_to_label.keys())
    
    def __len__(self) -> int:
        return len(self._label_to_idx)
    
    def __repr__(self) -> str:
        return f"IndexRegistry({len(self)} labels)"


class GraphTopology:
    """
    Wrapper around rustworkx.PyDiGraph with automatic index tracking.
    
    Features:
    - Automatic index registration on add
    - Dirty flag for CSR cache invalidation
    - Edge data extraction for sparse matrix construction
    
    CRITICAL: When nodes are deleted, Rustworkx leaves "holes" and 
    reuses those indices for new nodes. Callers MUST zero the state 
    arrays at deleted indices immediately using CognitiveState.zero_index().
    """
    
    def __init__(self):
        if not RUSTWORKX_AVAILABLE:
            raise ImportError("rustworkx is required for GraphTopology. Install with: pip install rustworkx")
        
        self.graph: rx.PyDiGraph = rx.PyDiGraph()
        self.registry = IndexRegistry()
        self._dirty = False  # Set True when topology changes
        self._edge_count = 0
    
    def add_concept(self, label: str) -> int:
        """
        Add a new concept node.
        
        Args:
            label: The concept label (string)
            
        Returns:
            The integer index assigned by Rustworkx.
            
        Note:
            If the label already exists, returns the existing index
            without modifying the graph.
        """
        existing_idx = self.registry.get_index(label)
        if existing_idx is not None:
            return existing_idx
        
        # Add to Rustworkx - returns the integer index
        idx = self.graph.add_node(label)
        self.registry.register(label, idx)
        self._dirty = True
        return idx
    
    def remove_concept(self, label: str) -> Optional[int]:
        """
        Remove a concept node.
        
        Args:
            label: The concept label to remove
            
        Returns:
            The index of removed node, or None if not found.
        
        CALLER MUST: Zero the state arrays at this index!
        Use CognitiveState.zero_index(idx) immediately after calling this.
        """
        idx = self.registry.get_index(label)
        if idx is None:
            return None
        
        # Remove from Rustworkx
        self.graph.remove_node(idx)
        self.registry.unregister(idx)
        self._dirty = True
        return idx
    
    def remove_concept_by_index(self, idx: int) -> Optional[str]:
        """
        Remove a concept node by index.
        
        Args:
            idx: The index to remove
            
        Returns:
            The label of removed node, or None if not found.
        
        CALLER MUST: Zero the state arrays at this index!
        """
        label = self.registry.get_label(idx)
        if label is None:
            return None
        
        self.graph.remove_node(idx)
        self.registry.unregister(idx)
        self._dirty = True
        return label
    
    def add_connection(
        self, 
        source: str, 
        target: str, 
        weight: float = 0.5
    ) -> Tuple[int, int]:
        """
        Add weighted edge between concepts.
        Creates nodes if they don't exist.
        
        Args:
            source: Source concept label
            target: Target concept label
            weight: Edge weight (default 0.5)
            
        Returns:
            Tuple of (source_idx, target_idx)
        """
        src_idx = self.add_concept(source)
        tgt_idx = self.add_concept(target)
        
        # Check if edge already exists
        if self.graph.has_edge(src_idx, tgt_idx):
            # Update existing edge weight
            self.graph.update_edge(src_idx, tgt_idx, weight)
        else:
            self.graph.add_edge(src_idx, tgt_idx, weight)
            self._edge_count += 1
        
        self._dirty = True
        return (src_idx, tgt_idx)
    
    def remove_connection(self, source: str, target: str) -> bool:
        """
        Remove edge between concepts.
        
        Returns:
            True if edge was removed, False if not found.
        """
        src_idx = self.registry.get_index(source)
        tgt_idx = self.registry.get_index(target)
        
        if src_idx is None or tgt_idx is None:
            return False
        
        if not self.graph.has_edge(src_idx, tgt_idx):
            return False
        
        self.graph.remove_edge(src_idx, tgt_idx)
        self._edge_count -= 1
        self._dirty = True
        return True
    
    def get_edge_weight(self, source: str, target: str) -> Optional[float]:
        """Get the weight of an edge."""
        src_idx = self.registry.get_index(source)
        tgt_idx = self.registry.get_index(target)
        
        if src_idx is None or tgt_idx is None:
            return None
        
        if not self.graph.has_edge(src_idx, tgt_idx):
            return None
        
        return self.graph.get_edge_data(src_idx, tgt_idx)
    
    def set_edge_weight(self, source: str, target: str, weight: float) -> bool:
        """Update the weight of an existing edge."""
        src_idx = self.registry.get_index(source)
        tgt_idx = self.registry.get_index(target)
        
        if src_idx is None or tgt_idx is None:
            return False
        
        if not self.graph.has_edge(src_idx, tgt_idx):
            return False
        
        self.graph.update_edge(src_idx, tgt_idx, weight)
        self._dirty = True
        return True
    
    def get_neighbors(self, label: str) -> List[str]:
        """Get labels of all outgoing neighbors."""
        idx = self.registry.get_index(label)
        if idx is None:
            return []
        
        neighbor_indices = self.graph.successor_indices(idx)
        return [
            self.registry.get_label(n) 
            for n in neighbor_indices 
            if self.registry.get_label(n) is not None
        ]
    
    def get_predecessors(self, label: str) -> List[str]:
        """Get labels of all incoming neighbors."""
        idx = self.registry.get_index(label)
        if idx is None:
            return []
        
        pred_indices = self.graph.predecessor_indices(idx)
        return [
            self.registry.get_label(p) 
            for p in pred_indices 
            if self.registry.get_label(p) is not None
        ]
    
    def get_adjacency_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract edge data for sparse matrix construction.
        
        Returns:
            Tuple of (sources, targets, weights) as numpy arrays.
            
        Usage:
            sources, targets, weights = topology.get_adjacency_data()
            coo = scipy.sparse.coo_matrix((weights, (sources, targets)), shape=(N, N))
            csr = coo.tocsr()
        """
        edge_list = self.graph.weighted_edge_list()
        
        if not edge_list:
            return (
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32),
                np.array([], dtype=np.float32)
            )
        
        sources = np.array([e[0] for e in edge_list], dtype=np.int32)
        targets = np.array([e[1] for e in edge_list], dtype=np.int32)
        weights = np.array([e[2] for e in edge_list], dtype=np.float32)
        
        return sources, targets, weights
    
    def get_subgraph_description(self, labels: List[str], max_depth: int = 1) -> str:
        """
        Get text description of subgraph for LLM context.
        
        Args:
            labels: Center concepts
            max_depth: How many hops to include
            
        Returns:
            Human-readable subgraph description
        """
        lines = []
        visited = set()
        
        def describe_node(label: str, depth: int):
            if depth > max_depth or label in visited:
                return
            visited.add(label)
            
            neighbors = self.get_neighbors(label)
            for neighbor in neighbors:
                weight = self.get_edge_weight(label, neighbor)
                lines.append(f"  {label} --({weight:.2f})--> {neighbor}")
                describe_node(neighbor, depth + 1)
        
        for label in labels:
            if self.registry.has_label(label):
                lines.append(f"[{label}]")
                describe_node(label, 0)
        
        return "\n".join(lines) if lines else "(empty subgraph)"
    
    @property
    def is_dirty(self) -> bool:
        """Check if topology changed since last matrix sync."""
        return self._dirty
    
    def mark_clean(self) -> None:
        """Mark topology as synchronized with matrix."""
        self._dirty = False
    
    @property
    def num_nodes(self) -> int:
        """Number of nodes in the graph."""
        return self.graph.num_nodes()
    
    @property
    def num_edges(self) -> int:
        """Number of edges in the graph."""
        return self.graph.num_edges()
    
    def clear(self) -> None:
        """Remove all nodes and edges."""
        self.graph = rx.PyDiGraph()
        self.registry = IndexRegistry()
        self._dirty = True
        self._edge_count = 0
    
    def get_stats(self) -> Dict:
        """Get graph statistics."""
        return {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "max_index": self.registry.max_index(),
            "is_dirty": self._dirty,
        }
    
    def __repr__(self) -> str:
        return f"GraphTopology(nodes={self.num_nodes}, edges={self.num_edges})"
