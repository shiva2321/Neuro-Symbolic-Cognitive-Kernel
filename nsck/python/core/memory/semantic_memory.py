"""
NSCK Semantic Memory Module (Phase 3.1)
=======================================
Structured knowledge base of concepts and relations.
Implements abstracted knowledge (Schemas) and spreading activation.
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import python.core.vsa.hypervec_shim as hypervec_rs

class SemanticMemory:
    """
    Structured knowledge base of concepts and relations.
    Built from episodic memory consolidation during sleep.
    
    Automatically uses Rust SemanticMemoryConcurrent when available for 10-100x speedup.
    Falls back to Python implementation transparently.
    """
    
    # Default relation weights for spreading activation.
    # Higher weight = stronger propagation through that edge type.
    DEFAULT_RELATION_WEIGHTS: Dict[str, float] = {
        "is_a": 0.9,         # Taxonomic links carry most meaning
        "has_property": 0.7,  # Properties propagate strongly
        "causes": 0.6,       # Causal links weaker than taxonomic
        "part_of": 0.5,      # Mereological
        "similar_to": 0.4,   # Weakest — associative
    }
    
    def __init__(self, relation_weights: Optional[Dict[str, float]] = None, use_rust: bool = True):
        # Try to use Rust backend if available and requested
        self._rust_backend = None
        if use_rust and hypervec_rs.SemanticMemoryConcurrent is not None:
            try:
                self._rust_backend = hypervec_rs.SemanticMemoryConcurrent()
                print("SemanticMemory Initialized with Rust backend (concurrent, optimized).")
            except Exception as e:
                print(f"[WARNING] Failed to initialize Rust backend: {e}")
                print("Falling back to Python implementation.")
        else:
            print("SemanticMemory Initialized with Python backend.")
        
        # Graph database of concepts (always maintained for graph operations)
        self.concept_graph = nx.DiGraph()
        
        # Concept -> HyperVector mapping (Python fallback)
        self.concept_hvs: Dict[str, hypervec_rs.HyperVector] = {}
        
        # Relation types (extensible — new types registered on first use)
        self.relations = list(self.DEFAULT_RELATION_WEIGHTS.keys())
        
        # Configurable relation weights for spreading activation
        self.relation_weights: Dict[str, float] = dict(self.DEFAULT_RELATION_WEIGHTS)
        if relation_weights:
            self.relation_weights.update(relation_weights)
    
    def reset(self):
        """Clear all semantic knowledge and re-initialize."""
        self.concept_graph = nx.DiGraph()
        self.concept_hvs = {}
        print("[SEMANTIC] Memory reset complete.")
    
    def add_concept(self, concept_name: str, properties: Dict[str, Any], hv_override: Optional[hypervec_rs.HyperVector] = None):
        """
        Add a new concept to semantic memory.
        
        Args:
            concept_name: Name of the concept (e.g. "Apple")
            properties: Dictionary of properties
            hv_override: Optional pre-calculated hypervector
        """
        if hv_override:
            hv = hv_override
        else:
            # Generate base hypervector for concept name
            hv = hypervec_rs.HyperVector(hash(concept_name) % (2**32))
            
            # Bind properties into concept HV
            for prop, value in properties.items():
                prop_hv = hypervec_rs.HyperVector(hash(prop) % (2**32))
                value_hv = hypervec_rs.HyperVector(hash(str(value)) % (2**32))
                # Role-filler binding: XOR the property role with value, then bundle into concept
                bound = prop_hv.xor(value_hv)
                hv = hv.bundle(bound)
        
        # Store in both backends
        self.concept_hvs[concept_name] = hv
        if self._rust_backend is not None:
            try:
                self._rust_backend.add_concept(concept_name, hv)
            except Exception as e:
                print(f"[WARNING] Rust backend add_concept failed: {e}")
        
        self.concept_graph.add_node(concept_name, **properties)
    
    def add_relation(self, concept1: str, relation: str, concept2: str, timestamp: float = 0.0):
        """
        Add relation between concepts with temporal validation.
        Only updates if the new information is more recent or same time.
        """
        if concept1 not in self.concept_graph or concept2 not in self.concept_graph:
            return
            
        # Check if edge exists
        if self.concept_graph.has_edge(concept1, concept2):
            existing_data = self.concept_graph.get_edge_data(concept1, concept2)
            existing_time = existing_data.get('timestamp', 0.0)
            
            # If explicit timestamp provided and it's older than existing knowledge, IGNORE
            if timestamp > 0 and timestamp < existing_time:
                # [Belief Revision] Reject outdated info
                return

        # Update or create edge
        self.concept_graph.add_edge(concept1, concept2, relation=relation, timestamp=timestamp)
    
    def query(self, query_hv: hypervec_rs.HyperVector, k: int = 5) -> List[Tuple[str, float]]:
        """
        Find concepts most similar to query HV.
        Uses Rust parallel search when available (10-100x faster), falls back to Python.
        """
        # Try Rust backend first
        if self._rust_backend is not None and hasattr(self._rust_backend, 'parallel_semantic_search'):
            try:
                return self._rust_backend.parallel_semantic_search(query_hv, k)
            except Exception as e:
                print(f"[WARNING] Rust backend query failed: {e}, falling back to Python")
        
        # Python fallback
        similarities = []
        for concept_name, concept_hv in self.concept_hvs.items():
            # Use robust cosine similarity instead of legacy Hamming
            sim = query_hv.cosine_similarity(concept_hv) if hasattr(query_hv, 'cosine_similarity') else query_hv.similarity(concept_hv)
            similarities.append((concept_name, sim))
        
        # Return top-k most similar
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def spread_activation(self, start_concepts: List[str], steps: int = 3, decay: float = 0.7) -> Dict[str, float]:
        """
        Spreading activation for associative retrieval.
        
        Propagation strength is modulated by relation type weights,
        so ``is_a`` edges (0.9) carry activation more strongly than
        ``similar_to`` edges (0.4).
        
        Returns activation levels for nodes.
        """
        activation = {c: 1.0 for c in start_concepts if c in self.concept_graph}
        
        # Safety cap: only propagate from the top-N most activated nodes
        # each step to prevent exponential blow-up on large graphs.
        _MAX_FRONTIER = 200
        
        for _ in range(steps):
            new_activation = activation.copy()

            # Only spread from the highest-activated nodes to bound work
            frontier = sorted(
                ((c, a) for c, a in activation.items() if a >= 0.01),
                key=lambda x: x[1], reverse=True)[:_MAX_FRONTIER]

            for concept, act in frontier:
                # Spread to neighbors with relation-weighted strength
                for _, neighbor, data in self.concept_graph.out_edges(concept, data=True):
                    rel = data.get("relation", "similar_to")
                    edge_weight = self.relation_weights.get(rel, 0.3)
                    spread_val = act * decay * edge_weight
                    new_activation[neighbor] = new_activation.get(neighbor, 0.0) + spread_val
            
            activation = new_activation
            
        return activation
    
    def get_inherited_properties(self, concept: str) -> Dict[str, Any]:
        """
        Get properties of a concept including those inherited via ``is_a`` edges.
        
        Walks the ``is_a`` hierarchy upward and merges properties, with the
        most specific (closest) concept's properties taking precedence.
        
        Args:
            concept: The concept to look up
            
        Returns:
            Merged properties dict (own + inherited)
        """
        if concept not in self.concept_graph:
            return {}
        
        # Collect all ancestors via is_a (BFS)
        visited = set()
        queue = [concept]
        ancestor_chain = []  # ordered from most specific to most general
        
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            ancestor_chain.append(current)
            
            # Follow is_a edges upward
            for _, parent, data in self.concept_graph.out_edges(current, data=True):
                if data.get("relation") == "is_a" and parent not in visited:
                    queue.append(parent)
        
        # Merge properties from general to specific (specific wins)
        merged = {}
        for ancestor in reversed(ancestor_chain):
            props = dict(self.concept_graph.nodes[ancestor])
            merged.update(props)
        
        return merged
    
    def extract_schema(self, concept: str) -> Dict[str, Any]:
        """Extract abstracted structure (Schema) for a concept."""
        if concept not in self.concept_graph:
            return {}
        
        schema = {
            "name": concept,
            "properties": dict(self.concept_graph.nodes[concept]),
            "is_a": [],
            "parts": [],
            "causes": [],
            "caused_by": []
        }
        
        # Outgoing relations
        for _, neighbor, data in self.concept_graph.out_edges(concept, data=True):
            rel = data.get("relation")
            if rel == "is_a": schema["is_a"].append(neighbor)
            elif rel == "part_of": schema["parts"].append(neighbor)
            elif rel == "causes": schema["causes"].append(neighbor)
            
        # Incoming relations
        for neighbor, _, data in self.concept_graph.in_edges(concept, data=True):
            rel = data.get("relation")
            if rel == "causes": schema["caused_by"].append(neighbor)
            
        return schema
