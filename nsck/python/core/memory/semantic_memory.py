"""
NSCK Semantic Memory Module (Phase 3.1)
=======================================
Structured knowledge base of concepts and relations.
Implements abstracted knowledge (Schemas) and spreading activation.
"""

import logging
import networkx as nx
import numpy as np
import pickle
import os
from typing import Dict, List, Any, Optional, Tuple
import python.core.vsa.hypervec_shim as hypervec_rs

# V3: optional HNSW index
try:
    import hnswlib as _hnswlib
    _HNSWLIB_AVAILABLE = True
except ImportError:
    _hnswlib = None
    _HNSWLIB_AVAILABLE = False

logger = logging.getLogger("nsck.semantic_memory")

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
        "is_a": 0.9,            # Taxonomic links carry most meaning
        "has_property": 0.7,    # Properties propagate strongly
        "causes": 0.6,          # Causal links
        "leads_to": 0.6,        # Consequential (alias for causes)
        "results_in": 0.6,      # Result of action
        "implies": 0.55,        # Logical implication
        "part_of": 0.5,         # Mereological
        "similar_to": 0.4,      # Associative
        "semantically_related": 0.35,  # Weakest — co-occurrence based
    }
    
    def __init__(self, relation_weights: Optional[Dict[str, float]] = None, use_rust: bool = True,
                 config=None):
        self._config = config
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

        # V3: Stigmergy — edge-level pheromone strengths
        self._stigmergy: Dict[Tuple[str, str], float] = {}

        # V3: HNSW approximate nearest-neighbour index
        self._hnsw_index = None
        self._hnsw_id_to_concept: List[str] = []
        self._hnsw_dim: int = 0
        self._hnsw_enabled: bool = False
        if _HNSWLIB_AVAILABLE and config is not None and getattr(config, 'enable_hnsw_index', False):
            self._hnsw_enabled = True
            logger.info("[SEMANTIC] HNSW index enabled (hnswlib available)")
        elif getattr(config, 'enable_hnsw_index', False) and not _HNSWLIB_AVAILABLE:
            logger.warning("[SEMANTIC] enable_hnsw_index=True but hnswlib not installed; "
                           "falling back to linear scan. Install with: pip install hnswlib")

    def _init_hnsw(self, dim: int):
        """Lazily initialise the HNSW index once the HV dimension is known."""
        if not self._hnsw_enabled or _hnswlib is None:
            return
        try:
            idx = _hnswlib.Index(space='cosine', dim=dim)
            idx.init_index(max_elements=100_000, ef_construction=200, M=16)
            idx.set_ef(50)
            self._hnsw_index = idx
            self._hnsw_dim = dim
            self._hnsw_id_to_concept = []
        except Exception as e:
            logger.warning("[SEMANTIC] HNSW init failed: %s; falling back to linear scan", e)
            self._hnsw_enabled = False

    def reset(self):
        """Clear all semantic knowledge and re-initialize."""
        self.concept_graph = nx.DiGraph()
        self.concept_hvs = {}
        self._stigmergy = {}
        self._hnsw_index = None
        self._hnsw_id_to_concept = []
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

        # V3: Incremental concept refinement — if concept exists, blend HVs
        if (concept_name in self.concept_hvs and
                self._config is not None and
                getattr(self._config, 'enable_incremental_concept_refinement', False)):
            existing_hv = self.concept_hvs[concept_name]
            # 90% old, 10% new: bundle 9 copies of old + 1 copy of new
            blended = existing_hv
            for _ in range(9):
                blended = blended.bundle(existing_hv)
            blended = blended.bundle(hv)
            hv = blended
        
        # Store in both backends
        self.concept_hvs[concept_name] = hv
        if self._rust_backend is not None:
            try:
                self._rust_backend.add_concept(concept_name, hv)
            except Exception as e:
                print(f"[WARNING] Rust backend add_concept failed: {e}")
        
        self.concept_graph.add_node(concept_name, **properties)

        # V3: Insert into HNSW index if enabled
        if self._hnsw_enabled:
            try:
                bits = np.asarray(hv.bits, dtype=np.float32)
                dim = len(bits)
                if self._hnsw_index is None:
                    self._init_hnsw(dim)
                if self._hnsw_index is not None:
                    idx = len(self._hnsw_id_to_concept)
                    self._hnsw_id_to_concept.append(concept_name)
                    self._hnsw_index.add_items(bits.reshape(1, -1), [idx])
            except Exception as e:
                logger.debug("[SEMANTIC] HNSW add failed: %s", e)
    
    def get_concept(self, concept_name: str) -> Optional[hypervec_rs.HyperVector]:
        """Retrieve the hypervector for a given concept."""
        return self.concept_hvs.get(concept_name)
    
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
        Uses HNSW when available, then Rust parallel search, then Python fallback.
        """
        # V3: HNSW approximate nearest-neighbour
        if self._hnsw_enabled and self._hnsw_index is not None and self._hnsw_id_to_concept:
            try:
                bits = np.asarray(query_hv.bits, dtype=np.float32).reshape(1, -1)
                n_results = min(k, len(self._hnsw_id_to_concept))
                labels, distances = self._hnsw_index.knn_query(bits, k=n_results)
                results = []
                for label, dist in zip(labels[0], distances[0]):
                    concept = self._hnsw_id_to_concept[label]
                    sim = 1.0 - float(dist)  # cosine distance → similarity
                    results.append((concept, sim))
                return results
            except Exception as e:
                logger.debug("[SEMANTIC] HNSW query failed: %s; falling back", e)

        # Try Rust backend
        if self._rust_backend is not None and hasattr(self._rust_backend, 'parallel_semantic_search'):
            try:
                return self._rust_backend.parallel_semantic_search(query_hv, k)
            except Exception as e:
                print(f"[WARNING] Rust backend query failed: {e}, falling back to Python")
        
        # Python fallback
        similarities = []
        for concept_name, concept_hv in self.concept_hvs.items():
            sim = query_hv.cosine_similarity(concept_hv) if hasattr(query_hv, 'cosine_similarity') else query_hv.similarity(concept_hv)
            similarities.append((concept_name, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def spread_activation(self, start_concepts: List[str], steps: int = 3, decay: float = 0.7) -> Dict[str, float]:
        """
        Spreading activation for associative retrieval.
        
        Propagation strength is modulated by relation type weights,
        so ``is_a`` edges (0.9) carry activation more strongly than
        ``similar_to`` edges (0.4).

        When stigmergy is present, edge weights are additionally boosted
        by their pheromone strength (V3 feature).
        
        Returns activation levels for nodes.
        """
        activation = {c: 1.0 for c in start_concepts if c in self.concept_graph}
        
        _MAX_FRONTIER = 200
        _stigmergy_boost = bool(self._stigmergy)
        
        for _ in range(steps):
            new_activation = activation.copy()

            frontier = sorted(
                ((c, a) for c, a in activation.items() if a >= 0.01),
                key=lambda x: x[1], reverse=True)[:_MAX_FRONTIER]

            for concept, act in frontier:
                for _, neighbor, data in self.concept_graph.out_edges(concept, data=True):
                    rel = data.get("relation", "similar_to")
                    edge_weight = self.relation_weights.get(rel, 0.3)
                    # V3: boost by stigmergy pheromone if present
                    if _stigmergy_boost:
                        stig = self._stigmergy.get((concept, neighbor), 0.0)
                        edge_weight = edge_weight * (1.0 + stig)
                    spread_val = act * decay * edge_weight
                    new_activation[neighbor] = new_activation.get(neighbor, 0.0) + spread_val
            
            activation = new_activation
            
        return activation

    # V3: Stigmergy methods

    def mark_path(self, path: List[str], reward: float = 1.0):
        """Increment stigmergy pheromone on consecutive edges in path."""
        for i in range(len(path) - 1):
            key = (path[i], path[i + 1])
            self._stigmergy[key] = self._stigmergy.get(key, 0.0) + reward

    def evaporate_stigmergy(self, decay_rate: float = 0.99):
        """Decay all stigmergy values (call during sleep)."""
        keys_to_remove = []
        for key in list(self._stigmergy):
            self._stigmergy[key] *= decay_rate
            if self._stigmergy[key] < 1e-6:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del self._stigmergy[key]

    def get_stigmergy(self, concept1: str, concept2: str) -> float:
        """Get stigmergy strength for an edge."""
        return self._stigmergy.get((concept1, concept2), 0.0)
    
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

    def save(self, filepath: str):
        """Save semantic memory to disk via pickle."""
        print(f"[SEMANTIC] Saving memory to {filepath}...")
        data = {
            "concept_graph": self.concept_graph,
            "concept_hvs": self.concept_hvs,
            "relation_weights": self.relation_weights
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        print(f"[SEMANTIC] Saved {len(self.concept_hvs)} concepts.")

    def load(self, filepath: str):
        """Load semantic memory from disk."""
        if not os.path.exists(filepath):
            print(f"[SEMANTIC] No memory file found at {filepath}")
            return
            
        print(f"[SEMANTIC] Loading memory from {filepath}...")
        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)
            
            self.concept_graph = data.get("concept_graph", nx.DiGraph())
            self.concept_hvs = data.get("concept_hvs", {})
            self.relation_weights = data.get("relation_weights", self.DEFAULT_RELATION_WEIGHTS)
            
            # Sync with Rust backend if active
            if self._rust_backend:
                print(f"[SEMANTIC] Syncing {len(self.concept_hvs)} concepts to Rust backend...")
                for name, hv in self.concept_hvs.items():
                    try:
                        self._rust_backend.add_concept(name, hv)
                    except Exception as e:
                        print(f"[WARNING] Rust sync failed for {name}: {e}")
                        
            print(f"[SEMANTIC] Loaded {len(self.concept_hvs)} concepts.")
        except Exception as e:
            print(f"[SEMANTIC] Failed to load memory: {e}")
