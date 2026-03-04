from typing import List, Tuple, Optional
import numpy as np
from python.core.vsa.hypervec_shim import HyperVector, SocietalHNSW as RustHNSW

class SocietalHNSW:
    """
    Python wrapper for the CPT-anchored SocietalHNSW Rust implementation.
    If Rust backend is not available, falls back to a minimal Python linear search.
    """
    def __init__(self, dim: int = 10240):
        self.dim = dim
        self._rust_backend = None
        if RustHNSW is not None:
            try:
                # The Rust SocietalHNSW might expect dim in its constructor
                self._rust_backend = RustHNSW(dim)
            except TypeError:
                # Or it might take no arguments
                self._rust_backend = RustHNSW()
        
        self._py_nodes = {}
        self._py_layers = {}
            
    def insert_node(self, concept_id: str, layer: int, vector: HyperVector):
        """
        Insert a node into the hierarchical structure.
        Uses Central Place Theory to assign higher layers to more central "hub" concepts.
        """
        try:
            if self._rust_backend is not None:
                # Rust requires raw list of floats
                # In shim, we'll ensure HyperVector has to_gradient_input()
                if hasattr(vector, "to_gradient_input"):
                    raw_floats = vector.to_gradient_input().tolist()
                else:
                    # Fallback bipolar conversion
                    bits = vector.bits if hasattr(vector, "bits") else np.zeros(self.dim)
                    raw_floats = ((bits.astype(np.float32) - 0.5) * 2).tolist()
                
                self._rust_backend.insert_node(concept_id, max(0, min(255, layer)), raw_floats)
            else:
                self._py_nodes[concept_id] = vector
                self._py_layers[concept_id] = layer
        except Exception:
            # Fallback for vectors that lack float conversion or Rust errors
            self._py_nodes[concept_id] = vector
            self._py_layers[concept_id] = layer
            
    def search(self, query: HyperVector, k: int = 10, ef: int = 50) -> List[Tuple[str, float]]:
        """
        Perform approximate nearest neighbor search through the Small World graph.
        """
        if self._rust_backend is not None:
            try:
                if hasattr(query, "to_gradient_input"):
                    raw_floats = query.to_gradient_input().tolist()
                else:
                    bits = query.bits if hasattr(query, "bits") else np.zeros(self.dim)
                    raw_floats = ((bits.astype(np.float32) - 0.5) * 2).tolist()
                
                results = self._rust_backend.search(raw_floats, k, ef)
                return results # List of (concept_id, distance)
            except Exception:
                pass
                
        # Pure Python Fallback (Brute force exact search instead of ANN)
        if not self._py_nodes:
            return []
            
        distances = []
        for cid, hv in self._py_nodes.items():
            # Use cosine similarity for robustness
            if hasattr(query, "similarity_robust"):
                sim = query.similarity_robust(hv, method='cosine')
            else:
                # basic Hamming fallback
                sim = query.similarity(hv)
                
            dist = 1.0 - sim
            distances.append((cid, dist))
            
        distances.sort(key=lambda x: x[1])
        return distances[:k]
