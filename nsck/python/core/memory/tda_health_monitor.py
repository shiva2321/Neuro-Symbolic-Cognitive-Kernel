from typing import Dict, List, Tuple
import numpy as np

from python.core.vsa.hypervec_shim import TDARipser as RustTDA
from python.core.memory.societal_knowledge_world import SocietalKnowledgeWorld

class TDAHealthMonitor:
    """
    Topological Data Analysis (TDA) Health Monitor.
    Runs persistent homology (Vietoris-Rips) in the background to detect 
    semantic "holes" (Betti-1) and components (Betti-0) in emergent knowledge domains.
    """
    
    def __init__(self, world: SocietalKnowledgeWorld, check_interval_ticks: int = 500):
        self.world = world
        self.check_interval = check_interval_ticks
        self._rust_backend = None
        
        if RustTDA is not None:
            self._rust_backend = RustTDA()
            
    def run_health_check(self) -> Dict[str, Dict[str, int]]:
        """
        Run TDA on all active Knowledge Domains and update their structural properties.
        """
        results = {}
        
        for domain_id, domain in self.world.domains.items():
            if not domain.neighborhoods:
                continue
                
            # Flatten all members in this domain
            members = []
            for nh in domain.neighborhoods.values():
                members.extend(list(nh.members))
                
            if len(members) < 3:
                continue
                
            # Construct distance matrix
            n = len(members)
            dist_matrix = np.zeros((n, n), dtype=np.float32)
            
            for i in range(n):
                hv_i = self.world.registry[members[i]].hv
                for j in range(i+1, n):
                    hv_j = self.world.registry[members[j]].hv
                    dist = 1.0 - hv_i.similarity_robust(hv_j, method='cosine')
                    dist_matrix[i, j] = dist
                    dist_matrix[j, i] = dist
                    
            betti_0, betti_1 = self._compute_betti_numbers(dist_matrix)
            
            # Store results
            result = {"betti_0": betti_0, "betti_1": betti_1}
            results[domain_id] = result
            
            # Update Domain State
            domain.tda_health_history.append(result)
            
            # Update average neighborhood properties if desired
            for nh in domain.neighborhoods.values():
                nh.betti_0 = betti_0
                nh.betti_1 = betti_1
                
        return results
        
    def _compute_betti_numbers(self, dist_matrix: np.ndarray) -> Tuple[int, int]:
        """
        Compute Betti 0 (connected components) and Betti 1 (holes) from distance matrix.
        Uses Rust backend if available, else a naive static threshold python approximation.
        """
        if self._rust_backend is not None:
            try:
                # TDARipser expects flattened 1D array of the matrix
                flat_dists = dist_matrix.flatten().tolist()
                betti_0, betti_1 = self._rust_backend.compute_betti_numbers(flat_dists, dist_matrix.shape[0])
                return betti_0, betti_1
            except Exception:
                pass
                
        # Pure Python Fallback (Naive Approximation at fixed threshold epsilon=0.5)
        epsilon = 0.5
        n = dist_matrix.shape[0]
        
        # Betti 0 via Union-Find
        parent = list(range(n))
        def find(i):
            if parent[i] == i:
                return i
            parent[i] = find(parent[i])
            return parent[i]
            
        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_i] = root_j
                return True
            return False
            
        components = n
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if dist_matrix[i, j] <= epsilon:
                    edges.append((i, j))
                    if union(i, j):
                        components -= 1
                        
        betti_0 = components
        
        # Betti 1 via naive Euler characteristic proxy for graphs (Cycles = E - V + C)
        betti_1 = max(0, len(edges) - n + betti_0)
        
        return betti_0, betti_1
