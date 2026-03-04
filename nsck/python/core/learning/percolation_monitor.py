from typing import Tuple, List, Dict
from python.core.vsa.hypervec_shim import PercolationDetector as RustPercolation

class PercolationMonitor:
    """
    Monitors a semantic graph for physical percolation (phase transition).
    When the largest connected component exceeds the percolation threshold,
    it identifies the emergence of a macroscopic Knowledge Domain.
    """
    
    def __init__(self, threshold: float = 0.65):
        """
        threshold: The similarity threshold above which two concepts are "connected" in the percolation graph.
        """
        self.threshold = threshold
        self._rust_backend = None
        if RustPercolation is not None:
            self._rust_backend = RustPercolation(self.threshold)
            
    def check_transition(self, num_nodes: int, edges: List[Tuple[int, int, float]]) -> Tuple[bool, int, Dict[int, int], List[int]]:
        """
        Check if the semantic space has percolated.
        
        Returns:
            (has_percolated, max_component_size, component_sizes_map, node_to_root_map)
        """
        if num_nodes == 0:
            return False, 0, {}, []
            
        if self._rust_backend is not None:
            # Assume rust backend returns (bool, usize, dict, list)
            return self._rust_backend.check_transition(num_nodes, edges)
            
        # Pure Python Fallback utilizing Union-Find
        parent = list(range(num_nodes))
        size = [1] * num_nodes
        
        def find(i):
            root = i
            while root != parent[root]:
                parent[root] = parent[parent[root]]
                root = parent[root]
            return root
            
        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                if size[root_i] < size[root_j]:
                    parent[root_i] = root_j
                    size[root_j] += size[root_i]
                else:
                    parent[root_j] = root_i
                    size[root_i] += size[root_j]
        
        for u, v, weight in edges:
            if weight > self.threshold:
                union(u, v)
                
        # Final pass to flatten parents and aggregate sizes
        node_to_root = [find(i) for i in range(num_nodes)]
        comp_sizes = {}
        for root in node_to_root:
            comp_sizes[root] = comp_sizes.get(root, 0) + 1
                
        max_size = max(comp_sizes.values()) if comp_sizes else 0
        percolated = max_size >= (num_nodes / 2) and num_nodes > 1
        
        return percolated, max_size, comp_sizes, node_to_root
