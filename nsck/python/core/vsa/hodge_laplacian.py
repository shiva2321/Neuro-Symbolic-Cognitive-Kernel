from typing import Dict, List, Optional
import numpy as np

class HodgeLaplacian:
    """
    Computes higher-order Hodge Laplacians (L0, L1, L2) for concept propagation.
    
    L0: Node-level spreading
    L1: Edge-level flow
    L2: Triadic closure (meaning combination)
    """

    def __init__(self, node_list: List[str]):
        """
        Args:
            node_list: The index mapping to row/cols
        """
        self.node_list = node_list
        self.node_idx = {k: v for v, k in enumerate(node_list)}
        self.n = len(node_list)
        
        # Track edges manually to build boundary matrices later
        self.edges = []
        self.edge_idx = {}
        self.W = np.zeros((self.n, self.n), dtype=np.float32)

    def add_edge(self, u: str, v: str, weight: float):
        if u not in self.node_idx or v not in self.node_idx: return
        i, j = self.node_idx[u], self.node_idx[v]
        self.W[i, j] = weight
        self.W[j, i] = weight
        
        # Store for L1
        e_name = tuple(sorted((u, v)))
        if e_name not in self.edge_idx:
            self.edge_idx[e_name] = len(self.edges)
            self.edges.append(e_name)

    def compute_L0(self) -> np.ndarray:
        """Compute the standard 0-Laplacian (node level)."""
        degrees = np.sum(self.W, axis=1)
        D = np.diag(degrees)
        return D - self.W

    def compute_L1(self) -> np.ndarray:
        """
        Compute the 1-Laplacian (edge level).
        L1 = B1^T W0^{-1} B1 W1 + W1 B2 W2^{-1} B2^T
        (Simplified version returning unweighted boundary matrices B1^T * B1).
        Requires explicit Triangles/Cliques to be computed for full L1, 
        using boundary matrix B1 here for approximation.
        """
        m = len(self.edges)
        
        # B1 matrix (nodes x edges) boundary operator
        B1 = np.zeros((self.n, m), dtype=np.float32)
        for e_idx, (u, v) in enumerate(self.edges):
            u_i = self.node_idx[u]
            v_i = self.node_idx[v]
            # arbitrarily orient edges u -> v
            B1[u_i, e_idx] = -1
            B1[v_i, e_idx] = 1

        # L1 = B1^T * B1 (combinatorial Hodge 1-laplacian)
        L1 = B1.T @ B1
        return L1

    def compute_L2(self) -> np.ndarray:
        """
        Compute the 2-Laplacian (triangle level) L2_down.
        L2_down = B2 * B2^T mapping edges to faces (triangles).
        We first find all 3-cliques (triangles) to form the 2-simplices.
        """
        import itertools
        
        # Build adjacency set for fast neighbor lookup
        adj = {i: set() for i in range(self.n)}
        for (u, v) in self.edges:
            ui = self.node_idx[u]
            vi = self.node_idx[v]
            adj[ui].add(vi)
            adj[vi].add(ui)
            
        # Find all triangles (3-cliques)
        triangles = set()
        for i in range(self.n):
            neighbors = list(adj[i])
            for j, k in itertools.combinations(neighbors, 2):
                if k in adj[j]:
                    # Canonical orientation: sorted tuple of indices
                    tri = tuple(sorted((i, j, k)))
                    triangles.add(tri)
                    
        triangles = sorted(list(triangles))
        m = len(self.edges)
        f = len(triangles)
        
        if f == 0:
            return np.zeros((m, m), dtype=np.float32)
            
        # B2 boundary matrix (edges x triangles)
        # Orientation: for triangle {i, j, k} with i < j < k
        # Boundaries are edges {i,j}, {j,k}, {i,k}.
        # Signs: +1 for {j,k}, -1 for {i,k}, +1 for {i,j}
        # to satisfy standard boundary operator alternating sum.
        B2 = np.zeros((m, f), dtype=np.float32)
        
        edges_list = self.edges
        # Needs O(1) edge index lookup considering orientation
        edge_to_idx = {}
        for idx, (u, v) in enumerate(edges_list):
            ui, vi = self.node_idx[u], self.node_idx[v]
            canon_edge = tuple(sorted((ui, vi)))
            edge_to_idx[canon_edge] = idx
            
        for f_idx, (i, j, k) in enumerate(triangles):
            e1 = tuple(sorted((j, k)))
            e2 = tuple(sorted((i, k)))
            e3 = tuple(sorted((i, j)))
            
            if e1 in edge_to_idx: B2[edge_to_idx[e1], f_idx] = 1.0
            if e2 in edge_to_idx: B2[edge_to_idx[e2], f_idx] = -1.0
            if e3 in edge_to_idx: B2[edge_to_idx[e3], f_idx] = 1.0
            
        # L2_down = B2 * B2^T
        L2 = B2 @ B2.T
        return L2
