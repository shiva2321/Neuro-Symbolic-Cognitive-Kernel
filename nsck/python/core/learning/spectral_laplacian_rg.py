import numpy as np
from typing import List, Dict, Tuple, Optional

class SpectralLaplacianRG:
    """
    Spectral Renormalization Group over knowledge graph Laplacian.
    Used for coarse-graining concepts into broader neighborhood abstractions.
    """

    def __init__(self, tau: float = 0.5):
        """
        tau: diffusion time parameter for the heat kernel.
        """
        self.tau = tau

    def compute_laplacian(self, adjacency_matrix: np.ndarray) -> np.ndarray:
        """
        Graph Laplacian L = D - A, where D is the degree matrix.
        Assumes A is symmetric and non-negative.
        """
        degrees = np.sum(adjacency_matrix, axis=1)
        D = np.diag(degrees)
        return D - adjacency_matrix

    def normalized_laplacian(self, adjacency_matrix: np.ndarray) -> np.ndarray:
        """
        L_norm = I - D^{-1/2} A D^{-1/2}
        """
        degrees = np.sum(adjacency_matrix, axis=1)
        
        # Avoid division by zero
        with np.errstate(divide='ignore'):
            d_inv_sqrt = np.power(degrees, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0

        n = adjacency_matrix.shape[0]
        L = self.compute_laplacian(adjacency_matrix)
        
        D_inv_sqrt_mat = np.diag(d_inv_sqrt)
        return D_inv_sqrt_mat @ L @ D_inv_sqrt_mat

    def spectral_eigendecomposition(self, laplacian: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the first k eigenvalues and eigenvectors of the Laplacian.
        Returns:
            eigenvalues: shape (k,)
            eigenvectors: shape (N, k)
        """
        # For symmetric matrices, eigh is more stable and faster
        eigvals, eigvecs = np.linalg.eigh(laplacian)
        
        # Sort by eigenvalue ascending (smallest are slowest decaying modes i.e., macroscopic features)
        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # We cap at k
        n_components = min(k, len(eigvals))
        return eigvals[:n_components], eigvecs[:, :n_components]

    def compute_heat_kernel_signature(self, eigenvalues: np.ndarray, eigenvectors: np.ndarray) -> np.ndarray:
        """
        Computes the Heat Kernel Signature (HKS) for each node at time self.tau.
        HKS(x, tau) = sum_{i} exp(-lambda_i * tau) * (phi_i(x))^2
        """
        # eigenvalues: (k,)
        # eigenvectors: (N, k)
        
        # exponential decay weights
        weights = np.exp(-eigenvalues * self.tau) # (k,)
        
        # square the eigenvectors (phi_i(x)^2)
        phi_sq = np.square(eigenvectors) # (N, k)
        
        # dot product
        hks = phi_sq @ weights # (N,)
        return hks

    def create_supernodes(self, node_ids: List[str], adjacency_matrix: np.ndarray, num_macrostates: int) -> Dict[str, List[str]]:
        """
        Perform coarse-graining by clustering nodes based on their heat kernel embeddings.
        This isolates functional macroscopic domains (cities).
        """
        n = adjacency_matrix.shape[0]
        if n == 0 or num_macrostates >= n:
            return {f"supernode_0": node_ids}
            
        L = self.normalized_laplacian(adjacency_matrix)
        
        # Use top K slow modes (eigenvectors associated to smallest eigenvalues > 0)
        # 0th eigenvalue is 0 for connected components, associated with constant vector.
        # k should be roughly num_macrostates + 1
        k_modes = min(n, num_macrostates + 2)
        evals, evecs = self.spectral_eigendecomposition(L, k_modes)
        
        # Use spectral embedding (ignoring the zeroth constant eigenvector if connected)
        # We weight the embeddings by exp(-lambda * tau) to emphasize slower modes
        weights = np.exp(-evals[1:] * self.tau)
        embedding = evecs[:, 1:] * np.sqrt(weights)[np.newaxis, :]
        
        # KMeans clustering on the spectral embedding
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=num_macrostates, n_init='auto', random_state=42)
        labels = kmeans.fit_predict(embedding)
        
        supernodes = {}
        for i in range(num_macrostates):
            supernodes[f"supernode_{i}"] = []
            
        for i, label in enumerate(labels):
            supernodes[f"supernode_{label}"].append(node_ids[i])
            
        return supernodes
