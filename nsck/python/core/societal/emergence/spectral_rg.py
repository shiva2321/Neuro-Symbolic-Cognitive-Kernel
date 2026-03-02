"""
SpectralLaplacianRG — Spectral Renormalization Group for NSCK V5.

Implements the spectral Laplacian coarse-graining pipeline:
  1. Build similarity graph from HV cosine similarities.
  2. Compute normalized graph Laplacian.
  3. Eigendecompose to expose spectral gap & community structure.
  4. Coarse-grain: cluster concepts into supernodes.
  5. Build supernode graph.
"""

from __future__ import annotations

import logging
import math
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from python.core.societal.living_hypervector import _hv_cosine_sim

logger = logging.getLogger("nsck.spectral_rg")


class SpectralLaplacianRG:
    """Spectral Renormalization Group coarse-graining.

    Parameters
    ----------
    n_eigenvectors:
        Number of eigenvectors to retain (default 32).
    similarity_threshold:
        Minimum cosine similarity to include an edge in the graph (default 0.3).
    coarse_grain_ratio:
        Target fraction of supernodes relative to input concepts (default 0.5).
    """

    def __init__(
        self,
        n_eigenvectors: int = 32,
        similarity_threshold: float = 0.30,
        coarse_grain_ratio: float = 0.50,
    ) -> None:
        self.n_eigenvectors = int(n_eigenvectors)
        self.similarity_threshold = float(similarity_threshold)
        self.coarse_grain_ratio = float(coarse_grain_ratio)

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def build_similarity_graph(
        self, concepts: Dict[str, Any]
    ) -> Tuple[List[str], np.ndarray]:
        """Compute pairwise HV cosine similarities.

        Parameters
        ----------
        concepts:
            Dict mapping concept_id → LivingHyperVector.

        Returns
        -------
        (concept_ids, adj_matrix) where adj_matrix is NxN float32.
        Diagonal is 0; entries below similarity_threshold are zeroed.
        """
        ids = list(concepts.keys())
        n = len(ids)
        adj = np.zeros((n, n), dtype=np.float32)

        if n < 2:
            return ids, adj

        # Extract bit arrays for vectorized similarity
        try:
            bits = np.stack(
                [np.asarray(concepts[cid].hv.bits, dtype=np.float32) for cid in ids]
            )  # (N, D)
            # Convert binary {0,1} → bipolar {-1,+1}
            bipolar = bits * 2.0 - 1.0
            norms = np.linalg.norm(bipolar, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            normalized = bipolar / norms  # (N, D)
            adj = (normalized @ normalized.T).astype(np.float32)  # cosine sim
        except Exception:
            # Fallback: pairwise
            for i in range(n):
                for j in range(i + 1, n):
                    s = _hv_cosine_sim(concepts[ids[i]].hv, concepts[ids[j]].hv)
                    adj[i, j] = adj[j, i] = float(s)

        np.fill_diagonal(adj, 0.0)
        # Threshold
        adj[adj < self.similarity_threshold] = 0.0
        return ids, adj

    # ------------------------------------------------------------------
    # Laplacian
    # ------------------------------------------------------------------

    def compute_laplacian(self, adj_matrix: np.ndarray) -> np.ndarray:
        """Compute the normalized graph Laplacian L = I − D^{-½} A D^{-½}."""
        n = adj_matrix.shape[0]
        degree = adj_matrix.sum(axis=1)
        # Suppress divide-by-zero for isolated nodes (degree=0 → d_inv_sqrt=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            d_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(np.where(degree > 0, degree, 1.0)), 0.0)
        # D^{-½} A D^{-½}
        norm = np.outer(d_inv_sqrt, d_inv_sqrt) * adj_matrix
        return np.eye(n, dtype=np.float32) - norm.astype(np.float32)

    # ------------------------------------------------------------------
    # Eigendecomposition
    # ------------------------------------------------------------------

    def eigendecompose(
        self, laplacian: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Eigendecompose the symmetric Laplacian.

        Returns
        -------
        (eigenvalues, eigenvectors) sorted ascending.
        """
        try:
            vals, vecs = np.linalg.eigh(laplacian.astype(np.float64))
        except np.linalg.LinAlgError:
            n = laplacian.shape[0]
            vals = np.zeros(n, dtype=np.float64)
            vecs = np.eye(n, dtype=np.float64)
        idx = np.argsort(vals)
        return vals[idx].astype(np.float32), vecs[:, idx].astype(np.float32)

    # ------------------------------------------------------------------
    # Spectral gap
    # ------------------------------------------------------------------

    def spectral_gap(self, eigenvalues: np.ndarray) -> float:
        """Return the gap between λ₁ (Fiedler value) and λ₀ ≈ 0."""
        if len(eigenvalues) < 2:
            return 0.0
        return float(eigenvalues[1] - eigenvalues[0])

    # ------------------------------------------------------------------
    # Coarse-graining
    # ------------------------------------------------------------------

    def coarse_grain(
        self,
        concept_ids: List[str],
        eigenvalues: np.ndarray,
        eigenvectors: np.ndarray,
    ) -> Dict[str, List[str]]:
        """Cluster concepts into supernodes using leading eigenvectors.

        Parameters
        ----------
        concept_ids:
            Ordered list of concept IDs (matches rows of eigenvectors).
        eigenvalues:
            Sorted eigenvalues.
        eigenvectors:
            Corresponding eigenvectors (columns), shape (N, N).

        Returns
        -------
        Dict mapping supernode_id → list[concept_id].
        """
        n = len(concept_ids)
        if n == 0:
            return {}

        n_clusters = max(1, int(math.ceil(n * self.coarse_grain_ratio)))
        n_clusters = min(n_clusters, n)

        # Use top-k eigenvectors as features for clustering
        k = min(self.n_eigenvectors, eigenvectors.shape[1], n_clusters)
        features = eigenvectors[:, :k]  # (N, k)

        assignments = self._kmeans_assign(features, n_clusters)

        supernodes: Dict[str, List[str]] = {}
        for i, cid in enumerate(concept_ids):
            sn_id = f"sn_{assignments[i]}"
            supernodes.setdefault(sn_id, []).append(cid)
        return supernodes

    @staticmethod
    def _kmeans_assign(features: np.ndarray, k: int) -> np.ndarray:
        """Assign rows of *features* to k clusters via k-means (numpy fallback)."""
        n = features.shape[0]
        if k >= n:
            return np.arange(n, dtype=np.int32)

        # Try scipy first
        try:
            from scipy.cluster.vq import kmeans2, whiten  # noqa: PLC0415

            feat = features.astype(np.float64)
            norms = np.linalg.norm(feat, axis=1, keepdims=True)
            feat_norm = feat / np.where(norms == 0, 1.0, norms)
            _, labels = kmeans2(feat_norm, k, minit="points", iter=30, seed=42)
            return labels.astype(np.int32)
        except Exception:
            pass

        # Pure numpy k-means
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=k, replace=False)
        centroids = features[idx].copy()  # (k, d)

        for _ in range(30):
            # Assign
            diffs = features[:, np.newaxis, :] - centroids[np.newaxis, :, :]
            dists = np.sum(diffs ** 2, axis=2)  # (n, k)
            labels = np.argmin(dists, axis=1)

            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for ci in range(k):
                mask = labels == ci
                if mask.any():
                    new_centroids[ci] = features[mask].mean(axis=0)
                else:
                    new_centroids[ci] = centroids[ci]

            if np.allclose(centroids, new_centroids, atol=1e-6):
                break
            centroids = new_centroids

        return labels.astype(np.int32)

    # ------------------------------------------------------------------
    # Supernode graph
    # ------------------------------------------------------------------

    def build_supernode_graph(
        self,
        concepts: Dict[str, Any],
        supernodes: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        """Build an inter-supernode graph.

        Returns
        -------
        Dict mapping supernode_id → {similar_supernodes: list, centroid_concept: str}.
        """
        # Find centroid concept for each supernode (highest electronegativity)
        centroids: Dict[str, str] = {}
        for sn_id, cids in supernodes.items():
            best_e = -1.0
            best_cid = cids[0] if cids else ""
            for cid in cids:
                lhv = concepts.get(cid)
                if lhv is not None and lhv.electronegativity > best_e:
                    best_e = lhv.electronegativity
                    best_cid = cid
            centroids[sn_id] = best_cid

        # Build membership map: concept → supernode
        membership: Dict[str, str] = {}
        for sn_id, cids in supernodes.items():
            for cid in cids:
                membership[cid] = sn_id

        # Find cross-supernode bonds
        cross: Dict[str, set] = {sn: set() for sn in supernodes}
        for cid, lhv in concepts.items():
            sn_a = membership.get(cid)
            if sn_a is None:
                continue
            for bonded_id in getattr(lhv, "bonds", {}):
                sn_b = membership.get(bonded_id)
                if sn_b is not None and sn_b != sn_a:
                    cross[sn_a].add(sn_b)

        return {
            sn_id: {
                "centroid_concept": centroids.get(sn_id, ""),
                "n_concepts": len(cids),
                "similar_supernodes": list(cross.get(sn_id, set())),
            }
            for sn_id, cids in supernodes.items()
        }

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run(self, concepts: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the full spectral RG pipeline.

        Returns
        -------
        Dict with keys: supernodes, supernode_graph, spectral_gap,
        eigenvalues, n_concepts, n_supernodes.
        """
        if len(concepts) < 2:
            return {
                "supernodes": {},
                "supernode_graph": {},
                "spectral_gap": 0.0,
                "eigenvalues": [],
                "n_concepts": len(concepts),
                "n_supernodes": 0,
            }

        ids, adj = self.build_similarity_graph(concepts)
        L = self.compute_laplacian(adj)
        evals, evecs = self.eigendecompose(L)
        gap = self.spectral_gap(evals)
        supernodes = self.coarse_grain(ids, evals, evecs)
        sn_graph = self.build_supernode_graph(concepts, supernodes)

        return {
            "supernodes": supernodes,
            "supernode_graph": sn_graph,
            "spectral_gap": float(gap),
            "eigenvalues": evals[:10].tolist(),
            "n_concepts": len(concepts),
            "n_supernodes": len(supernodes),
        }
