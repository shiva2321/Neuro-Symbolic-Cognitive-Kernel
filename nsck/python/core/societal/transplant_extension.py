"""
SocietalTransplantStage — Stage 7 Societal Seeding for NSCK V5.

Extends the TransplantPipeline with a societal seeding stage that:
  7a. Assigns stability classes by token frequency proxy.
  7b. Clusters embeddings into sub-domains.
  7c. Registers each token as a LivingHyperVector.
  7d. Forms neighbourhoods from clusters.
  7e. Initialises bonds from embedding similarity.
  7f. Forms a domain from the neighbourhoods.
"""

from __future__ import annotations

import logging
import math
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from python.core.societal.societal_world import SocietalKnowledgeWorld

logger = logging.getLogger("nsck.societal_transplant")


def _kmeans_numpy(
    embeddings: np.ndarray, n_clusters: int, seed: int = 42, max_iter: int = 30
) -> np.ndarray:
    """Minimal k-means via numpy (labels array)."""
    n = embeddings.shape[0]
    k = min(n_clusters, n)
    rng = np.random.default_rng(seed)
    centroids = embeddings[rng.choice(n, size=k, replace=False)].copy()

    labels = np.zeros(n, dtype=np.int32)
    for _ in range(max_iter):
        dists = np.linalg.norm(embeddings[:, np.newaxis] - centroids[np.newaxis], axis=2)
        new_labels = np.argmin(dists, axis=1).astype(np.int32)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        for ci in range(k):
            mask = labels == ci
            if mask.any():
                centroids[ci] = embeddings[mask].mean(axis=0)
    return labels


class SocietalTransplantStage:
    """Stage 7 of the transplantation pipeline: Societal Seeding.

    Parameters
    ----------
    world:
        The target SocietalKnowledgeWorld.
    n_domain_clusters:
        Number of sub-domain clusters to form (default 10).
    seed:
        Random seed for reproducibility (default 42).
    max_pairs:
        Maximum number of pairwise bond candidates to evaluate (default 5000).
    """

    def __init__(
        self,
        world: "SocietalKnowledgeWorld",
        n_domain_clusters: int = 10,
        seed: int = 42,
        max_pairs: int = 5000,
    ) -> None:
        self.world = world
        self.n_domain_clusters = int(n_domain_clusters)
        self.seed = int(seed)
        self.max_pairs = int(max_pairs)

    # ------------------------------------------------------------------

    def run_stage7(
        self,
        transplant_report: Any,
        embeddings: Dict[str, np.ndarray],
        domain_name: str,
        codebook: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute Stage 7 societal seeding.

        Parameters
        ----------
        transplant_report:
            The TransplantReport from the preceding pipeline stages.
        embeddings:
            Dict mapping token string → dense embedding (np.ndarray).
        domain_name:
            Name for the new domain.
        codebook:
            Optional dict token → HV (from TransplantPipeline.run()).
            If None, new HVs are generated via hypervec_shim.

        Returns
        -------
        Dict with seeding statistics.
        """
        import python.core.vsa.hypervec_shim as _hypervec_rs  # noqa: PLC0415

        tokens = list(embeddings.keys())
        n = len(tokens)
        if n == 0:
            return {"n_concepts_registered": 0, "n_neighborhoods": 0,
                    "n_bonds_formed": 0, "domain_id": None, "societal_stats": {}}

        before_stats = self.world.stats_report()

        # 7a — stability class by frequency proxy
        # Assume tokens at the start of the list are more frequent
        stabilities = np.linspace(0.9, 0.2, n)  # frequent=0.9, rare=0.2

        # 7b — cluster embeddings into sub-domains
        emb_matrix = np.stack([embeddings[t] for t in tokens]).astype(np.float32)
        n_clusters = min(self.n_domain_clusters, n)

        try:
            from sklearn.cluster import KMeans  # noqa: PLC0415
            km = KMeans(n_clusters=n_clusters, random_state=self.seed, n_init=5)
            cluster_labels = km.fit_predict(emb_matrix).astype(np.int32)
        except Exception:
            cluster_labels = _kmeans_numpy(emb_matrix, n_clusters, self.seed)

        # 7c — register each token as LivingHyperVector
        import time as _time  # noqa: PLC0415
        n_registered = 0
        for i, token in enumerate(tokens):
            if codebook and token in codebook:
                hv = codebook[token]
            else:
                hv = _hypervec_rs.HyperVector(seed=hash(token) % (2 ** 32))
            md: Dict[str, Any] = {
                "stability": float(stabilities[i]),
                "embedding": embeddings[token],
                "provenance": {
                    "created_at": _time.time(),
                    "method": "societal_transplant",
                    "source_domain": domain_name,
                },
            }
            self.world.register_concept(token, hv, md)
            n_registered += 1

        # 7d — form neighbourhoods from clusters
        cluster_to_tokens: Dict[int, List[str]] = {}
        for i, token in enumerate(tokens):
            c = int(cluster_labels[i])
            cluster_to_tokens.setdefault(c, []).append(token)

        nbhd_ids = []
        for cluster_id, cluster_tokens in cluster_to_tokens.items():
            nbhd_id = f"nbhd_{domain_name}_{cluster_id}"
            self.world.form_neighborhood(cluster_tokens, nbhd_id)
            nbhd_ids.append(nbhd_id)

        # 7e — initialise bonds from embedding similarity (capped to max_pairs)
        n_bonds = 0
        norm_embs = emb_matrix / (np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-9)

        # Only evaluate within-cluster pairs
        for cluster_tokens in cluster_to_tokens.values():
            idx = [tokens.index(t) for t in cluster_tokens]
            if len(idx) < 2:
                continue
            sub_embs = norm_embs[idx]
            sim_mat = sub_embs @ sub_embs.T  # (m, m) cosine sims

            pairs_evaluated = 0
            for ii in range(len(idx)):
                for jj in range(ii + 1, len(idx)):
                    if pairs_evaluated >= self.max_pairs:
                        break
                    sim_val = float(sim_mat[ii, jj])
                    if sim_val > self.world._valence.bond_threshold - 0.1:
                        ta = cluster_tokens[ii]
                        tb = cluster_tokens[jj]
                        lhv_a = self.world.concepts.get(ta)
                        lhv_b = self.world.concepts.get(tb)
                        if lhv_a and lhv_b:
                            strength = self.world._valence.try_form_bond(lhv_a, lhv_b)
                            if strength is not None:
                                n_bonds += 1
                    pairs_evaluated += 1

        # 7f — form domain from neighbourhoods
        domain_id = f"domain_{domain_name}_{uuid.uuid4().hex[:6]}"
        domain = self.world.form_domain(nbhd_ids, domain_name, domain_id)

        after_stats = self.world.stats_report()
        delta = self.compute_societal_stats_delta(before_stats, after_stats)

        return {
            "n_concepts_registered": n_registered,
            "n_neighborhoods": len(nbhd_ids),
            "n_bonds_formed": n_bonds,
            "domain_id": domain.domain_id,
            "societal_stats": after_stats,
            "delta": delta,
        }

    # ------------------------------------------------------------------

    @staticmethod
    def compute_societal_stats_delta(
        before: Dict[str, Any], after: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compute the difference between two stats_report() outputs."""
        numeric_keys = [
            "n_concepts", "n_neighborhoods", "n_domains",
            "total_bonds", "avg_bonds_per_concept",
        ]
        delta: Dict[str, Any] = {}
        for key in numeric_keys:
            b = before.get(key, 0)
            a = after.get(key, 0)
            if isinstance(b, (int, float)) and isinstance(a, (int, float)):
                delta[f"d_{key}"] = a - b
        return delta
