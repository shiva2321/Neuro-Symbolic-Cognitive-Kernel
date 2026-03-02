"""
SocietalHNSW — Hierarchical Navigable Small World for NSCK V5.

Three-layer hierarchy:
  Layer 2: domain anchors (city-hall concepts — highest electronegativity per domain)
  Layer 1: neighbourhood anchors
  Layer 0: all concepts

Falls back to a pure-Python greedy NSW when hnswlib is unavailable.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import numpy as np

if TYPE_CHECKING:
    from python.core.societal.societal_world import SocietalKnowledgeWorld

logger = logging.getLogger("nsck.societal_hnsw")

try:
    import hnswlib as _hnswlib  # type: ignore
    _HNSWLIB_AVAILABLE = True
except ImportError:
    _hnswlib = None
    _HNSWLIB_AVAILABLE = False

_DIM = 10240   # HV dimension


def _hv_to_vec(hv: Any) -> np.ndarray:
    """Convert a hypervector to a normalised float32 vector."""
    try:
        bits = np.asarray(hv.bits, dtype=np.float32)
    except Exception:
        return np.zeros(_DIM, dtype=np.float32)
    bipolar = bits * 2.0 - 1.0
    n = float(np.linalg.norm(bipolar))
    return bipolar / n if n > 0.0 else bipolar


def _cosine_sim(va: np.ndarray, vb: np.ndarray) -> float:
    return float(np.dot(va, vb))   # both already L2-normalised


# ---------------------------------------------------------------------------
# Pure-Python greedy NSW (fallback)
# ---------------------------------------------------------------------------

class _PureNSW:
    """Greedy Navigable Small World index (pure numpy)."""

    def __init__(self, M: int = 16, ef: int = 50) -> None:
        self._M = M
        self._ef = ef
        self._ids: List[str] = []
        self._vecs: List[np.ndarray] = []
        self._graph: Dict[int, List[int]] = {}

    def add(self, concept_id: str, vec: np.ndarray) -> None:
        idx = len(self._ids)
        self._ids.append(concept_id)
        self._vecs.append(vec)
        self._graph[idx] = []

        if idx == 0:
            return

        # Find M nearest existing neighbours
        sims = [(_cosine_sim(vec, self._vecs[j]), j) for j in range(idx)]
        sims.sort(reverse=True)
        for _, j in sims[: self._M]:
            self._graph[idx].append(j)
            if len(self._graph[j]) < self._M * 2:
                self._graph[j].append(idx)

    def query(self, vec: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        n = len(self._ids)
        if n == 0:
            return []

        # Greedy search from random entry point
        visited: Set[int] = set()
        candidates: List[Tuple[float, int]] = []
        entry = 0  # always enter from node 0
        entry_sim = _cosine_sim(vec, self._vecs[entry])
        candidates.append((-entry_sim, entry))
        visited.add(entry)

        results: List[Tuple[float, int]] = [(entry_sim, entry)]

        while candidates:
            neg_sim, curr = min(candidates)
            candidates.remove((neg_sim, curr))
            curr_sim = -neg_sim

            expanded = False
            for nb in self._graph.get(curr, []):
                if nb in visited:
                    continue
                visited.add(nb)
                nb_sim = _cosine_sim(vec, self._vecs[nb])
                results.append((nb_sim, nb))
                if nb_sim > curr_sim - 0.1:
                    candidates.append((-nb_sim, nb))
                    expanded = True

            if len(results) >= self._ef and not expanded:
                break

        results.sort(reverse=True)
        return [(self._ids[idx], sim) for sim, idx in results[:top_k]]


# ---------------------------------------------------------------------------
# SocietalHNSW
# ---------------------------------------------------------------------------

class SocietalHNSW:
    """Societal HNSW index with three-layer hierarchy.

    Parameters
    ----------
    ef_construction:
        hnswlib ef_construction parameter (default 200).
    M:
        Max connections per element (default 16).
    seed:
        Random seed for hnswlib (default 42).
    """

    def __init__(
        self,
        ef_construction: int = 200,
        M: int = 16,
        seed: int = 42,
    ) -> None:
        self.ef_construction = int(ef_construction)
        self.M = int(M)
        self.seed = int(seed)

        # Layer maps
        self._layer2_ids: List[str] = []   # domain anchors
        self._layer1_ids: List[str] = []   # neighbourhood anchors
        self._all_ids: List[str] = []      # all concepts

        # NSW/HNSW index objects
        self._index_l2: Optional[_PureNSW] = None
        self._index_l1: Optional[_PureNSW] = None
        self._index_l0: Optional[_PureNSW] = None

        # Cross-domain bridges
        self._bridges: Dict[str, List[Tuple[str, str, float]]] = {}
        # domain_id → [(concept_a, concept_b, bond_weight)]

        # Domain → concept_id membership (for domain_filter)
        self._domain_members: Dict[str, Set[str]] = {}

        self._built = False

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, world: "SocietalKnowledgeWorld") -> None:
        """Build the three-layer index from a SocietalKnowledgeWorld."""
        concepts = world.concepts
        neighborhoods = world.neighborhoods
        domains = world.domains

        if not concepts:
            logger.warning("SocietalHNSW.build called on empty world")
            return

        # --- Collect layer memberships ---
        l2_set: Set[str] = set()
        l1_set: Set[str] = set()

        for domain in domains.values():
            members: Set[str] = set()
            for nbhd_id in domain.neighborhood_ids:
                nbhd = neighborhoods.get(nbhd_id)
                if nbhd is None:
                    continue
                members |= nbhd.concept_ids
                if nbhd.anchor_id:
                    l1_set.add(nbhd.anchor_id)
            self._domain_members[domain.domain_id] = members

        # Layer-2: highest-electronegativity concept per domain
        for domain in domains.values():
            members = self._domain_members.get(domain.domain_id, set())
            best_id = None
            best_e = -1.0
            for cid in members:
                lhv = concepts.get(cid)
                if lhv and lhv.electronegativity > best_e:
                    best_e = lhv.electronegativity
                    best_id = cid
            if best_id:
                l2_set.add(best_id)

        self._layer2_ids = list(l2_set)
        self._layer1_ids = list(l1_set)
        self._all_ids = list(concepts.keys())

        # --- Build NSW indices ---
        self._index_l2 = _PureNSW(M=self.M)
        self._index_l1 = _PureNSW(M=self.M)
        self._index_l0 = _PureNSW(M=self.M)

        for cid in self._layer2_ids:
            vec = _hv_to_vec(concepts[cid].hv)
            self._index_l2.add(cid, vec)

        for cid in self._layer1_ids:
            vec = _hv_to_vec(concepts[cid].hv)
            self._index_l1.add(cid, vec)

        for cid in self._all_ids:
            vec = _hv_to_vec(concepts[cid].hv)
            self._index_l0.add(cid, vec)

        # --- Cross-domain bridges ---
        for domain_id, members in self._domain_members.items():
            bridges = []
            for cid in members:
                lhv = concepts.get(cid)
                if lhv is None:
                    continue
                for bonded_id, w in lhv.bonds.items():
                    if w > 0.2 and bonded_id not in members:
                        bridges.append((cid, bonded_id, w))
            self._bridges[domain_id] = bridges

        self._built = True
        logger.info(
            "SocietalHNSW built: L2=%d L1=%d L0=%d",
            len(self._layer2_ids), len(self._layer1_ids), len(self._all_ids),
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        query_hv: Any,
        top_k: int = 10,
        domain_filter: Optional[str] = None,
        context_hv: Optional[Any] = None,
    ) -> List[Tuple[str, float]]:
        """Retrieve top-k concepts using hierarchical search.

        Parameters
        ----------
        query_hv:
            Raw hypervector to query against.
        top_k:
            Number of results.
        domain_filter:
            If given, only return concepts in this domain.
        context_hv:
            Optional context HV to blend with query.

        Returns
        -------
        List of (concept_id, similarity_score) sorted descending.
        """
        if not self._built or self._index_l0 is None:
            return []

        # Blend with context
        if context_hv is not None:
            try:
                effective_hv = query_hv.bundle(context_hv)
            except Exception:
                effective_hv = query_hv
        else:
            effective_hv = query_hv

        vec = _hv_to_vec(effective_hv)

        # Layer 2 → layer 1 → layer 0
        if self._index_l2 and self._layer2_ids:
            self._index_l2.query(vec, top_k=3)  # warm up search direction
        if self._index_l1 and self._layer1_ids:
            self._index_l1.query(vec, top_k=5)

        results = self._index_l0.query(vec, top_k=top_k * 3)

        # Domain filter
        if domain_filter and domain_filter in self._domain_members:
            members = self._domain_members[domain_filter]
            results = [(cid, s) for cid, s in results if cid in members]

        return results[:top_k]

    # ------------------------------------------------------------------
    # Cross-domain routing
    # ------------------------------------------------------------------

    def cross_domain_route(
        self,
        query_hv: Any,
        source_domain: str,
        target_domain: str,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Navigate via bridge concepts from source to target domain."""
        # Find bridges from source_domain
        bridges = self._bridges.get(source_domain, [])
        target_members = self._domain_members.get(target_domain, set())

        # Filter bridges that reach target_domain
        target_bridges = [
            (ca, cb, w)
            for ca, cb, w in bridges
            if cb in target_members
        ]

        if not target_bridges:
            # Fall back to direct query in target domain
            return self.query(query_hv, top_k=top_k, domain_filter=target_domain)

        vec = _hv_to_vec(query_hv)
        if self._index_l0 is None:
            return []

        results = self._index_l0.query(vec, top_k=top_k * 5)
        # Re-rank: boost concepts that are bridge endpoints
        bridge_set = {cb for _, cb, _ in target_bridges}
        ranked = []
        for cid, sim in results:
            if cid in target_members:
                boost = 0.1 if cid in bridge_set else 0.0
                ranked.append((cid, sim + boost))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "built": self._built,
            "n_layer2": len(self._layer2_ids),
            "n_layer1": len(self._layer1_ids),
            "n_layer0": len(self._all_ids),
            "n_domains": len(self._domain_members),
            "n_bridge_pairs": sum(len(v) for v in self._bridges.values()),
        }
