"""
KnowledgeNeighborhood and KnowledgeDomain — societal structure for NSCK V5.

Concepts self-organize into neighbourhoods (districts) and domains (cities).
Each neighbourhood elects an anchor concept (mayor) based on electronegativity.
Each domain bundles anchor HVs into a central "city-hall" HV for fast routing.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Set

import numpy as np

import python.core.vsa.hypervec_shim as hypervec_rs


# ---------------------------------------------------------------------------
# KnowledgeNeighborhood
# ---------------------------------------------------------------------------

class KnowledgeNeighborhood:
    """A cluster of thematically related concepts.

    Parameters
    ----------
    neighborhood_id:
        Unique identifier.  Auto-generated if ``None``.
    """

    def __init__(self, neighborhood_id: Optional[str] = None) -> None:
        self.neighborhood_id: str = neighborhood_id or f"nbhd_{uuid.uuid4().hex[:8]}"
        self.concept_ids: Set[str] = set()
        self.anchor_id: Optional[str] = None
        self.domain_id: Optional[str] = None
        self.border_concepts: Set[str] = set()

    # ------------------------------------------------------------------

    def add_concept(self, concept_id: str, lhv: Any = None) -> None:
        """Add a concept to this neighbourhood (and update its neighborhood_id)."""
        self.concept_ids.add(concept_id)
        if lhv is not None:
            lhv.neighborhood_id = self.neighborhood_id

    def remove_concept(self, concept_id: str) -> None:
        """Remove a concept from the neighbourhood."""
        self.concept_ids.discard(concept_id)
        self.border_concepts.discard(concept_id)
        if self.anchor_id == concept_id:
            self.anchor_id = None

    def elect_anchor(self, concepts_dict: Dict[str, Any]) -> Optional[str]:
        """Elect the concept with highest ``electronegativity`` as anchor."""
        best_id: Optional[str] = None
        best_score: float = -1.0
        for cid in self.concept_ids:
            lhv = concepts_dict.get(cid)
            if lhv is None:
                continue
            if lhv.electronegativity > best_score:
                best_score = lhv.electronegativity
                best_id = cid
        self.anchor_id = best_id
        return best_id

    def compute_border_concepts(
        self,
        concepts_dict: Dict[str, Any],
        world_graph: Optional[Any] = None,  # reserved for graph-based border detection
    ) -> Set[str]:
        """Find boundary concepts — those with bonds to concepts outside this neighbourhood."""
        borders: Set[str] = set()
        for cid in self.concept_ids:
            lhv = concepts_dict.get(cid)
            if lhv is None:
                continue
            for bonded_id in lhv.bonds:
                if bonded_id not in self.concept_ids:
                    borders.add(cid)
                    break
        self.border_concepts = borders
        return borders

    def merge_with(self, other: "KnowledgeNeighborhood") -> "KnowledgeNeighborhood":
        """Return a new neighbourhood that is the union of self and other."""
        merged = KnowledgeNeighborhood()
        merged.concept_ids = self.concept_ids | other.concept_ids
        merged.domain_id = self.domain_id or other.domain_id
        return merged

    def stats(self) -> Dict[str, Any]:
        return {
            "neighborhood_id": self.neighborhood_id,
            "n_concepts": len(self.concept_ids),
            "anchor_id": self.anchor_id,
            "domain_id": self.domain_id,
            "n_border_concepts": len(self.border_concepts),
        }

    def __repr__(self) -> str:
        return (
            f"KnowledgeNeighborhood(id={self.neighborhood_id!r}, "
            f"n={len(self.concept_ids)}, anchor={self.anchor_id!r})"
        )


# ---------------------------------------------------------------------------
# KnowledgeDomain
# ---------------------------------------------------------------------------

class KnowledgeDomain:
    """A domain (city) composed of neighbourhoods.

    Parameters
    ----------
    domain_id:
        Unique identifier.  Auto-generated if ``None``.
    name:
        Human-readable name for this domain.
    """

    def __init__(
        self,
        name: str,
        domain_id: Optional[str] = None,
        percolation_threshold: float = 0.30,
    ) -> None:
        self.domain_id: str = domain_id or f"domain_{uuid.uuid4().hex[:8]}"
        self.name: str = name
        self.neighborhood_ids: Set[str] = set()
        self.city_hall_hv: Optional[Any] = None
        self.bridge_concepts: Dict[str, List[str]] = {}
        self.percolation_threshold: float = float(percolation_threshold)

    # ------------------------------------------------------------------

    def add_neighborhood(self, nbhd: KnowledgeNeighborhood) -> None:
        """Register a neighbourhood and link it back to this domain."""
        self.neighborhood_ids.add(nbhd.neighborhood_id)
        nbhd.domain_id = self.domain_id

    def remove_neighborhood(self, nbhd_id: str) -> None:
        """Deregister a neighbourhood."""
        self.neighborhood_ids.discard(nbhd_id)

    def update_city_hall(
        self,
        neighborhoods: Dict[str, KnowledgeNeighborhood],
        concepts: Dict[str, Any],
    ) -> None:
        """Recompute city_hall_hv as a bundle of anchor HVs."""
        anchor_hvs = []
        for nbhd_id in self.neighborhood_ids:
            nbhd = neighborhoods.get(nbhd_id)
            if nbhd is None or nbhd.anchor_id is None:
                continue
            lhv = concepts.get(nbhd.anchor_id)
            if lhv is not None:
                anchor_hvs.append(lhv.hv)

        if not anchor_hvs:
            self.city_hall_hv = None
            return

        # Bundle all anchor HVs into a single city-hall vector
        city_hv = anchor_hvs[0]
        for hv in anchor_hvs[1:]:
            try:
                city_hv = city_hv.bundle(hv)
            except Exception:
                pass
        self.city_hall_hv = city_hv

    def compute_bridge_score(
        self, concept_id: str, concepts_dict: Dict[str, Any]
    ) -> float:
        """Return how strongly a concept bridges to other domains (0–1)."""
        lhv = concepts_dict.get(concept_id)
        if lhv is None:
            return 0.0
        other_domain_count = 0
        for bonded_id in lhv.bonds:
            bonded = concepts_dict.get(bonded_id)
            if bonded is not None and bonded.primary_domain != lhv.primary_domain:
                other_domain_count += 1
        n_bonds = len(lhv.bonds)
        if n_bonds == 0:
            return 0.0
        return float(other_domain_count / n_bonds)

    def stats(self) -> Dict[str, Any]:
        return {
            "domain_id": self.domain_id,
            "name": self.name,
            "n_neighborhoods": len(self.neighborhood_ids),
            "has_city_hall": self.city_hall_hv is not None,
            "percolation_threshold": self.percolation_threshold,
            "n_bridge_connections": sum(len(v) for v in self.bridge_concepts.values()),
        }

    def __repr__(self) -> str:
        return (
            f"KnowledgeDomain(id={self.domain_id!r}, name={self.name!r}, "
            f"neighborhoods={len(self.neighborhood_ids)})"
        )
