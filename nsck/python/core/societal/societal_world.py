"""
SocietalKnowledgeWorld — orchestrator for NSCK V5.

Manages the full lifecycle of LivingHyperVectors, KnowledgeNeighborhoods,
and KnowledgeDomains. Provides register/activate/query and the periodic
``run_societal_tick()`` that drives societal dynamics.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

import python.core.vsa.hypervec_shim as hypervec_rs
from python.core.societal.living_hypervector import LivingHyperVector, _hv_cosine_sim
from python.core.societal.valence_engine import ValenceEngine
from python.core.societal.knowledge_neighborhood import (
    KnowledgeNeighborhood,
    KnowledgeDomain,
)

logger = logging.getLogger("nsck.societal_world")

_TICK_BOND_UPDATE_INTERVAL = 10   # re-evaluate bonds every N ticks
_TICK_CITY_HALL_INTERVAL = 25     # rebuild city-hall HVs every N ticks
_SPREADING_ACTIVATION_FRACTION = 0.30   # fraction of activation shared with neighbours


class SocietalKnowledgeWorld:
    """The societal world orchestrator.

    Creates and manages a population of LivingHyperVectors organized
    into KnowledgeNeighborhoods and KnowledgeDomains.

    Parameters
    ----------
    config:
        Optional NSCK config object (currently unused, reserved for future use).
    bond_threshold:
        Minimum cosine-based score to form a bond between concepts.
    """

    def __init__(
        self,
        config: Any = None,
        bond_threshold: float = 0.30,
        break_threshold: float = 0.10,
    ) -> None:
        self._config = config
        self.concepts: Dict[str, LivingHyperVector] = {}
        self.neighborhoods: Dict[str, KnowledgeNeighborhood] = {}
        self.domains: Dict[str, KnowledgeDomain] = {}
        self.tick_count: int = 0
        self._valence = ValenceEngine(
            bond_threshold=bond_threshold,
            break_threshold=break_threshold,
        )

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_concept(
        self,
        concept_id: str,
        hv: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LivingHyperVector:
        """Register a new concept (or update existing one) in the world.

        Parameters
        ----------
        concept_id:
            Unique string identifier.
        hv:
            Hypervector object.
        metadata:
            Optional dict with initial property overrides.

        Returns
        -------
        The (new or updated) LivingHyperVector.
        """
        if concept_id in self.concepts:
            # Update HV but keep existing metadata
            self.concepts[concept_id].hv = hv
            return self.concepts[concept_id]

        lhv = LivingHyperVector(concept_id, hv, metadata)
        self.concepts[concept_id] = lhv
        logger.debug("Registered concept %r", concept_id)
        return lhv

    # ------------------------------------------------------------------
    # Activation
    # ------------------------------------------------------------------

    def activate_concept(
        self, concept_id: str, strength: float = 1.0
    ) -> Optional[LivingHyperVector]:
        """Activate a concept and propagate activation to bonded neighbours.

        Parameters
        ----------
        concept_id:
            The concept to activate.
        strength:
            Activation level (0–1).

        Returns
        -------
        The activated LHV, or ``None`` if not found.
        """
        lhv = self.concepts.get(concept_id)
        if lhv is None:
            return None
        lhv.update_activation(strength)

        # Spreading activation to neighbours
        spread = strength * _SPREADING_ACTIVATION_FRACTION
        for bonded_id, bond_w in lhv.bonds.items():
            neighbour = self.concepts.get(bonded_id)
            if neighbour is not None:
                new_act = min(1.0, neighbour.activation + spread * bond_w)
                neighbour.update_activation(new_act)

        return lhv

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        query_hv: Any,
        top_k: int = 10,
        context_hv: Optional[Any] = None,
        domain_filter: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """Retrieve top-k concepts most similar to query_hv.

        Parameters
        ----------
        query_hv:
            The query hypervector.
        top_k:
            Number of results to return.
        context_hv:
            Optional context HV blended with the query (50/50).
        domain_filter:
            If given, restrict results to concepts whose primary_domain matches.

        Returns
        -------
        List of (concept_id, score) tuples, sorted descending.
        """
        candidates = list(self.concepts.values())
        if domain_filter:
            candidates = [c for c in candidates if c.primary_domain == domain_filter]

        return self._valence.context_conditioned_retrieval(
            query_hv, candidates, context_hv=context_hv, top_k=top_k
        )

    # ------------------------------------------------------------------
    # Neighbourhood / Domain formation
    # ------------------------------------------------------------------

    def form_neighborhood(
        self,
        concept_ids: List[str],
        neighborhood_id: Optional[str] = None,
    ) -> KnowledgeNeighborhood:
        """Create a neighbourhood from a list of concept IDs.

        Parameters
        ----------
        concept_ids:
            IDs of concepts to include.
        neighborhood_id:
            Optional explicit ID (auto-generated if None).

        Returns
        -------
        The new KnowledgeNeighborhood.
        """
        nbhd = KnowledgeNeighborhood(neighborhood_id)
        for cid in concept_ids:
            lhv = self.concepts.get(cid)
            nbhd.add_concept(cid, lhv)
        nbhd.elect_anchor(self.concepts)
        nbhd.compute_border_concepts(self.concepts)
        self.neighborhoods[nbhd.neighborhood_id] = nbhd
        return nbhd

    def form_domain(
        self,
        neighborhood_ids: List[str],
        domain_name: str,
        domain_id: Optional[str] = None,
    ) -> KnowledgeDomain:
        """Create a domain from a list of neighbourhood IDs.

        Parameters
        ----------
        neighborhood_ids:
            IDs of neighbourhoods to include.
        domain_name:
            Human-readable domain name.
        domain_id:
            Optional explicit ID (auto-generated if None).

        Returns
        -------
        The new KnowledgeDomain.
        """
        domain = KnowledgeDomain(domain_name, domain_id)
        for nbhd_id in neighborhood_ids:
            nbhd = self.neighborhoods.get(nbhd_id)
            if nbhd is not None:
                domain.add_neighborhood(nbhd)
                # Assign domain affinity to all concepts in the neighbourhood
                for cid in nbhd.concept_ids:
                    lhv = self.concepts.get(cid)
                    if lhv is not None:
                        lhv.set_affinity(domain.name, 1.0)
        domain.update_city_hall(self.neighborhoods, self.concepts)
        self.domains[domain.domain_id] = domain
        return domain

    # ------------------------------------------------------------------
    # Societal tick
    # ------------------------------------------------------------------

    def run_societal_tick(self) -> None:
        """Execute one societal tick.

        Steps
        -----
        1. Tick every LHV (age++, activation decay, stability update).
        2. Periodically re-evaluate bonds within each neighbourhood.
        3. Periodically re-elect neighbourhood anchors.
        4. Periodically rebuild domain city-hall HVs.
        """
        self.tick_count += 1
        t = self.tick_count

        # Step 1: tick all LHVs
        for lhv in self.concepts.values():
            lhv.tick()

        # Step 2: bond updates (every N ticks)
        if t % _TICK_BOND_UPDATE_INTERVAL == 0:
            for nbhd in self.neighborhoods.values():
                concept_list = [
                    self.concepts[cid]
                    for cid in nbhd.concept_ids
                    if cid in self.concepts
                ]
                # Break weak bonds
                for i, lhv_a in enumerate(concept_list):
                    for lhv_b in concept_list[i + 1 :]:
                        self._valence.try_break_bond(lhv_a, lhv_b)
                # Try forming new bonds
                for i, lhv_a in enumerate(concept_list):
                    for lhv_b in concept_list[i + 1 :]:
                        if lhv_b.concept_id not in lhv_a.bonds:
                            self._valence.try_form_bond(lhv_a, lhv_b)

        # Step 3: anchor re-election (every 5 × bond interval)
        if t % (_TICK_BOND_UPDATE_INTERVAL * 5) == 0:
            for nbhd in self.neighborhoods.values():
                nbhd.elect_anchor(self.concepts)
                nbhd.compute_border_concepts(self.concepts)

        # Step 4: city-hall rebuild
        if t % _TICK_CITY_HALL_INTERVAL == 0:
            for domain in self.domains.values():
                domain.update_city_hall(self.neighborhoods, self.concepts)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_concept_domain(self, concept_id: str) -> Optional[str]:
        """Return the primary domain name for a concept, or ``None``."""
        lhv = self.concepts.get(concept_id)
        return lhv.primary_domain if lhv else None

    def stats_report(self) -> Dict[str, Any]:
        """Return a comprehensive statistics dictionary."""
        n = len(self.concepts)
        stability_classes: Dict[str, int] = {
            "crystallized": 0, "stable": 0, "active": 0, "volatile": 0
        }
        hybridization: Dict[str, int] = {"free": 0, "bonded": 0, "hybridized": 0}
        total_bonds = 0
        for lhv in self.concepts.values():
            stability_classes[lhv.stability_class] = (
                stability_classes.get(lhv.stability_class, 0) + 1
            )
            hybridization[lhv.hybridization_state] = (
                hybridization.get(lhv.hybridization_state, 0) + 1
            )
            total_bonds += len(lhv.bonds)

        return {
            "tick_count": self.tick_count,
            "n_concepts": n,
            "n_neighborhoods": len(self.neighborhoods),
            "n_domains": len(self.domains),
            "total_bonds": total_bonds // 2,  # each bond counted from both sides
            "avg_bonds_per_concept": (total_bonds / (2.0 * n)) if n > 0 else 0.0,
            "stability_classes": stability_classes,
            "hybridization_states": hybridization,
            "domains": [d.stats() for d in self.domains.values()],
            "neighborhoods": [n_.stats() for n_ in self.neighborhoods.values()],
        }
