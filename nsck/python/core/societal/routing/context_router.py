"""
SocietalContextRouter — domain-aware routing for NSCK V5.

Routes queries through the societal world, detecting relevant domains
and performing domain-weighted spreading activation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from python.core.societal.living_hypervector import _hv_cosine_sim

logger = logging.getLogger("nsck.context_router")


class SocietalContextRouter:
    """Routes queries through societal domains for GWT-integrated reasoning.

    Parameters
    ----------
    world:
        The SocietalKnowledgeWorld to route through.
    border_zone_radius:
        Number of bond hops from a border concept to include in the border zone (default 2).
    spreading_decay:
        Activation fraction retained per spreading hop (default 0.30).
    max_hops:
        Maximum spreading activation hops (default 3).
    """

    def __init__(
        self,
        world: Any,
        border_zone_radius: int = 2,
        spreading_decay: float = 0.30,
        max_hops: int = 3,
    ) -> None:
        self.world = world
        self.border_zone_radius = int(border_zone_radius)
        self.spreading_decay = float(spreading_decay)
        self.max_hops = int(max_hops)

    # ------------------------------------------------------------------

    def detect_domain(
        self, query_hv: Any, top_k_candidates: int = 5
    ) -> Tuple[Optional[str], float]:
        """Detect the most relevant domain by comparing query to city-hall HVs.

        Returns
        -------
        (domain_id, confidence) — confidence in [0, 1].
        """
        domains = self.world.domains
        if not domains:
            return None, 0.0

        best_domain: Optional[str] = None
        best_score: float = -1.0

        for domain in domains.values():
            if domain.city_hall_hv is None:
                continue
            sim = _hv_cosine_sim(query_hv, domain.city_hall_hv)
            # Normalise [-1, 1] → [0, 1]
            sim_norm = (sim + 1.0) / 2.0
            if sim_norm > best_score:
                best_score = sim_norm
                best_domain = domain.domain_id

        return best_domain, float(best_score)

    # ------------------------------------------------------------------

    def border_zone_routing(
        self, query_hv: Any, domain_id: str
    ) -> List[Tuple[str, float]]:
        """Find concepts in the border zone with relevance scores.

        Border concepts are those with bonds crossing into other domains.
        We score them by query similarity + bridge score.

        Returns
        -------
        List of (concept_id, relevance) sorted descending.
        """
        domain = self.world.domains.get(domain_id)
        if domain is None:
            return []

        # Collect border concepts from all neighbourhoods in this domain
        border_set: Set[str] = set()
        for nbhd_id in domain.neighborhood_ids:
            nbhd = self.world.neighborhoods.get(nbhd_id)
            if nbhd is not None:
                border_set |= nbhd.border_concepts

        results: List[Tuple[str, float]] = []
        for cid in border_set:
            lhv = self.world.concepts.get(cid)
            if lhv is None:
                continue
            sim = _hv_cosine_sim(query_hv, lhv.hv)
            sim_norm = (sim + 1.0) / 2.0
            bridge_score = domain.compute_bridge_score(cid, self.world.concepts)
            relevance = 0.7 * sim_norm + 0.3 * bridge_score
            results.append((cid, float(relevance)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    # ------------------------------------------------------------------

    def domain_weighted_spreading_activation(
        self,
        seed_concept_id: str,
        domain_id: Optional[str] = None,
        n_hops: int = 3,
        top_k: int = 20,
    ) -> Dict[str, float]:
        """Spreading activation from a seed concept, weighted by domain.

        Parameters
        ----------
        seed_concept_id:
            Starting concept.
        domain_id:
            If provided, boost activation for concepts in this domain.
        n_hops:
            Maximum propagation hops.
        top_k:
            Number of top activated concepts to return.

        Returns
        -------
        Dict concept_id → activation_score (top_k entries).
        """
        concepts = self.world.concepts
        seed = concepts.get(seed_concept_id)
        if seed is None:
            return {}

        # Get domain member set
        domain_members: Set[str] = set()
        if domain_id and domain_id in self.world.domains:
            domain = self.world.domains[domain_id]
            for nbhd_id in domain.neighborhood_ids:
                nbhd = self.world.neighborhoods.get(nbhd_id)
                if nbhd:
                    domain_members |= nbhd.concept_ids

        activations: Dict[str, float] = {seed_concept_id: 1.0}
        frontier: Dict[str, float] = {seed_concept_id: 1.0}

        for hop in range(min(n_hops, self.max_hops)):
            decay = self.spreading_decay ** (hop + 1)
            new_frontier: Dict[str, float] = {}
            for cid, act in frontier.items():
                lhv = concepts.get(cid)
                if lhv is None:
                    continue
                for bonded_id, bond_w in lhv.bonds.items():
                    spread = act * bond_w * decay
                    # Domain affinity boost
                    if domain_members and bonded_id in domain_members:
                        bonded_lhv = concepts.get(bonded_id)
                        if bonded_lhv:
                            aff = bonded_lhv.get_affinity(
                                self.world.domains[domain_id].name
                                if domain_id in self.world.domains else ""
                            )
                            spread *= (1.0 + 0.5 * aff)
                    current = activations.get(bonded_id, 0.0)
                    if spread > current:
                        activations[bonded_id] = spread
                        new_frontier[bonded_id] = spread
            frontier = new_frontier
            if not frontier:
                break

        # Return top_k
        sorted_acts = sorted(activations.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_acts[:top_k])

    # ------------------------------------------------------------------

    def route(self, query_hv: Any, top_k: int = 10) -> Dict[str, Any]:
        """Full routing pipeline.

        Returns
        -------
        Dict with: domain_id, confidence, concepts (list of (id, score)),
        border_zone_active, routing_trace.
        """
        domain_id, confidence = self.detect_domain(query_hv)

        trace: Dict[str, Any] = {
            "detected_domain": domain_id,
            "domain_confidence": confidence,
        }

        # Direct query
        results = self.world.query(query_hv, top_k=top_k, domain_filter=domain_id)

        # Border zone check
        border_zone_active = False
        if domain_id and confidence < 0.65:
            border = self.border_zone_routing(query_hv, domain_id)
            if border:
                border_zone_active = True
                trace["border_zone_concepts"] = [b[0] for b in border[:5]]
                # Blend border zone results
                border_dict = dict(border)
                merged: Dict[str, float] = dict(results)
                for cid, rel in border_dict.items():
                    existing = merged.get(cid, 0.0)
                    merged[cid] = max(existing, rel * 0.8)
                results = sorted(merged.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Spreading activation from top result
        if results:
            top_cid = results[0][0]
            spread = self.domain_weighted_spreading_activation(
                top_cid, domain_id, n_hops=2, top_k=top_k
            )
            trace["spreading_activation_top"] = list(spread.keys())[:5]

        return {
            "domain_id": domain_id,
            "confidence": confidence,
            "concepts": list(results),
            "border_zone_active": border_zone_active,
            "routing_trace": trace,
        }

    # ------------------------------------------------------------------

    def update_coalition_scores(
        self, coalitions: List[Any], world_stats: Dict[str, Any]
    ) -> List[Any]:
        """Boost coalition salience if content mentions activated concepts.

        Parameters
        ----------
        coalitions:
            List of Coalition-like objects (dataclass or dict) with
            ``source``, ``content``, ``base_salience`` fields.
        world_stats:
            Output of world.stats_report().

        Returns
        -------
        The same list with salience updated in-place.
        """
        active_concepts = {
            cid
            for cid, lhv in self.world.concepts.items()
            if lhv.activation > 0.5
        }

        for coal in coalitions:
            content = (
                coal.get("content", "") if isinstance(coal, dict)
                else getattr(coal, "content", "")
            )
            content_str = str(content).lower()
            boost = sum(
                0.1 for cid in active_concepts if cid.lower() in content_str
            )
            if boost > 0:
                if isinstance(coal, dict):
                    coal["base_salience"] = min(1.0, coal.get("base_salience", 0.0) + boost)
                else:
                    try:
                        coal.base_salience = min(1.0, getattr(coal, "base_salience", 0.0) + boost)
                    except AttributeError:
                        pass

        return coalitions
