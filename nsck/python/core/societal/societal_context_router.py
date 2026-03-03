"""
SocietalContextRouter — integrates societal HV knowledge into substrate queries.

When wired into NSCKSubstrate (via ``init_societal_world()``), every call to
``process()`` / ``ingest()`` passes through the router, which:

1. Finds the top-k nearest LivingHyperVectors to the query HV.
2. Activates matching concepts and spreads activation along bonds.
3. Retrieves the active community (cluster) containing the best match.
4. Returns a ``societal_context`` dict attached to SubstrateResult.

Usage (inside NSCKSubstrate)::

    router = SocietalContextRouter(manager=self._societal_manager)
    ctx = router.route(query_hv, task_tag=task_tag)
    # ctx is a dict suitable for SubstrateResult.societal_context
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from python.core.societal.living_hypervector import LivingHyperVector, _hv_cosine_sim
from python.core.societal.society_manager import SocietyManager

logger = logging.getLogger(__name__)


class SocietalContextRouter:
    """Route substrate queries through the societal knowledge graph.

    Parameters
    ----------
    manager:
        The SocietyManager instance holding all registered LHVs.
    top_k:
        Number of nearest neighbours to retrieve per query.
    activation_delta:
        How much to spike the matched concept's activation.
    min_similarity:
        Minimum similarity to include a concept in the context.
    auto_register:
        If True, automatically register previously unseen concept_ids
        as new LHVs when called from feedback().
    """

    def __init__(
        self,
        manager: SocietyManager,
        top_k: int = 5,
        activation_delta: float = 0.3,
        min_similarity: float = 0.55,
        auto_register: bool = True,
    ) -> None:
        self.manager = manager
        self.top_k = int(top_k)
        self.activation_delta = float(activation_delta)
        self.min_similarity = float(min_similarity)
        self.auto_register = bool(auto_register)

    # ------------------------------------------------------------------
    # Primary routing
    # ------------------------------------------------------------------

    def route(
        self,
        query_hv: Any,
        task_tag: str = "",
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Route a query HV through the society and return societal context.

        Parameters
        ----------
        query_hv:
            The situation hypervector for the current percept.
        task_tag:
            Task identifier (stored in metadata).
        extra_context:
            Optional extra key-value pairs merged into the returned dict.

        Returns
        -------
        Dict[str, Any]
            Societal context with keys:
            ``matches``, ``active_cluster``, ``cluster_concepts``,
            ``top_concept``, ``top_similarity``, ``activated_count``,
            ``epoch``.
        """
        if len(self.manager) == 0:
            return self._empty_context(task_tag, extra_context)

        # Find nearest neighbours
        neighbours: List[Tuple[str, float]] = self.manager.nearest_neighbors(
            query_hv, k=self.top_k
        )

        # Filter by min_similarity
        matches = [
            {"concept_id": cid, "similarity": round(sim, 6)}
            for cid, sim in neighbours
            if sim >= self.min_similarity
        ]

        # Activate matching concepts
        activated_count = 0
        for cid, sim in neighbours:
            if sim < self.min_similarity:
                continue
            delta = self.activation_delta * sim
            if self.manager.activate_concept(cid, delta=delta, spread=True):
                activated_count += 1

        # Top concept info
        top_concept = matches[0]["concept_id"] if matches else None
        top_similarity = matches[0]["similarity"] if matches else 0.0

        # Active cluster
        active_cluster: Optional[int] = None
        cluster_concepts: List[str] = []
        if top_concept is not None:
            lhv = self.manager.get(top_concept)
            if lhv is not None:
                active_cluster = lhv._cluster_id
                if active_cluster is not None:
                    # Find all cluster members (from last clustering result)
                    cluster_result = self.manager._cluster_cache.get(1.0)
                    if cluster_result is not None:
                        community = cluster_result.communities.get(active_cluster)
                        if community:
                            cluster_concepts = sorted(community)

        ctx: Dict[str, Any] = {
            "matches": matches,
            "active_cluster": active_cluster,
            "cluster_concepts": cluster_concepts,
            "top_concept": top_concept,
            "top_similarity": top_similarity,
            "activated_count": activated_count,
            "epoch": self.manager.epoch,
            "task_tag": task_tag,
        }
        if extra_context:
            ctx.update(extra_context)
        return ctx

    # ------------------------------------------------------------------
    # Feedback path: auto-register action concepts
    # ------------------------------------------------------------------

    def register_action(
        self,
        action: str,
        hv: Any,
        domain: str = "actions",
        reward: float = 0.0,
    ) -> bool:
        """Register (or reinforce) an action concept in the society.

        Called from NSCKSubstrate.feedback() to auto-populate action nodes.

        Parameters
        ----------
        action:
            Action string (concept_id).
        hv:
            Hypervector representing this action.
        domain:
            Domain label.
        reward:
            Positive reward spikes activation; negative decays it.

        Returns
        -------
        bool
            True if a new LHV was created; False if updated existing.
        """
        existing = self.manager.get(action)
        if existing is not None:
            delta = max(0.0, reward) * 0.3
            if delta > 0:
                existing.activate(delta, epoch=self.manager.epoch)
            return False

        if not self.auto_register:
            return False

        lhv = LivingHyperVector(
            concept_id=action,
            hv=hv,
            domain_path=[domain],
            role="leaf",
            initial_activation=max(0.1, reward),
            birth_epoch=self.manager.epoch,
            metadata={"source": "feedback", "reward": reward},
        )
        self.manager.register(lhv)
        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _empty_context(
        self,
        task_tag: str = "",
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ctx: Dict[str, Any] = {
            "matches": [],
            "active_cluster": None,
            "cluster_concepts": [],
            "top_concept": None,
            "top_similarity": 0.0,
            "activated_count": 0,
            "epoch": self.manager.epoch,
            "task_tag": task_tag,
        }
        if extra:
            ctx.update(extra)
        return ctx

    def get_community_summary(self) -> List[Dict[str, Any]]:
        """Return a summary of all current communities.

        Runs leiden_cluster at resolution=1.0 if not already cached.
        """
        result = self.manager.leiden_cluster(resolution=1.0)
        summary = []
        for cluster_id, members in sorted(result.communities.items()):
            # Compute mean activation for this community
            activations = [
                self.manager.get(cid).activation
                for cid in members
                if self.manager.get(cid) is not None
            ]
            summary.append({
                "cluster_id": cluster_id,
                "size": len(members),
                "concepts": sorted(members),
                "mean_activation": round(
                    float(np.mean(activations)) if activations else 0.0, 4
                ),
            })
        return summary
