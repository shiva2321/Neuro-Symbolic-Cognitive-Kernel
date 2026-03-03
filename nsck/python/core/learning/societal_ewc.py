"""
Societal-Guided Elastic Weight Consolidation — NSCK V27
=========================================================
Integrates the Societal Hypervector Knowledge Representation (V26) with the
Elastic Weight Consolidation (EWC) continual-learning framework (V16).

Insight
-------
Concepts that are *central* in the societal knowledge graph — those with high
bond degree, high cumulative activation, or that occupy a cluster hub — are
likely cross-task knowledge that should be protected from overwriting during
subsequent task learning.  The ``SocietalEWCCombiner`` computes a per-concept
societal-centrality score and uses it to **boost** the EWC Fisher-information
diagonal for those concepts, making the EWC penalty larger and thereby
protecting them more strongly.

Centrality score (∈ [0, 1])
---------------------------
  score = w_d · degree_norm
        + w_a · activation
        + w_c · cluster_centrality_norm

where
  • degree_norm         = bond-degree / max_degree  (in-graph connectivity)
  • activation          = concept.activation        (recent salience, ∈ [0,1])
  • cluster_centrality  = fraction of cluster owned by this concept as a hub
                          proxy (1/cluster_size, normalised across all clusters)

Usage
-----
::

    from python.core.learning.continual_learning import ContinualLearner
    from python.core.societal.society_manager import SocietyManager
    from python.core.learning.societal_ewc import SocietalEWCCombiner

    combiner = SocietalEWCCombiner(centrality_weight=0.5)

    # After ContinualLearner.compute_importance() has been called:
    combiner.apply_societal_boost(learner, task_tag="nav", manager=soc_mgr)

    # Protected concepts now include both high-Fisher-importance concepts AND
    # societal-central concepts.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, Optional

import numpy as np

if TYPE_CHECKING:
    from python.core.learning.continual_learning import ContinualLearner
    from python.core.societal.society_manager import SocietyManager

logger = logging.getLogger("nsck.learning.societal_ewc")

__all__ = ["SocietalEWCCombiner", "compute_societal_centrality"]


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def compute_societal_centrality(
    concept_id: str,
    manager: "SocietyManager",
    weight_degree: float = 0.4,
    weight_activation: float = 0.4,
    weight_cluster: float = 0.2,
) -> float:
    """Return a scalar centrality score ∈ [0, 1] for *concept_id*.

    Parameters
    ----------
    concept_id:
        Identifier of the concept whose centrality is requested.
    manager:
        :class:`~python.core.societal.society_manager.SocietyManager` holding
        the current societal world.
    weight_degree:
        Weight for the bond-degree component.
    weight_activation:
        Weight for the current activation component.
    weight_cluster:
        Weight for the cluster-centrality component.

    Returns
    -------
    float
        Score in [0, 1].  Returns 0.0 if the concept is not registered.
    """
    lhv = manager.get(concept_id)
    if lhv is None:
        return 0.0

    # --- Degree component (bond count / max bond count in society) ---
    degree = len(lhv._bonds)
    max_degree = max(
        (len(other._bonds) for other in manager),
        default=1,
    )
    degree_norm = degree / max_degree if max_degree > 0 else 0.0

    # --- Activation component (already in [0, 1]) ---
    activation = float(lhv.activation)

    # --- Cluster-centrality component ---
    # Proxy: 1/cluster_size (concepts in small clusters are more "central"
    # relative to their cluster, but cross-cluster hubs want HIGH degree).
    # We invert: concept is central if its cluster is LARGE (hub of a big
    # community), so use cluster_size / total_concepts.
    cluster_centrality = 0.0
    n_total = len(manager)
    if lhv._cluster_id is not None and n_total > 0:
        cluster_id = lhv._cluster_id
        cluster_size = sum(
            1 for other in manager if other._cluster_id == cluster_id
        )
        cluster_centrality = cluster_size / n_total
    elif n_total > 0:
        cluster_centrality = 1.0 / n_total  # unclustered — minimal

    score = (
        weight_degree * degree_norm
        + weight_activation * activation
        + weight_cluster * cluster_centrality
    )
    return float(min(1.0, max(0.0, score)))


class SocietalEWCCombiner:
    """Boosts EWC Fisher importance weights using societal centrality scores.

    Parameters
    ----------
    centrality_weight:
        Multiplier controlling how much the societal centrality boosts the
        Fisher diagonal.  The effective Fisher weight for a concept is::

            F_boosted = F_original * (1 + centrality_weight * centrality_score)

        With ``centrality_weight=0.5`` a concept with centrality=1.0 has its
        Fisher weight multiplied by 1.5 (50% boost).
    degree_weight:
        Sub-weight for bond-degree in the centrality score.
    activation_weight:
        Sub-weight for activation level in the centrality score.
    cluster_weight:
        Sub-weight for cluster-centrality in the centrality score.
    """

    def __init__(
        self,
        centrality_weight: float = 0.5,
        degree_weight: float = 0.4,
        activation_weight: float = 0.4,
        cluster_weight: float = 0.2,
    ) -> None:
        if not 0.0 <= centrality_weight:
            raise ValueError("centrality_weight must be non-negative")
        self.centrality_weight = centrality_weight
        self.degree_weight = degree_weight
        self.activation_weight = activation_weight
        self.cluster_weight = cluster_weight

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_centrality_map(
        self, manager: "SocietyManager"
    ) -> Dict[str, float]:
        """Compute centrality scores for all registered concepts.

        Returns
        -------
        dict
            Mapping concept_id → centrality ∈ [0, 1].
        """
        return {
            lhv.concept_id: compute_societal_centrality(
                lhv.concept_id,
                manager,
                weight_degree=self.degree_weight,
                weight_activation=self.activation_weight,
                weight_cluster=self.cluster_weight,
            )
            for lhv in manager
        }

    def boost_task_importance(
        self,
        task_tag: str,
        learner: "ContinualLearner",
        manager: "SocietyManager",
    ) -> Dict[str, float]:
        """Boost Fisher importance for *task_tag* using societal centrality.

        For each concept registered in *manager* that also appears in
        ``learner.tasks[task_tag].importance_weights``, the Fisher diagonal
        value is multiplied by ``(1 + centrality_weight × centrality_score)``.

        Parameters
        ----------
        task_tag:
            Task whose importance weights to modify.
        learner:
            :class:`~python.core.learning.continual_learning.ContinualLearner`
            whose ``tasks[task_tag].importance_weights`` is updated in-place.
        manager:
            Current :class:`~python.core.societal.society_manager.SocietyManager`.

        Returns
        -------
        dict
            Mapping concept_id → boost_factor applied (1.0 = no boost).
            Only concepts where a boost was applied (factor > 1.0) are
            returned.
        """
        if task_tag not in learner.tasks:
            logger.debug(
                "[SoCL] task_tag %r not in learner — skipping boost", task_tag
            )
            return {}

        centrality_map = self.compute_centrality_map(manager)
        task_mem = learner.tasks[task_tag]
        boosts: Dict[str, float] = {}

        for concept_id, centrality in centrality_map.items():
            if centrality <= 0.0:
                continue
            if concept_id not in task_mem.importance_weights:
                # Concept not in this task's Fisher weights — skip
                continue

            factor = 1.0 + self.centrality_weight * centrality
            task_mem.importance_weights[concept_id] = (
                task_mem.importance_weights[concept_id] * factor
            )
            if factor > 1.0:
                boosts[concept_id] = factor

        logger.debug(
            "[SoCL] Boosted %d/%d concepts for task %r",
            len(boosts),
            len(centrality_map),
            task_tag,
        )
        return boosts

    def apply_societal_boost(
        self,
        learner: "ContinualLearner",
        task_tag: str,
        manager: "SocietyManager",
    ) -> Dict[str, float]:
        """Convenience wrapper: boost all tasks OR just *task_tag*.

        Parameters
        ----------
        learner:
            The EWC learner to modify.
        task_tag:
            If provided, only boost the named task.  Pass an empty string to
            boost all registered tasks.
        manager:
            Current societal world.

        Returns
        -------
        dict
            Mapping task_tag → {concept_id: boost_factor} for all boosted
            tasks.
        """
        if task_tag:
            boosts = self.boost_task_importance(task_tag, learner, manager)
            return {task_tag: boosts} if boosts else {}

        # Boost all tasks
        all_boosts: Dict[str, Dict[str, float]] = {}
        for tt in list(learner.tasks.keys()):
            b = self.boost_task_importance(tt, learner, manager)
            if b:
                all_boosts[tt] = b
        return all_boosts

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_top_central_concepts(
        self,
        manager: "SocietyManager",
        top_k: int = 10,
    ) -> list:
        """Return the top-K most central concepts as (concept_id, score) pairs."""
        cmap = self.compute_centrality_map(manager)
        sorted_items = sorted(cmap.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:top_k]
