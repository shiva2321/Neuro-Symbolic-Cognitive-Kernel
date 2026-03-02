"""
ValenceEngine — bond formation/breaking and context-conditioned retrieval for NSCK V5.

Bond strength is a weighted combination of:
  - HV cosine similarity (structural affinity)
  - valence compatibility  (motivational alignment)
  - domain affinity overlap (categorical proximity)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from python.core.societal.living_hypervector import LivingHyperVector, _hv_cosine_sim


class ValenceEngine:
    """Manages bond chemistry between LivingHyperVectors.

    Parameters
    ----------
    bond_threshold:
        Minimum bond strength to form a bond (default 0.30).
    break_threshold:
        Bond strength below which a bond is broken (default 0.10).
    max_bonds_per_concept:
        Hard cap on outgoing bonds per concept (default 20).
    """

    def __init__(
        self,
        bond_threshold: float = 0.30,
        break_threshold: float = 0.10,
        max_bonds_per_concept: int = 20,
    ) -> None:
        self.bond_threshold = float(bond_threshold)
        self.break_threshold = float(break_threshold)
        self.max_bonds_per_concept = int(max_bonds_per_concept)

    # ------------------------------------------------------------------
    # Core bond computation
    # ------------------------------------------------------------------

    def compute_bond_strength(
        self, lhv_a: LivingHyperVector, lhv_b: LivingHyperVector
    ) -> float:
        """Compute bond strength between two LHVs.

        Components
        ----------
        0.60 × HV cosine similarity
        0.20 × valence compatibility  (1 - |va - vb| / 2)
        0.20 × domain affinity overlap (Jaccard over shared top domains)
        """
        # --- structural ---
        sim = _hv_cosine_sim(lhv_a.hv, lhv_b.hv)
        # Normalise from [-1, 1] → [0, 1]
        struct = (sim + 1.0) / 2.0

        # --- valence ---
        val_compat = 1.0 - abs(lhv_a.valence - lhv_b.valence) / 2.0

        # --- domain overlap ---
        da = lhv_a.domain_affinities
        db = lhv_b.domain_affinities
        if da and db:
            shared = set(da) & set(db)
            union = set(da) | set(db)
            domain_overlap = len(shared) / len(union) if union else 0.0
        else:
            domain_overlap = 0.0

        strength = 0.60 * struct + 0.20 * val_compat + 0.20 * domain_overlap
        return float(np.clip(strength, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Bond formation / breaking
    # ------------------------------------------------------------------

    def try_form_bond(
        self, lhv_a: LivingHyperVector, lhv_b: LivingHyperVector
    ) -> Optional[float]:
        """Attempt to form a bond between two LHVs.

        Returns the bond strength if a bond was (re-)formed, else ``None``.
        """
        strength = self.compute_bond_strength(lhv_a, lhv_b)
        if strength < self.bond_threshold:
            return None

        # Enforce per-concept max-bond cap on both sides
        a_id, b_id = lhv_a.concept_id, lhv_b.concept_id
        if (
            len(lhv_a.bonds) >= self.max_bonds_per_concept
            and b_id not in lhv_a.bonds
        ):
            # Evict weakest bond before adding new one
            weakest = min(lhv_a.bonds, key=lhv_a.bonds.get)
            lhv_a.remove_bond(weakest)

        if (
            len(lhv_b.bonds) >= self.max_bonds_per_concept
            and a_id not in lhv_b.bonds
        ):
            weakest = min(lhv_b.bonds, key=lhv_b.bonds.get)
            lhv_b.remove_bond(weakest)

        lhv_a.add_bond(b_id, strength)
        lhv_b.add_bond(a_id, strength)
        return strength

    def try_break_bond(
        self, lhv_a: LivingHyperVector, lhv_b: LivingHyperVector
    ) -> bool:
        """Break the bond between two LHVs if it is weaker than break_threshold.

        Returns ``True`` if a bond was broken.
        """
        a_id, b_id = lhv_a.concept_id, lhv_b.concept_id
        current = lhv_a.bonds.get(b_id, 0.0)
        if current > 0.0 and current < self.break_threshold:
            lhv_a.remove_bond(b_id)
            lhv_b.remove_bond(a_id)
            return True
        return False

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    def bulk_initialize(
        self,
        concepts: List[LivingHyperVector],
        similarity_matrix: np.ndarray,
    ) -> int:
        """Initialize bonds for a list of concepts using a precomputed matrix.

        Parameters
        ----------
        concepts:
            List of LHVs to process.
        similarity_matrix:
            NxN float32 array of pairwise cosine similarities.

        Returns
        -------
        Number of bonds formed.
        """
        n = len(concepts)
        bonds_formed = 0
        for i in range(n):
            for j in range(i + 1, n):
                raw_sim = float(similarity_matrix[i, j]) if similarity_matrix.ndim == 2 else 0.0
                struct = (raw_sim + 1.0) / 2.0  # [-1,1] → [0,1]
                # Quick bond-strength approx (no full compute, just structural)
                strength = 0.60 * struct + 0.20 * 0.5  # assume neutral valence compat
                strength = float(np.clip(strength, 0.0, 1.0))
                if strength >= self.bond_threshold:
                    a_id = concepts[i].concept_id
                    b_id = concepts[j].concept_id
                    concepts[i].add_bond(b_id, strength)
                    concepts[j].add_bond(a_id, strength)
                    bonds_formed += 1
        return bonds_formed

    def update_hybridization(
        self, lhv: LivingHyperVector, context_hv: object
    ) -> None:
        """Update hybridization state based on current bonds and context."""
        n_bonds = len(lhv.bonds)
        if n_bonds == 0:
            lhv.hybridization_state = "free"
        elif n_bonds >= 3:
            lhv.hybridization_state = "hybridized"
        else:
            lhv.hybridization_state = "bonded"

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def context_conditioned_retrieval(
        self,
        query_hv: object,
        concepts: List[LivingHyperVector],
        context_hv: Optional[object] = None,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Retrieve top-k concepts by similarity, optionally conditioning on context.

        If *context_hv* is provided the query is blended with it (50/50 bundle)
        before measuring similarity.
        """
        if context_hv is not None:
            try:
                effective_query = query_hv.bundle(context_hv)
            except Exception:
                effective_query = query_hv
        else:
            effective_query = query_hv

        scores: List[Tuple[str, float]] = []
        for lhv in concepts:
            sim = _hv_cosine_sim(effective_query, lhv.hv)
            # Scale by activation to prefer active concepts
            score = 0.8 * sim + 0.2 * lhv.activation
            scores.append((lhv.concept_id, float(score)))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
