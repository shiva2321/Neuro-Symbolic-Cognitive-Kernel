"""
CrossModalCorrelationLearner — NSCK V11
=======================================
Learns associations between different input modalities using Hebbian-style VSA binding.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import python.core.vsa.hypervec_shim as hypervec_rs


class CrossModalCorrelationLearner:
    """
    Learns associations between different input modalities.

    When NSCK processes multimodal input (e.g. image + text together),
    this module learns which text concepts tend to co-occur with which
    visual/audio/numeric patterns.

    Uses Hebbian-style VSA binding:
    - co-occurrence of (HV_text, HV_image) → strengthen their association
    - at retrieval: given HV_image, predict HV_text and vice versa
    """

    def __init__(self, hv_dim: int = 10240, learning_rate: float = 0.01) -> None:
        self.hv_dim = hv_dim
        self.learning_rate = learning_rate

        # (modality_a, modality_b) → accumulated association HV
        self._assoc_hvs: Dict[Tuple[str, str], hypervec_rs.HyperVector] = {}
        # (modality_a, modality_b) → list of (HV_a, HV_b)
        self._observations: Dict[Tuple[str, str], List[Tuple[hypervec_rs.HyperVector, hypervec_rs.HyperVector]]] = defaultdict(list)
        # Pair co-occurrence counts
        self._counts: Dict[Tuple[str, str], int] = defaultdict(int)
        # Context tags seen
        self._context_tags: List[str] = []

    def observe(
        self,
        modality_hvs: Dict[str, hypervec_rs.HyperVector],
        context_tag: str = "",
    ) -> None:
        """
        Called whenever multiple modalities arrive together.
        Learns pairwise associations between all modality HVs present.
        """
        if context_tag and context_tag not in self._context_tags:
            self._context_tags.append(context_tag)

        modalities = list(modality_hvs.keys())
        for i, mod_a in enumerate(modalities):
            for mod_b in modalities[i + 1:]:
                hv_a = modality_hvs[mod_a]
                hv_b = modality_hvs[mod_b]
                pair_key = (mod_a, mod_b)
                pair_key_rev = (mod_b, mod_a)

                # Store the observation for later retrieval
                self._observations[pair_key].append((hv_a, hv_b))
                self._observations[pair_key_rev].append((hv_b, hv_a))
                self._counts[pair_key] += 1
                self._counts[pair_key_rev] += 1

                # Accumulate association HV via bundling (Hebbian-style)
                assoc_hv = hv_a.bind(hv_b) if hasattr(hv_a, 'bind') else hv_a.bundle(hv_b)
                if pair_key not in self._assoc_hvs:
                    self._assoc_hvs[pair_key] = assoc_hv
                else:
                    self._assoc_hvs[pair_key] = self._assoc_hvs[pair_key].bundle(assoc_hv)
                if pair_key_rev not in self._assoc_hvs:
                    self._assoc_hvs[pair_key_rev] = assoc_hv
                else:
                    self._assoc_hvs[pair_key_rev] = self._assoc_hvs[pair_key_rev].bundle(assoc_hv)

    def predict_modality(
        self,
        known_hv: hypervec_rs.HyperVector,
        known_modality: str,
        target_modality: str,
        top_k: int = 3,
    ) -> List[Tuple[hypervec_rs.HyperVector, float]]:
        """
        Given a known HV in one modality, predict likely HVs in another modality.
        """
        pair_key = (known_modality, target_modality)
        observations = self._observations.get(pair_key, [])
        if not observations:
            return []

        results = []
        for hv_a, hv_b in observations:
            sim = known_hv.similarity(hv_a)
            results.append((hv_b, float(sim)))

        # Sort by similarity descending, return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_correlation_strength(
        self,
        hv_a: hypervec_rs.HyperVector,
        modality_a: str,
        hv_b: hypervec_rs.HyperVector,
        modality_b: str,
    ) -> float:
        """
        How strongly correlated are these two HVs from different modalities?
        Returns similarity score in [0, 1].
        """
        pair_key = (modality_a, modality_b)
        assoc_hv = self._assoc_hvs.get(pair_key)
        if assoc_hv is None:
            return 0.0

        # Unbind known_hv from association and compare with target
        try:
            predicted = assoc_hv.bind(hv_a) if hasattr(assoc_hv, 'bind') else assoc_hv
            return float(predicted.similarity(hv_b))
        except Exception:
            return 0.0

    def get_statistics(self) -> Dict[str, Any]:
        """Return learning statistics."""
        return {
            "pairs_seen": dict(self._counts),
            "associations_learned": len(self._assoc_hvs),
            "total_observations": sum(len(v) for v in self._observations.values()),
            "context_tags": list(self._context_tags),
        }
