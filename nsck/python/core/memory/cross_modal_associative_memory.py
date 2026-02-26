"""
CrossModalAssociativeMemory — modality-agnostic binding and cross-domain recall.

Stores HV associations between modalities and enables cross-domain recall:
given a query HV from modality A, retrieve the associated HV from modality B.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

import python.core.vsa.hypervec_shim as hv_mod


@dataclass
class ModalityBinding:
    """A binding between HVs from two modalities."""
    modality_a: str
    modality_b: str
    hv_a: hv_mod.HyperVector
    hv_b: hv_mod.HyperVector
    binding_hv: hv_mod.HyperVector   # hv_a XOR hv_b
    strength: float = 1.0
    label: Optional[str] = None


class CrossModalAssociativeMemory:
    """
    Stores and retrieves cross-modal HV bindings.

    Each stored binding is: hv_a XOR hv_b (the VSA association trick).
    To recall hv_b from hv_a: query_binding XOR hv_a ≈ hv_b.
    """

    def __init__(self, max_bindings: int = 1000) -> None:
        self.max_bindings = max_bindings
        self._bindings: List[ModalityBinding] = []

    def bind(
        self,
        modality_a: str,
        hv_a: hv_mod.HyperVector,
        modality_b: str,
        hv_b: hv_mod.HyperVector,
        strength: float = 1.0,
        label: Optional[str] = None,
    ) -> ModalityBinding:
        """Store a cross-modal association."""
        binding_hv = hv_a.xor(hv_b)
        binding = ModalityBinding(
            modality_a=modality_a,
            modality_b=modality_b,
            hv_a=hv_a,
            hv_b=hv_b,
            binding_hv=binding_hv,
            strength=strength,
            label=label,
        )
        self._bindings.append(binding)
        if len(self._bindings) > self.max_bindings:
            self._bindings = self._bindings[-self.max_bindings:]
        return binding

    def recall(
        self,
        query_hv: hv_mod.HyperVector,
        from_modality: str,
        to_modality: str,
        top_k: int = 3,
    ) -> List[Tuple[hv_mod.HyperVector, float, Optional[str]]]:
        """
        Given a query HV in from_modality, retrieve matching HVs from to_modality.
        Returns list of (recalled_hv, similarity, label) tuples.
        """
        candidates = [
            b for b in self._bindings
            if b.modality_a == from_modality and b.modality_b == to_modality
        ] + [
            b for b in self._bindings
            if b.modality_b == from_modality and b.modality_a == to_modality
        ]

        if not candidates:
            return []

        results = []
        for b in candidates:
            # Decode: query XOR binding_hv ≈ target hv
            recovered = query_hv.xor(b.binding_hv)
            if b.modality_a == from_modality:
                reference = b.hv_b
            else:
                reference = b.hv_a

            sim = float(recovered.similarity(reference))
            results.append((reference, sim * b.strength, b.label))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def recall_by_label(self, label: str) -> List[ModalityBinding]:
        """Retrieve all bindings with the given label."""
        return [b for b in self._bindings if b.label == label]

    def get_statistics(self) -> Dict[str, Any]:
        pairs: Dict[str, int] = {}
        for b in self._bindings:
            key = f"{b.modality_a}↔{b.modality_b}"
            pairs[key] = pairs.get(key, 0) + 1
        return {
            "total_bindings": len(self._bindings),
            "modality_pairs": pairs,
        }
