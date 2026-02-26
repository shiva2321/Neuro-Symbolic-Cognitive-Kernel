"""
PerceptPacket — the universal modality-agnostic percept contract.

Every input to the cognitive engine is normalised into a PerceptPacket
before the reasoning core sees it.  This decouples perception from
cognition and allows any modality (text, numeric, dict, SNN, multimodal,
sensor stream) to be processed identically by the core.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

import python.core.vsa.hypervec_shim as hypervec_rs


@dataclass(frozen=True)
class PerceptPacket:
    """Immutable, modality-agnostic percept contract.

    Every ModalityAdapter must produce exactly one PerceptPacket.
    The cognitive engine core only reads from PerceptPackets, never
    from raw modality-specific data, ensuring clean separation between
    perception and reasoning.

    Attributes
    ----------
    modality : str
        One of ``"text"``, ``"numeric"``, ``"dict"``, ``"snn"``,
        ``"multimodal"``, or ``"stream"``.
    timestamp : float
        Unix timestamp when the percept was created.
    situation_hv : HyperVector
        Single bundled HV representing the entire situation.
    entity_hvs : dict
        Named concepts → HyperVector.
    relation_hvs : list
        ``(subject, predicate, object, triple_hv)`` tuples.
    active_predicates : frozenset
        Grounded symbolic predicates (e.g. ``{"DANGER_UP", "TARGET_NEAR"}``).
    confidence : float
        Overall encoding confidence in ``[0.0, 1.0]``.
    raw_state : dict or None
        Original raw input dict (for backward-compatibility logging).
    adapter_name : str
        Name of the adapter that produced this packet.
    adapter_trace : dict
        Arbitrary adapter-specific metadata for glass-box transparency.
    """

    modality: str
    timestamp: float
    situation_hv: hypervec_rs.HyperVector
    entity_hvs: Dict[str, hypervec_rs.HyperVector]
    relation_hvs: List[Tuple[str, str, str, hypervec_rs.HyperVector]]
    active_predicates: FrozenSet[str]
    confidence: float
    raw_state: Optional[Dict[str, Any]]
    adapter_name: str
    adapter_trace: Dict[str, Any]

    # -----------------------------------------------------------------
    # Convenience factory
    # -----------------------------------------------------------------

    @staticmethod
    def make(
        modality: str,
        situation_hv: hypervec_rs.HyperVector,
        active_predicates: FrozenSet[str],
        *,
        entity_hvs: Optional[Dict[str, hypervec_rs.HyperVector]] = None,
        relation_hvs: Optional[List[Tuple[str, str, str, hypervec_rs.HyperVector]]] = None,
        confidence: float = 1.0,
        raw_state: Optional[Dict[str, Any]] = None,
        adapter_name: str = "unknown",
        adapter_trace: Optional[Dict[str, Any]] = None,
    ) -> "PerceptPacket":
        """Convenience constructor — all optional fields default to empty."""
        return PerceptPacket(
            modality=modality,
            timestamp=time.time(),
            situation_hv=situation_hv,
            entity_hvs=entity_hvs or {},
            relation_hvs=relation_hvs or [],
            active_predicates=active_predicates,
            confidence=confidence,
            raw_state=raw_state,
            adapter_name=adapter_name,
            adapter_trace=adapter_trace or {},
        )
