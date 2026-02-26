"""
MultimodalFuser — fuses multiple PerceptPackets from different modalities
into a single unified PerceptPacket using VSA bundle operations.

This is the STEP 4 implementation of multimodal fusion.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import python.core.vsa.hypervec_shim as hypervec_rs

from python.core.types.percept_packet import PerceptPacket


class MultimodalFuser:
    """Fuses a list of :class:`PerceptPacket` objects into one.

    Algorithm
    ---------
    1. Bundle all situation HVs (VSA superposition).
    2. Merge entity HVs: same key → bundle, new key → add.
    3. Collect all relation triples (deduplicated by subject+pred+obj).
    4. Union all active predicates.
    5. Average confidence scores.
    6. Merge raw_state dicts (later packets overwrite earlier on conflict).
    """

    def fuse(self, packets: List[PerceptPacket]) -> PerceptPacket:
        """Fuse *packets* into a single multimodal PerceptPacket.

        Parameters
        ----------
        packets : list of PerceptPacket
            At least one packet is required.

        Returns
        -------
        PerceptPacket
            Combined percept with ``modality="multimodal"``.

        Raises
        ------
        ValueError
            If *packets* is empty.
        """
        if not packets:
            raise ValueError("MultimodalFuser.fuse() requires at least one packet")

        if len(packets) == 1:
            return packets[0]

        # 1. Bundle situation HVs
        situation_hv = packets[0].situation_hv
        for p in packets[1:]:
            situation_hv = situation_hv.bundle(p.situation_hv)

        # 2. Merge entity HVs
        entity_hvs: Dict[str, hypervec_rs.HyperVector] = {}
        for p in packets:
            for key, hv in p.entity_hvs.items():
                if key in entity_hvs:
                    entity_hvs[key] = entity_hvs[key].bundle(hv)
                else:
                    entity_hvs[key] = hv

        # 3. Collect relation triples (deduplicate by (subj, pred, obj))
        seen_triples: set = set()
        relation_hvs = []
        for p in packets:
            for subj, pred, obj, triple_hv in p.relation_hvs:
                key = (subj, pred, obj)
                if key not in seen_triples:
                    seen_triples.add(key)
                    relation_hvs.append((subj, pred, obj, triple_hv))

        # 4. Union all active predicates
        active_predicates: frozenset = frozenset().union(*[p.active_predicates for p in packets])

        # 5. Average confidence
        confidence = sum(p.confidence for p in packets) / len(packets)

        # 6. Merge raw_state dicts (last writer wins on key conflict)
        raw_state: Dict[str, Any] = {}
        for p in packets:
            if p.raw_state:
                raw_state.update(p.raw_state)

        adapter_trace = {
            "source_modalities": [p.modality for p in packets],
            "source_adapters": [p.adapter_name for p in packets],
            "packet_count": len(packets),
        }

        return PerceptPacket(
            modality="multimodal",
            timestamp=time.time(),
            situation_hv=situation_hv,
            entity_hvs=entity_hvs,
            relation_hvs=relation_hvs,
            active_predicates=active_predicates,
            confidence=confidence,
            raw_state=raw_state if raw_state else None,
            adapter_name="MultimodalFuser",
            adapter_trace=adapter_trace,
        )
