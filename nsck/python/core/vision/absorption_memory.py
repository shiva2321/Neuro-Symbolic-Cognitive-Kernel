"""AbsorptionMemory — stores HyperVector records absorbed from pretrained models."""
from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

import python.core.vsa.hypervec_shim as hypervec_rs


@dataclass
class AbsorptionRecord:
    """A single absorbed concept stored as a HyperVector with provenance."""

    hv: Any          # HyperVector
    label: str
    domain: str
    model_id: str
    confidence: float
    timestamp: float
    metadata: dict = field(default_factory=dict)


class AbsorptionMemory:
    """Persistent store of AbsorptionRecords with HV-similarity querying.

    Wraps SemanticMemory (when available) for concept-level operations while
    maintaining its own fast in-memory list of AbsorptionRecords.

    Parameters
    ----------
    semantic_memory: Optional SemanticMemory instance to mirror concepts into.
    """

    def __init__(self, semantic_memory=None) -> None:
        self._records: List[AbsorptionRecord] = []
        self._semantic_memory = semantic_memory

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def store(
        self,
        hv: Any,
        label: str,
        domain: str,
        model_id: str,
        confidence: float,
        metadata: Optional[Dict] = None,
    ) -> AbsorptionRecord:
        """Store an absorbed HyperVector record.

        Parameters
        ----------
        hv:         HyperVector for the concept.
        label:      Human-readable label / class name.
        domain:     Task domain (e.g. "imagenet", "coco").
        model_id:   Source pretrained model identifier.
        confidence: Absorption confidence score [0, 1].
        metadata:   Arbitrary provenance dict.

        Returns
        -------
        The created AbsorptionRecord.
        """
        rec = AbsorptionRecord(
            hv=hv,
            label=label,
            domain=domain,
            model_id=model_id,
            confidence=confidence,
            timestamp=time.time(),
            metadata=metadata or {},
        )
        self._records.append(rec)

        # Mirror into SemanticMemory when available
        if self._semantic_memory is not None:
            try:
                props = {
                    "domain": domain,
                    "model_id": model_id,
                    "confidence": confidence,
                    **rec.metadata,
                }
                self._semantic_memory.add_concept(label, props, hv_override=hv)
            except Exception:
                pass  # SemanticMemory is optional; never crash the store

        return rec

    def query_by_hv(
        self, query_hv: Any, top_k: int = 5
    ) -> List[AbsorptionRecord]:
        """Return the top-k records most similar to query_hv (Hamming similarity).

        Parameters
        ----------
        query_hv: HyperVector to compare against.
        top_k:    Maximum number of results to return.
        """
        if not self._records:
            return []

        scored: List[tuple] = []
        for rec in self._records:
            try:
                sim = rec.hv.similarity(query_hv)
            except Exception:
                sim = 0.0
            scored.append((sim, rec))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [rec for _, rec in scored[:top_k]]

    def query_by_domain(self, domain: str) -> List[AbsorptionRecord]:
        """Return all records belonging to *domain*."""
        return [r for r in self._records if r.domain == domain]

    # ------------------------------------------------------------------
    # Statistics & persistence
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict:
        """Return summary statistics about stored records."""
        domains: Dict[str, int] = {}
        models: Dict[str, int] = {}
        for r in self._records:
            domains[r.domain] = domains.get(r.domain, 0) + 1
            models[r.model_id] = models.get(r.model_id, 0) + 1

        confidences = [r.confidence for r in self._records]
        return {
            "total_records": len(self._records),
            "domains": domains,
            "models": models,
            "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
            "min_confidence": float(np.min(confidences)) if confidences else 0.0,
            "max_confidence": float(np.max(confidences)) if confidences else 0.0,
        }

    def save(self, path: str) -> None:
        """Pickle the absorption memory to *path*."""
        with open(path, "wb") as fh:
            pickle.dump(self._records, fh)

    def load(self, path: str) -> None:
        """Load records from a previously saved pickle file."""
        with open(path, "rb") as fh:
            self._records = pickle.load(fh)
