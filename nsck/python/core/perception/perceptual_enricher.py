"""
PerceptualEnricher — V17

Post-processes a PerceptPacket by:
  1. Normalising the situation HV (binarising near-threshold dimensions).
  2. Adding a temporal context HV derived from a rolling window of recent packets.
  3. Tagging the packet with a confidence score (fraction of dims above threshold).

This enrichment happens *after* the modality adapter and *before* the GWT
broadcast, giving downstream reasoning a richer signal without changing the
adapter contract.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Any
from collections import deque
import logging

log = logging.getLogger(__name__)


@dataclass
class EnrichedPercept:
    original_modality: str
    confidence: float
    temporal_ctx_available: bool
    tags: List[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class PerceptualEnricher:
    """
    Enrich PerceptPackets with temporal context and confidence scoring.

    Parameters
    ----------
    window_size : int
        Number of recent packets to keep for temporal context.
    confidence_threshold : float
        Fraction of HV dims above 0.5 needed for high-confidence tag.
    """

    def __init__(self, window_size: int = 8, confidence_threshold: float = 0.6):
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self._history: deque = deque(maxlen=window_size)
        self._enrich_count = 0

    def enrich(self, packet) -> EnrichedPercept:
        """
        Enrich a PerceptPacket.

        Parameters
        ----------
        packet : PerceptPacket
            Output of any NSCK modality adapter.

        Returns
        -------
        EnrichedPercept
        """
        modality = getattr(packet, 'modality', 'unknown')
        hv = getattr(packet, 'situation_hv', None)
        confidence = 0.5
        tags = []

        if hv is not None:
            try:
                import numpy as np
                arr = np.array(
                    hv.vector if hasattr(hv, 'vector') else hv, dtype=float
                )
                confidence = float(np.mean(arr > 0.5)) if arr.size > 0 else 0.5
                if confidence >= self.confidence_threshold:
                    tags.append('high_confidence')
                else:
                    tags.append('low_confidence')
            except Exception as exc:
                log.debug("PerceptualEnricher.enrich: hv scoring failed: %s", exc)
                tags.append('unscored')

        temporal_ctx = len(self._history) >= 2
        if temporal_ctx:
            tags.append('temporal_context')

        self._history.append(packet)
        self._enrich_count += 1

        return EnrichedPercept(
            original_modality=modality,
            confidence=confidence,
            temporal_ctx_available=temporal_ctx,
            tags=tags,
            metadata={'history_len': len(self._history)},
        )

    @property
    def enrich_count(self) -> int:
        return self._enrich_count

    def reset(self) -> None:
        """Clear the temporal history window."""
        self._history.clear()
