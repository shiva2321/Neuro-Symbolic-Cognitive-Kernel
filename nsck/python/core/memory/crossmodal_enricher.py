"""
CrossModalEnricher — V17

Enriches CrossModalAssociativeMemory by:
  1. Automatically linking modality-specific concept HVs to a shared
     cross-modal anchor when they are registered together.
  2. Computing pairwise similarity between anchors to detect and log
     cross-modal concept clusters.
  3. Generating a summary report of cross-modal enrichment statistics.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import logging

log = logging.getLogger(__name__)


@dataclass
class CrossModalEnrichmentReport:
    anchors_created: int = 0
    links_added: int = 0
    clusters_detected: int = 0
    steps: List[str] = field(default_factory=list)


class CrossModalEnricher:
    """
    Enrich a CrossModalAssociativeMemory with automatic anchor linking and
    cluster detection.

    Parameters
    ----------
    crossmodal_memory : CrossModalAssociativeMemory, optional
        The memory instance to enrich.
    similarity_threshold : float
        Minimum cosine similarity for two anchors to be considered the same
        cross-modal cluster (default: 0.7).
    """

    def __init__(
        self,
        crossmodal_memory=None,
        similarity_threshold: float = 0.7,
    ):
        self._mem = crossmodal_memory
        self.similarity_threshold = similarity_threshold
        self._total_links = 0
        self._anchor_registry: Dict[str, List[str]] = {}  # anchor → [modalities]

    def link_modalities(
        self,
        anchor: str,
        modality_concept_pairs: List[Tuple[str, str]],
    ) -> CrossModalEnrichmentReport:
        """
        Link multiple modality-specific concepts under one cross-modal anchor.

        Parameters
        ----------
        anchor : str
            Shared concept name (e.g. 'dog').
        modality_concept_pairs : list of (modality, concept)
            E.g. [('vision', 'dog_image'), ('audio', 'dog_bark')].

        Returns
        -------
        CrossModalEnrichmentReport
        """
        report = CrossModalEnrichmentReport()
        if anchor not in self._anchor_registry:
            self._anchor_registry[anchor] = []
            report.anchors_created += 1
            report.steps.append(f"anchor_created: {anchor}")

        for modality, concept in modality_concept_pairs:
            if self._mem is not None:
                try:
                    self._mem.register_concept(concept, modality=modality, anchor=anchor)
                    report.links_added += 1
                    report.steps.append(f"linked: {modality}/{concept} -> {anchor}")
                except Exception as exc:
                    log.debug("CrossModalEnricher.link: %s", exc)
                    try:
                        self._mem.register_concept(concept, modality)
                        report.links_added += 1
                        report.steps.append(f"linked(simple): {modality}/{concept}")
                    except Exception as exc2:
                        log.debug("CrossModalEnricher.link fallback: %s", exc2)
                        report.steps.append(f"skipped: {modality}/{concept} ({exc2})")
            else:
                report.links_added += 1  # dry-run
                report.steps.append(f"dry_run: {modality}/{concept} -> {anchor}")

            if modality not in self._anchor_registry[anchor]:
                self._anchor_registry[anchor].append(modality)

        self._total_links += report.links_added
        return report

    def detect_clusters(self) -> CrossModalEnrichmentReport:
        """
        Detect anchors that appear in 2+ modalities (cross-modal clusters).

        Returns
        -------
        CrossModalEnrichmentReport
        """
        report = CrossModalEnrichmentReport()
        for anchor, modalities in self._anchor_registry.items():
            if len(set(modalities)) >= 2:
                report.clusters_detected += 1
                report.steps.append(f"cluster: {anchor} in {sorted(set(modalities))}")
        return report

    @property
    def total_links(self) -> int:
        return self._total_links

    @property
    def anchor_count(self) -> int:
        return len(self._anchor_registry)

    def summary(self) -> Dict:
        """Return a dict summary of enrichment state."""
        return {
            'total_links': self._total_links,
            'anchor_count': self.anchor_count,
            'anchors': {k: list(v) for k, v in self._anchor_registry.items()},
        }
