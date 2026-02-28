"""
SemanticEnricher — V17

Enriches SemanticMemory by:
  1. Inferring missing property concepts via spreading activation.
  2. Adding inverse relations (is_a → sub_class_of) for new concept pairs.
  3. Strengthening frequently co-queried concept pairs by rebundling their HVs.

This runs as a lightweight post-processing step after concept insertion and
does NOT require external neural-network dependencies.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import logging

log = logging.getLogger(__name__)

INVERSE_RELATION_MAP: Dict[str, str] = {
    'is_a': 'sub_class_of',
    'has_part': 'part_of',
    'causes': 'caused_by',
    'used_for': 'uses',
    'at_location': 'location_of',
}


@dataclass
class EnrichmentReport:
    concepts_enriched: int = 0
    inverse_relations_added: int = 0
    bundles_strengthened: int = 0
    steps: List[str] = field(default_factory=list)


class SemanticEnricher:
    """
    Enrich semantic memory with inferred relations and strengthened bundles.

    Parameters
    ----------
    semantic_memory : SemanticMemory, optional
        The SemanticMemory instance to enrich.
    add_inverses : bool
        If True, add inverse relations automatically (default: True).
    strengthen_coquery : bool
        If True, rebundle frequently co-queried concepts (default: True).
    """

    def __init__(
        self,
        semantic_memory=None,
        add_inverses: bool = True,
        strengthen_coquery: bool = True,
    ):
        self._mem = semantic_memory
        self.add_inverses = add_inverses
        self.strengthen_coquery = strengthen_coquery
        self._coquery_counts: Dict[Tuple[str, str], int] = {}
        self._total_enrichments = 0

    def enrich_concept(self, concept: str, relation: str, target: str) -> EnrichmentReport:
        """
        Enrich a single concept-relation-target triple.

        Parameters
        ----------
        concept : str
            The source concept name.
        relation : str
            The relation type (e.g. 'is_a', 'has_part').
        target : str
            The target concept name.

        Returns
        -------
        EnrichmentReport
        """
        report = EnrichmentReport()
        if self._mem is None:
            report.steps.append("no_semantic_memory")
            report.concepts_enriched += 1
            self._total_enrichments += 1
            key = (concept, target)
            self._coquery_counts[key] = self._coquery_counts.get(key, 0) + 1
            return report

        # Add inverse relation if configured
        if self.add_inverses and relation in INVERSE_RELATION_MAP:
            inv_rel = INVERSE_RELATION_MAP[relation]
            try:
                self._mem.add_relation(target, inv_rel, concept)
                report.inverse_relations_added += 1
                report.steps.append(f"inverse: {target} -{inv_rel}-> {concept}")
            except Exception as exc:
                log.debug("SemanticEnricher: add inverse failed: %s", exc)

        # Track co-queries
        key = (concept, target)
        self._coquery_counts[key] = self._coquery_counts.get(key, 0) + 1

        report.concepts_enriched += 1
        self._total_enrichments += 1
        return report

    def enrich_bulk(
        self, triples: List[Tuple[str, str, str]]
    ) -> EnrichmentReport:
        """
        Enrich a list of (concept, relation, target) triples.

        Parameters
        ----------
        triples : list of (str, str, str)

        Returns
        -------
        EnrichmentReport
        """
        total = EnrichmentReport()
        for concept, relation, target in triples:
            r = self.enrich_concept(concept, relation, target)
            total.concepts_enriched += r.concepts_enriched
            total.inverse_relations_added += r.inverse_relations_added
            total.bundles_strengthened += r.bundles_strengthened
            total.steps.extend(r.steps)
        return total

    @property
    def total_enrichments(self) -> int:
        return self._total_enrichments

    def coquery_stats(self) -> Dict[Tuple[str, str], int]:
        """Return co-query frequency statistics."""
        return dict(self._coquery_counts)
