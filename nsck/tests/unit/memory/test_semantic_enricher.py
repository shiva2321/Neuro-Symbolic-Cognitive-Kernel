"""Tests for SemanticEnricher (V17)."""
import pytest
from python.core.memory.semantic_enricher import SemanticEnricher, EnrichmentReport, INVERSE_RELATION_MAP


def test_enrich_without_memory():
    enricher = SemanticEnricher(semantic_memory=None)
    report = enricher.enrich_concept("dog", "is_a", "animal")
    assert isinstance(report, EnrichmentReport)
    assert "no_semantic_memory" in report.steps


def test_enrich_bulk_empty():
    enricher = SemanticEnricher(semantic_memory=None)
    report = enricher.enrich_bulk([])
    assert report.concepts_enriched == 0


def test_enrich_bulk_multiple():
    enricher = SemanticEnricher(semantic_memory=None)
    triples = [("a", "is_a", "b"), ("c", "has_part", "d"), ("x", "causes", "y")]
    report = enricher.enrich_bulk(triples)
    assert report.concepts_enriched == len(triples)


def test_total_enrichments_increments():
    enricher = SemanticEnricher(semantic_memory=None)
    assert enricher.total_enrichments == 0
    enricher.enrich_concept("cat", "is_a", "mammal")
    assert enricher.total_enrichments == 1


def test_inverse_relation_map_coverage():
    required = {'is_a', 'has_part', 'causes', 'used_for', 'at_location'}
    assert required <= set(INVERSE_RELATION_MAP.keys())


def test_coquery_stats():
    enricher = SemanticEnricher(semantic_memory=None)
    enricher.enrich_concept("dog", "is_a", "animal")
    enricher.enrich_concept("dog", "is_a", "animal")
    stats = enricher.coquery_stats()
    assert stats[("dog", "animal")] == 2


def test_add_inverses_false():
    from unittest.mock import MagicMock
    mem = MagicMock()
    enricher = SemanticEnricher(semantic_memory=mem, add_inverses=False)
    report = enricher.enrich_concept("a", "is_a", "b")
    mem.add_relation.assert_not_called()
    assert report.inverse_relations_added == 0


def test_add_inverses_true_calls_memory():
    from unittest.mock import MagicMock
    mem = MagicMock()
    enricher = SemanticEnricher(semantic_memory=mem, add_inverses=True)
    report = enricher.enrich_concept("dog", "is_a", "animal")
    mem.add_relation.assert_called_once_with("animal", "sub_class_of", "dog")
    assert report.inverse_relations_added == 1
