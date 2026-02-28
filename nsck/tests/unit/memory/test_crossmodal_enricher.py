"""Tests for CrossModalEnricher (V17)."""
import pytest
from python.core.memory.crossmodal_enricher import CrossModalEnricher, CrossModalEnrichmentReport


def test_link_modalities_dry_run():
    enricher = CrossModalEnricher(crossmodal_memory=None)
    report = enricher.link_modalities("dog", [("vision", "dog_img"), ("audio", "dog_bark")])
    assert isinstance(report, CrossModalEnrichmentReport)
    assert report.anchors_created == 1
    assert report.links_added == 2


def test_total_links():
    enricher = CrossModalEnricher(crossmodal_memory=None)
    enricher.link_modalities("cat", [("vision", "cat_img"), ("audio", "cat_meow")])
    assert enricher.total_links == 2


def test_anchor_count():
    enricher = CrossModalEnricher(crossmodal_memory=None)
    enricher.link_modalities("dog", [("vision", "v_dog")])
    enricher.link_modalities("cat", [("audio", "a_cat")])
    assert enricher.anchor_count == 2


def test_detect_clusters_two_modalities():
    enricher = CrossModalEnricher(crossmodal_memory=None)
    enricher.link_modalities("dog", [("vision", "v_dog"), ("audio", "a_dog")])
    report = enricher.detect_clusters()
    assert report.clusters_detected == 1
    assert any("dog" in s for s in report.steps)


def test_detect_clusters_single_modality_no_cluster():
    enricher = CrossModalEnricher(crossmodal_memory=None)
    enricher.link_modalities("cat", [("vision", "v_cat")])
    report = enricher.detect_clusters()
    assert report.clusters_detected == 0


def test_summary_structure():
    enricher = CrossModalEnricher(crossmodal_memory=None)
    enricher.link_modalities("fish", [("vision", "v_fish"), ("text", "t_fish")])
    s = enricher.summary()
    assert 'total_links' in s
    assert 'anchor_count' in s
    assert 'anchors' in s
    assert s['anchor_count'] == 1


def test_same_anchor_twice():
    enricher = CrossModalEnricher(crossmodal_memory=None)
    enricher.link_modalities("bird", [("vision", "v_bird")])
    report2 = enricher.link_modalities("bird", [("audio", "a_bird")])
    assert report2.anchors_created == 0  # anchor already exists
    assert report2.links_added == 1


def test_empty_pairs():
    enricher = CrossModalEnricher(crossmodal_memory=None)
    report = enricher.link_modalities("empty", [])
    assert report.links_added == 0
