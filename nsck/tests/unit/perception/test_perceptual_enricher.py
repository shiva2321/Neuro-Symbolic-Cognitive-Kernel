"""Tests for PerceptualEnricher (V17)."""
import pytest
from unittest.mock import MagicMock
from python.core.perception.perceptual_enricher import PerceptualEnricher, EnrichedPercept


def _make_packet(modality="text"):
    p = MagicMock()
    p.modality = modality
    p.situation_hv = None
    return p


def test_enrich_returns_enriched_percept():
    enricher = PerceptualEnricher()
    pkt = _make_packet()
    result = enricher.enrich(pkt)
    assert isinstance(result, EnrichedPercept)


def test_enrich_modality_preserved():
    enricher = PerceptualEnricher()
    pkt = _make_packet(modality="audio")
    result = enricher.enrich(pkt)
    assert result.original_modality == "audio"


def test_enrich_count_increments():
    enricher = PerceptualEnricher()
    assert enricher.enrich_count == 0
    for _ in range(5):
        enricher.enrich(_make_packet())
    assert enricher.enrich_count == 5


def test_temporal_context_after_history():
    enricher = PerceptualEnricher(window_size=4)
    # First packet: no temporal context yet
    r0 = enricher.enrich(_make_packet())
    assert not r0.temporal_ctx_available
    # After 2 packets, context is available
    enricher.enrich(_make_packet())
    r2 = enricher.enrich(_make_packet())
    assert r2.temporal_ctx_available


def test_reset_clears_history():
    enricher = PerceptualEnricher(window_size=4)
    for _ in range(3):
        enricher.enrich(_make_packet())
    enricher.reset()
    r = enricher.enrich(_make_packet())
    assert not r.temporal_ctx_available


def test_confidence_with_hv():
    import numpy as np
    enricher = PerceptualEnricher()
    hv_mock = MagicMock()
    hv_mock.vector = np.ones(16)  # all > 0.5 → confidence = 1.0
    pkt = MagicMock()
    pkt.modality = "text"
    pkt.situation_hv = hv_mock
    result = enricher.enrich(pkt)
    assert result.confidence == pytest.approx(1.0, abs=0.01)
    assert 'high_confidence' in result.tags


def test_tags_list_present():
    enricher = PerceptualEnricher()
    result = enricher.enrich(_make_packet())
    assert isinstance(result.tags, list)
