"""Tests for CausalEnricher (V17)."""
import pytest
from python.core.reasoning.causal_enricher import CausalEnricher, CausalTrace


def test_enrich_basic():
    enricher = CausalEnricher()
    trace = enricher.enrich("fire", "smoke")
    assert isinstance(trace, CausalTrace)
    assert trace.cause == "fire"
    assert trace.effect == "smoke"
    assert 0.0 < trace.strength <= 1.0


def test_enrich_strength_custom():
    enricher = CausalEnricher()
    trace = enricher.enrich("rain", "flood", strength=0.8)
    assert abs(trace.strength - 0.8) < 1e-9


def test_enrich_chain():
    enricher = CausalEnricher()
    traces = enricher.enrich_chain(["a", "b", "c", "d"])
    assert len(traces) == 3
    assert traces[0].strength > traces[1].strength > traces[2].strength


def test_enrich_chain_single_hop():
    enricher = CausalEnricher()
    traces = enricher.enrich_chain(["x", "y"])
    assert len(traces) == 1


def test_enrich_chain_empty():
    enricher = CausalEnricher()
    traces = enricher.enrich_chain(["only"])
    assert traces == []


def test_enrichment_count():
    enricher = CausalEnricher()
    assert enricher.enrichment_count == 0
    enricher.enrich("a", "b")
    enricher.enrich("c", "d")
    assert enricher.enrichment_count == 2


def test_enrich_with_no_semantic_memory():
    enricher = CausalEnricher(semantic_memory=None)
    trace = enricher.enrich("heat", "expansion")
    assert "no_semantic_memory" in trace.enrichment_steps


def test_trace_steps_recorded():
    enricher = CausalEnricher()
    trace = enricher.enrich("virus", "disease")
    assert isinstance(trace.enrichment_steps, list)


def test_enrich_chain_strength_decay():
    enricher = CausalEnricher()
    chain = ["a", "b", "c", "d", "e"]
    traces = enricher.enrich_chain(chain, base_strength=1.0)
    for t in traces:
        assert 0 < t.strength <= 1.0
