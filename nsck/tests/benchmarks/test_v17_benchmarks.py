"""
V17 Benchmarks — pytest-benchmark tests for enrichment modules.

Run::
    python -m pytest nsck/tests/benchmarks/test_v17_benchmarks.py -v --benchmark-disable
"""
import pytest
from unittest.mock import MagicMock


@pytest.fixture
def causal_enricher():
    from python.core.reasoning.causal_enricher import CausalEnricher
    return CausalEnricher()


@pytest.fixture
def perceptual_enricher():
    from python.core.perception.perceptual_enricher import PerceptualEnricher
    return PerceptualEnricher()


@pytest.fixture
def semantic_enricher():
    from python.core.memory.semantic_enricher import SemanticEnricher
    return SemanticEnricher(semantic_memory=None)


@pytest.fixture
def glass_box_tracer():
    from python.core.cognitive.glass_box_tracer import GlassBoxTracer
    return GlassBoxTracer()


@pytest.fixture
def crossmodal_enricher():
    from python.core.memory.crossmodal_enricher import CrossModalEnricher
    return CrossModalEnricher(crossmodal_memory=None)


@pytest.fixture
def mock_packet():
    p = MagicMock()
    p.modality = "text"
    p.situation_hv = None
    return p


def test_causal_enrich_throughput(benchmark, causal_enricher):
    result = benchmark(lambda: causal_enricher.enrich("fire", "smoke"))
    assert result is not None


def test_causal_chain_throughput(benchmark, causal_enricher):
    result = benchmark(lambda: causal_enricher.enrich_chain(["a", "b", "c", "d"]))
    assert len(result) == 3


def test_perceptual_enrich_throughput(benchmark, perceptual_enricher, mock_packet):
    result = benchmark(lambda: perceptual_enricher.enrich(mock_packet))
    assert result is not None


def test_semantic_enrich_throughput(benchmark, semantic_enricher):
    result = benchmark(lambda: semantic_enricher.enrich_concept("dog", "is_a", "animal"))
    assert result is not None


def test_glass_box_decision_throughput(benchmark, glass_box_tracer):
    def one_decision():
        glass_box_tracer.begin_decision()
        glass_box_tracer.record("M", "msg", confidence=0.9)
        return glass_box_tracer.end_decision()

    result = benchmark(one_decision)
    assert result is not None


def test_crossmodal_link_throughput(benchmark, crossmodal_enricher):
    result = benchmark(
        lambda: crossmodal_enricher.link_modalities("dog", [("vision", "v"), ("audio", "a")])
    )
    assert result is not None
