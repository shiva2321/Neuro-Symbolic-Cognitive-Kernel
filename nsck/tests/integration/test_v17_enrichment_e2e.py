"""V17 End-to-End Integration Test — Enrichment + Glass-Box Tracing."""
import pytest

from python.core.reasoning.causal_enricher import CausalEnricher
from python.core.perception.perceptual_enricher import PerceptualEnricher
from python.core.memory.semantic_enricher import SemanticEnricher
from python.core.cognitive.glass_box_tracer import GlassBoxTracer
from python.core.memory.crossmodal_enricher import CrossModalEnricher
from python.core.integration.config import NSCKConfig


class TestV17ConfigPreset:
    def test_v17_preset_flags(self):
        cfg = NSCKConfig.v17()
        assert cfg.enable_causal_enrichment
        assert cfg.enable_perceptual_enrichment
        assert cfg.enable_semantic_enrichment
        assert cfg.enable_glass_box_tracer
        assert cfg.enable_crossmodal_enrichment

    def test_v17_preset_defaults(self):
        cfg = NSCKConfig.v17()
        assert cfg.causal_enrichment_n_context == 3
        assert cfg.perceptual_enricher_window == 8
        assert cfg.semantic_enrichment_add_inverses is True


class TestV17FullPipeline:
    """Simulated pipeline: perception → causal enrichment → glass-box trace."""

    def test_enrichment_pipeline(self):
        # 1. Glass-box tracer
        tracer = GlassBoxTracer()
        tracer.begin_decision("e2e-v17")

        # 2. Perceptual enrichment
        enricher_p = PerceptualEnricher(window_size=4)
        from unittest.mock import MagicMock
        pkt = MagicMock()
        pkt.modality = "text"
        pkt.situation_hv = None
        ep = enricher_p.enrich(pkt)
        with tracer.span("perception"):
            tracer.record(
                "PerceptualEnricher",
                f"enriched {ep.original_modality}",
                confidence=ep.confidence,
            )

        # 3. Causal enrichment
        causal_e = CausalEnricher()
        chain = ["heat", "expansion", "pressure_increase"]
        traces = causal_e.enrich_chain(chain)
        with tracer.span("causal"):
            for t in traces:
                tracer.record(
                    "CausalEnricher",
                    f"{t.cause} -> {t.effect}",
                    confidence=t.strength,
                )

        # 4. Semantic enrichment
        sem_e = SemanticEnricher(semantic_memory=None)
        report = sem_e.enrich_bulk([("dog", "is_a", "animal"), ("cat", "is_a", "animal")])
        with tracer.span("semantic"):
            tracer.record(
                "SemanticEnricher",
                f"enriched {report.concepts_enriched} concepts",
            )

        # 5. Cross-modal enrichment
        cm_e = CrossModalEnricher(crossmodal_memory=None)
        cm_report = cm_e.link_modalities("dog", [("vision", "v_dog"), ("audio", "a_dog")])
        with tracer.span("crossmodal"):
            tracer.record(
                "CrossModalEnricher",
                f"linked {cm_report.links_added} modalities",
            )

        # 6. End decision + verify trace
        final_trace = tracer.end_decision()
        assert final_trace is not None
        spans_used = {e.span for e in final_trace.entries}
        assert {"perception", "causal", "semantic", "crossmodal"} <= spans_used

        text = GlassBoxTracer.format_trace(final_trace)
        assert "perception" in text
        assert "causal" in text

    def test_causal_chain_multi_hop(self):
        enricher = CausalEnricher()
        chains = enricher.enrich_chain(
            ["virus", "infection", "fever", "dehydration"], base_strength=0.9
        )
        assert len(chains) == 3
        for c in chains:
            assert 0 < c.strength <= 1.0

    def test_semantic_bulk_inverses_added(self):
        sem = SemanticEnricher(semantic_memory=None, add_inverses=True)
        triples = [
            ("dog", "is_a", "mammal"),
            ("mammal", "has_part", "heart"),
            ("rain", "causes", "flood"),
        ]
        report = sem.enrich_bulk(triples)
        assert report.concepts_enriched == 3

    def test_crossmodal_cluster_detection(self):
        cm = CrossModalEnricher(crossmodal_memory=None)
        cm.link_modalities(
            "fire", [("vision", "v_fire"), ("audio", "a_fire"), ("text", "t_fire")]
        )
        cr = cm.detect_clusters()
        assert cr.clusters_detected == 1

    def test_glass_box_trace_format_all_spans(self):
        tracer = GlassBoxTracer()
        tracer.begin_decision("fmt-test")
        for span_name in ["perception", "causal", "semantic", "crossmodal", "reasoning"]:
            with tracer.span(span_name):
                tracer.record("TestModule", f"step in {span_name}")
        trace = tracer.end_decision()
        text = GlassBoxTracer.format_trace(trace)
        for span_name in ["perception", "causal", "semantic", "crossmodal", "reasoning"]:
            assert span_name in text
