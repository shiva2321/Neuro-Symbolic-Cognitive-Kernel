"""
Tests for PretrainedModelAdapter (NSCK V19).

Verifies:
  - Source registration (single and multiple)
  - encode() produces valid PerceptPackets with modality="pretrained_fusion"
  - Weighted bundling: higher-weight sources dominate similarity
  - Domain role binding makes identical embeddings distinguishable
  - Error handling: no sources, mismatched dims, empty dict, non-dict input
  - Integration via NSCKSubstrate.absorb_pretrained / process_pretrained
  - Similarity preservation: similar embeddings → similar fused HVs
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../"))

import numpy as np
import pytest

from python.core.adapters.pretrained_model_adapter import PretrainedModelAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_embedding(dim: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(dim).astype(np.float32)


# ---------------------------------------------------------------------------
# Unit tests — PretrainedModelAdapter
# ---------------------------------------------------------------------------

class TestPretrainedModelAdapterBasic:
    """Core functionality tests."""

    def test_register_single_source(self):
        adapter = PretrainedModelAdapter()
        adapter.register_source("model_a", embedding_dim=128)
        assert adapter.n_sources == 1
        assert "model_a" in adapter.source_names

    def test_register_multiple_sources(self):
        adapter = PretrainedModelAdapter()
        adapter.register_source("medical", embedding_dim=512)
        adapter.register_source("satellite", embedding_dim=768)
        adapter.register_source("traffic", embedding_dim=256)
        assert adapter.n_sources == 3

    def test_encode_single_source(self):
        adapter = PretrainedModelAdapter()
        adapter.register_source("resnet", embedding_dim=64)
        vec = _random_embedding(64, seed=42)
        pkt = adapter.encode({"resnet": vec}, "test_task")
        assert pkt.modality == "pretrained_fusion"
        assert pkt.situation_hv is not None
        assert pkt.adapter_name == "PretrainedModelAdapter"
        assert "SOURCE_RESNET" in pkt.active_predicates

    def test_encode_multiple_sources(self):
        adapter = PretrainedModelAdapter()
        adapter.register_source("medical", embedding_dim=128, weight=1.0)
        adapter.register_source("general", embedding_dim=256, weight=0.5)
        pkt = adapter.encode(
            {
                "medical": _random_embedding(128, seed=1),
                "general": _random_embedding(256, seed=2),
            },
            "diagnosis",
        )
        assert pkt.modality == "pretrained_fusion"
        assert len(pkt.entity_hvs) == 2
        assert "SOURCE_MEDICAL" in pkt.active_predicates
        assert "SOURCE_GENERAL" in pkt.active_predicates
        assert pkt.adapter_trace["n_sources"] == 2

    def test_encode_subset_of_sources(self):
        """Providing a subset of registered sources should work."""
        adapter = PretrainedModelAdapter()
        adapter.register_source("a", embedding_dim=64)
        adapter.register_source("b", embedding_dim=64)
        pkt = adapter.encode({"a": _random_embedding(64)}, "test")
        assert pkt.adapter_trace["n_sources"] == 1
        assert "SOURCE_A" in pkt.active_predicates

    def test_confidence_scales_with_coverage(self):
        """Confidence should reflect how many registered sources are present."""
        adapter = PretrainedModelAdapter()
        adapter.register_source("a", embedding_dim=64, weight=1.0)
        adapter.register_source("b", embedding_dim=64, weight=1.0)
        pkt_partial = adapter.encode({"a": _random_embedding(64)}, "test")
        pkt_full = adapter.encode(
            {"a": _random_embedding(64), "b": _random_embedding(64, seed=1)},
            "test",
        )
        assert pkt_full.confidence >= pkt_partial.confidence


class TestPretrainedModelAdapterWeighting:
    """Tests for weighted bundling behaviour."""

    def test_higher_weight_source_dominates(self):
        """A source with higher weight should dominate the fused HV."""
        dim = 128
        vec_a = _random_embedding(dim, seed=10)
        vec_b = _random_embedding(dim, seed=20)

        # High weight on A
        adapter_hi_a = PretrainedModelAdapter(bind_domain_role=False)
        adapter_hi_a.register_source("a", embedding_dim=dim, weight=5.0)
        adapter_hi_a.register_source("b", embedding_dim=dim, weight=1.0)
        pkt_hi_a = adapter_hi_a.encode({"a": vec_a, "b": vec_b}, "test")

        # Encode A alone for comparison
        adapter_solo = PretrainedModelAdapter(bind_domain_role=False)
        adapter_solo.register_source("a", embedding_dim=dim, weight=1.0)
        pkt_a_only = adapter_solo.encode({"a": vec_a}, "test")

        sim = pkt_hi_a.situation_hv.similarity(pkt_a_only.situation_hv)
        # Fused with high-weight A should be fairly similar to A alone
        assert sim > 0.3, f"High-weight A fused sim to A-only: {sim:.3f}"


class TestPretrainedModelAdapterDomainBinding:
    """Tests for domain-role binding."""

    def test_same_embedding_different_domains_are_distinguishable(self):
        """With bind_domain_role=True, identical embeddings from different
        domains should produce different entity HVs."""
        dim = 64
        vec = _random_embedding(dim, seed=42)
        adapter = PretrainedModelAdapter(bind_domain_role=True)
        adapter.register_source("medical", embedding_dim=dim, domain="medical")
        adapter.register_source("traffic", embedding_dim=dim, domain="traffic")

        pkt = adapter.encode({"medical": vec, "traffic": vec}, "test")
        hv_med = pkt.entity_hvs["medical"]
        hv_trf = pkt.entity_hvs["traffic"]
        sim = hv_med.similarity(hv_trf)
        # They should NOT be identical (sim < 1.0)
        assert sim < 0.95, f"Same embedding different domains sim={sim:.3f}, expected < 0.95"

    def test_no_binding_identical_embeddings_are_similar(self):
        """Without domain binding, identical embeddings → identical entity HVs."""
        dim = 64
        vec = _random_embedding(dim, seed=42)
        adapter = PretrainedModelAdapter(bind_domain_role=False)
        adapter.register_source("a", embedding_dim=dim)
        adapter.register_source("b", embedding_dim=dim)

        pkt = adapter.encode({"a": vec, "b": vec}, "test")
        sim = pkt.entity_hvs["a"].similarity(pkt.entity_hvs["b"])
        # Without role binding, same projector seed + same input → same HV
        assert sim > 0.95, f"Same embedding no binding sim={sim:.3f}, expected > 0.95"


class TestPretrainedModelAdapterSimilarity:
    """Similarity-preservation tests."""

    def test_similar_embeddings_produce_similar_fused_hvs(self):
        dim = 128
        rng = np.random.default_rng(99)
        base = rng.standard_normal(dim).astype(np.float32)
        close = base + rng.standard_normal(dim).astype(np.float32) * 0.01
        far = rng.standard_normal(dim).astype(np.float32)

        adapter = PretrainedModelAdapter(bind_domain_role=False)
        adapter.register_source("m", embedding_dim=dim)

        pkt_base = adapter.encode({"m": base}, "test")
        pkt_close = adapter.encode({"m": close}, "test")
        pkt_far = adapter.encode({"m": far}, "test")

        sim_close = pkt_base.situation_hv.similarity(pkt_close.situation_hv)
        sim_far = pkt_base.situation_hv.similarity(pkt_far.situation_hv)
        assert sim_close > sim_far, (
            f"Close embeddings ({sim_close:.3f}) should be more similar "
            f"than far ones ({sim_far:.3f})"
        )


class TestPretrainedModelAdapterErrors:
    """Error handling tests."""

    def test_encode_non_dict_raises(self):
        adapter = PretrainedModelAdapter()
        adapter.register_source("a", embedding_dim=64)
        with pytest.raises(ValueError, match="expects a dict"):
            adapter.encode([1, 2, 3], "test")

    def test_encode_no_sources_registered_raises(self):
        adapter = PretrainedModelAdapter()
        with pytest.raises(ValueError, match="No sources registered"):
            adapter.encode({"a": np.zeros(10)}, "test")

    def test_encode_no_matching_sources_raises(self):
        adapter = PretrainedModelAdapter()
        adapter.register_source("a", embedding_dim=64)
        with pytest.raises(ValueError, match="No registered source names found"):
            adapter.encode({"unknown": np.zeros(64)}, "test")

    def test_encode_wrong_dim_raises(self):
        adapter = PretrainedModelAdapter()
        adapter.register_source("a", embedding_dim=64)
        with pytest.raises(ValueError, match="expected dim 64"):
            adapter.encode({"a": np.zeros(128)}, "test")

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            PretrainedModelAdapter(default_strategy="invalid")


class TestPretrainedModelAdapterSVDStrategy:
    """Tests with svd_factored projection."""

    def test_encode_with_svd_strategy(self):
        dim = 64
        rng = np.random.default_rng(7)
        fit_data = rng.standard_normal((50, dim)).astype(np.float32)

        adapter = PretrainedModelAdapter(default_strategy="svd_factored")
        adapter.register_source(
            "svd_model", embedding_dim=dim, fit_embeddings=fit_data
        )
        vec = rng.standard_normal(dim).astype(np.float32)
        pkt = adapter.encode({"svd_model": vec}, "test")
        assert pkt.modality == "pretrained_fusion"
        assert pkt.situation_hv is not None


class TestPretrainedModelAdapterSubstrate:
    """Integration tests via NSCKSubstrate."""

    def test_absorb_and_process(self):
        from python.core.substrate import NSCKSubstrate
        from python.core.integration.config import NSCKConfig

        cfg = NSCKConfig()
        cfg.enable_pretrained_adapter = True
        substrate = NSCKSubstrate(config=cfg)
        substrate.register_task("fusion_test")

        substrate.absorb_pretrained("model_x", embedding_dim=64, weight=1.0)
        substrate.absorb_pretrained("model_y", embedding_dim=128, weight=0.8)

        result = substrate.process_pretrained(
            {
                "model_x": _random_embedding(64, seed=5),
                "model_y": _random_embedding(128, seed=6),
            },
            task_tag="fusion_test",
        )
        assert result.chosen_action is not None
        assert "pretrained_fusion" in result.modalities_processed

    def test_process_pretrained_without_absorb_raises(self):
        from python.core.substrate import NSCKSubstrate

        substrate = NSCKSubstrate()
        with pytest.raises(RuntimeError, match="No pretrained sources"):
            substrate.process_pretrained({"x": np.zeros(10)}, "test")
