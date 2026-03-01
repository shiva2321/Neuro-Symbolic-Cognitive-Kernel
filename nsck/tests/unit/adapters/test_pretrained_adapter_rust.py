"""
Tests verifying the PretrainedModelAdapter runs end-to-end through the Rust backend.

These tests explicitly check that every HV operation in the adapter pipeline
(projection, XOR binding, bundling, similarity) uses the Rust ``hypervec_rs``
backend when available, and that the full Substrate flow produces correct
results with Rust acceleration.

Skipped automatically when the Rust extension is not compiled.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../"))

import numpy as np
import pytest

import python.core.vsa.hypervec_shim as shim

_RUST_AVAILABLE = shim.get_backend_info()["vsa_backend"] == "Rust"
pytestmark = pytest.mark.skipif(not _RUST_AVAILABLE, reason="Rust backend not available")


def _random_embedding(dim: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(dim).astype(np.float32)


class TestPretrainedAdapterRustBackend:
    """Verify the PretrainedModelAdapter pipeline uses Rust HVs end-to-end."""

    def test_backend_is_rust(self):
        info = shim.get_backend_info()
        assert info["vsa_backend"] == "Rust"
        assert info["hypervec_rs_available"] is True

    def test_hv_type_is_rust(self):
        hv = shim.HyperVector(42)
        assert type(hv).__module__ == "hypervec_rs"

    def test_adapter_produces_rust_hvs(self):
        """HVs in the PerceptPacket must be Rust HyperVector instances."""
        from python.core.adapters.pretrained_model_adapter import PretrainedModelAdapter

        adapter = PretrainedModelAdapter()
        adapter.register_source("m1", embedding_dim=64)
        pkt = adapter.encode({"m1": _random_embedding(64)}, "test")

        assert type(pkt.situation_hv).__module__ == "hypervec_rs"
        for name, hv in pkt.entity_hvs.items():
            assert type(hv).__module__ == "hypervec_rs", f"entity {name} not Rust"

    def test_multi_source_all_rust(self):
        """Multiple sources should all produce Rust HVs and fuse via Rust bundle."""
        from python.core.adapters.pretrained_model_adapter import PretrainedModelAdapter

        adapter = PretrainedModelAdapter()
        adapter.register_source("med", embedding_dim=128, weight=1.0, domain="medical")
        adapter.register_source("sat", embedding_dim=256, weight=0.8, domain="satellite")
        adapter.register_source("gen", embedding_dim=64, weight=0.5, domain="general")

        pkt = adapter.encode(
            {
                "med": _random_embedding(128, seed=1),
                "sat": _random_embedding(256, seed=2),
                "gen": _random_embedding(64, seed=3),
            },
            "multi_test",
        )
        assert type(pkt.situation_hv).__module__ == "hypervec_rs"
        assert pkt.adapter_trace["n_sources"] == 3
        assert pkt.modality == "pretrained_fusion"

    def test_domain_binding_uses_rust_xor(self):
        """Domain-role binding (XOR) should use Rust's xor() method."""
        from python.core.adapters.pretrained_model_adapter import PretrainedModelAdapter

        adapter = PretrainedModelAdapter(bind_domain_role=True)
        adapter.register_source("a", embedding_dim=64, domain="alpha")
        adapter.register_source("b", embedding_dim=64, domain="beta")

        vec = _random_embedding(64, seed=42)
        pkt = adapter.encode({"a": vec, "b": vec}, "test")

        hv_a = pkt.entity_hvs["a"]
        hv_b = pkt.entity_hvs["b"]

        # Both must be Rust HVs
        assert type(hv_a).__module__ == "hypervec_rs"
        assert type(hv_b).__module__ == "hypervec_rs"

        # XOR binding with different domain roles → not identical
        sim = hv_a.similarity(hv_b)
        assert sim < 0.95, f"Domain-bound HVs should differ, got sim={sim:.3f}"

    def test_similarity_preservation_with_rust(self):
        """Verify that similar embeddings → similar HVs when using Rust."""
        from python.core.adapters.pretrained_model_adapter import PretrainedModelAdapter

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
            f"Rust: close ({sim_close:.3f}) should be > far ({sim_far:.3f})"
        )

    def test_weighted_bundling_with_rust(self):
        """Higher-weight source should dominate the fused HV."""
        dim = 128
        vec_a = _random_embedding(dim, seed=10)
        vec_b = _random_embedding(dim, seed=20)

        from python.core.adapters.pretrained_model_adapter import PretrainedModelAdapter

        adapter = PretrainedModelAdapter(bind_domain_role=False)
        adapter.register_source("a", embedding_dim=dim, weight=5.0)
        adapter.register_source("b", embedding_dim=dim, weight=1.0)
        pkt_fused = adapter.encode({"a": vec_a, "b": vec_b}, "test")

        adapter_solo = PretrainedModelAdapter(bind_domain_role=False)
        adapter_solo.register_source("a", embedding_dim=dim, weight=1.0)
        pkt_solo = adapter_solo.encode({"a": vec_a}, "test")

        sim = pkt_fused.situation_hv.similarity(pkt_solo.situation_hv)
        assert sim > 0.3, f"Rust weighted bundle: sim to dominant source = {sim:.3f}"


class TestSubstrateRustEndToEnd:
    """Verify the full Substrate → PretrainedModelAdapter → Decide flow with Rust."""

    def test_absorb_and_process_with_rust(self):
        from python.core.substrate import NSCKSubstrate
        from python.core.integration.config import NSCKConfig

        cfg = NSCKConfig()
        substrate = NSCKSubstrate(config=cfg)
        substrate.register_task("rust_e2e")

        substrate.absorb_pretrained("model_a", embedding_dim=64, weight=1.0, domain="alpha")
        substrate.absorb_pretrained("model_b", embedding_dim=128, weight=0.8, domain="beta")

        result = substrate.process_pretrained(
            {
                "model_a": _random_embedding(64, seed=7),
                "model_b": _random_embedding(128, seed=8),
            },
            task_tag="rust_e2e",
        )
        assert result.chosen_action is not None
        assert "pretrained_fusion" in result.modalities_processed
        assert result.confidence > 0.0

    def test_rust_snn_backend_available(self):
        info = shim.get_backend_info()
        assert info["snn_backend"] == "Rust"
        assert info["snn_rs_available"] is True

    def test_rust_concurrent_classes_available(self):
        """All Rust concurrent classes should be non-None when Rust is active."""
        assert shim.SemanticMemoryConcurrent is not None
        assert shim.EpisodicMemoryConcurrent is not None
        assert shim.CognitiveWorkerPool is not None
        assert shim.PersistentStorage is not None

    def test_rust_parallel_functions_available(self):
        """Rust parallel free functions should be available."""
        assert shim.parallel_bundle is not None
        assert shim.parallel_similarity_search is not None
        assert shim.batch_parallel_similarity_search is not None
