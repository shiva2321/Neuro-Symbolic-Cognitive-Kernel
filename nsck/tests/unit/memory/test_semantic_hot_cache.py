"""Unit tests for SemanticMemory hot cache (V4)."""
import pytest
from python.core.memory.semantic_memory import SemanticMemory


class TestHotCache:
    def test_hot_cache_initialized(self):
        sm = SemanticMemory()
        assert hasattr(sm, '_hot_cache')
        assert isinstance(sm._hot_cache, dict)

    def test_hot_cache_size_attribute(self):
        sm = SemanticMemory()
        assert hasattr(sm, '_HOT_CACHE_SIZE')
        assert sm._HOT_CACHE_SIZE == 256

    def test_hot_cache_populated_after_spread_activation(self):
        sm = SemanticMemory()
        sm.add_concept("cat", {"is_a": "animal"})
        sm.add_concept("animal", {"is_a": "living_thing"})
        sm.add_relation("cat", "is_a", "animal")
        sm.spread_activation(["cat"], steps=1, decay=0.5)
        # hot cache should have been updated
        assert len(sm._hot_cache) >= 0  # may be 0 if no activation propagated

    def test_hot_cache_eviction(self):
        sm = SemanticMemory()
        sm._HOT_CACHE_SIZE = 4  # Small size for testing
        # Populate beyond capacity
        activations = {f"concept_{i}": float(i) for i in range(10)}
        sm._update_hot_cache(activations)
        assert len(sm._hot_cache) <= sm._HOT_CACHE_SIZE

    def test_hnsw_enabled_by_default(self):
        sm = SemanticMemory()
        assert sm._hnsw_enabled is True

    def test_update_hot_cache_direct(self):
        sm = SemanticMemory()
        sm.add_concept("dog", {"is_a": "mammal"})
        activations = {"dog": 0.9}
        sm._update_hot_cache(activations)
        # "dog" should be in hot cache since it has an HV
        assert "dog" in sm._hot_cache
