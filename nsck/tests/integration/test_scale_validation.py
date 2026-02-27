"""Scale validation tests (WP-6)."""
from __future__ import annotations
import time
import pytest


def _build_mem(n: int):
    from python.core.memory.semantic_memory import SemanticMemory
    import random
    random.seed(42)
    mem = SemanticMemory(use_rust=False)
    for i in range(n):
        mem.add_concept(f"c{i}", {})
    for i in range(min(n, 500)):
        j = random.randint(0, n - 1)
        if i != j:
            mem.add_relation(f"c{i}", "similar_to", f"c{j}")
    return mem


def test_100_concepts_under_100ms():
    """100 concepts: spread activation under 100ms."""
    mem = _build_mem(100)
    t = time.perf_counter()
    result = mem.spread_activation(["c0"], steps=3, decay=0.7)
    elapsed_ms = (time.perf_counter() - t) * 1000
    assert elapsed_ms < 100, f"Too slow: {elapsed_ms:.1f}ms"
    assert len(result) > 0


def test_1k_concepts_under_500ms():
    """1K concepts: spread activation under 500ms."""
    mem = _build_mem(1000)
    t = time.perf_counter()
    result = mem.spread_activation(["c0"], steps=3, decay=0.7)
    elapsed_ms = (time.perf_counter() - t) * 1000
    assert elapsed_ms < 500, f"Too slow: {elapsed_ms:.1f}ms"
    assert len(result) > 0


def test_for_scale_factory_valid():
    """for_scale() factory produces valid config."""
    from python.core.integration.config import NSCKConfig
    
    cfg = NSCKConfig.for_scale(100)
    assert cfg.memory_capacity >= 200
    assert isinstance(cfg.perception_mode, str)
    
    cfg_large = NSCKConfig.for_scale(10_000)
    assert cfg_large.enable_hnsw_index is True
    assert cfg_large.memory_capacity >= 20_000
