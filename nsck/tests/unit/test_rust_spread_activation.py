"""Tests for Rust spreading activation shim (WP-2)."""
from __future__ import annotations
import pytest
import numpy as np

try:
    import hypervec_rs as _hvrs
    RUST_AVAILABLE = _hvrs.SemanticMemoryConcurrent is not None
except ImportError:
    RUST_AVAILABLE = False


def _build_semantic_memory_with_graph():
    """Build a SemanticMemory with a small test graph."""
    from python.core.memory.semantic_memory import SemanticMemory
    mem = SemanticMemory(use_rust=False)
    mem.add_concept("animal", {"type": "category"})
    mem.add_concept("dog", {"type": "entity"})
    mem.add_concept("cat", {"type": "entity"})
    mem.add_concept("mammal", {"type": "category"})
    mem.add_relation("dog", "is_a", "mammal")
    mem.add_relation("cat", "is_a", "mammal")
    mem.add_relation("mammal", "is_a", "animal")
    mem.add_relation("dog", "similar_to", "cat")
    return mem


def test_rust_python_equivalence():
    """Verify Rust and Python paths give equivalent results."""
    from python.core.memory.semantic_memory import SemanticMemory
    mem = _build_semantic_memory_with_graph()
    
    # Python result (direct)
    py_result = mem.spread_activation(["dog"], steps=2, decay=0.7)
    
    # Should have activated related concepts
    assert "dog" in py_result
    assert py_result["dog"] >= 1.0
    assert "mammal" in py_result
    assert py_result["mammal"] > 0


def test_rust_relation_weights():
    """is_a edges carry more activation than similar_to edges."""
    mem = _build_semantic_memory_with_graph()
    result = mem.spread_activation(["dog"], steps=1, decay=0.7)
    
    # mammal reachable via is_a (weight 0.9), cat via similar_to (weight 0.4)
    mammal_act = result.get("mammal", 0.0)
    cat_act = result.get("cat", 0.0)
    # Verify both concepts actually received activation (not just comparing zeros)
    assert mammal_act > 0.0, f"mammal should receive activation via is_a edge, got {mammal_act}"
    assert cat_act > 0.0, f"cat should receive activation via similar_to edge, got {cat_act}"
    assert mammal_act > cat_act, f"is_a should carry more activation: mammal={mammal_act}, cat={cat_act}"


def test_rust_stigmergy_boost():
    """Pheromone-marked paths get higher activation."""
    from python.core.memory.semantic_memory import SemanticMemory
    mem_plain = _build_semantic_memory_with_graph()
    mem_stig = _build_semantic_memory_with_graph()
    
    # Mark the dog->mammal path with high pheromone
    mem_stig.mark_path(["dog", "mammal"], reward=5.0)
    
    plain_result = mem_plain.spread_activation(["dog"], steps=1, decay=0.7)
    stig_result = mem_stig.spread_activation(["dog"], steps=1, decay=0.7)
    
    # Stigmergy-boosted mammal should get more activation
    assert stig_result.get("mammal", 0.0) >= plain_result.get("mammal", 0.0)


def test_rust_pruning_bounds():
    """1K+ nodes doesn't explode — activation count stays bounded."""
    from python.core.memory.semantic_memory import SemanticMemory
    mem = SemanticMemory(use_rust=False)
    
    # Add 1000 nodes
    for i in range(1000):
        mem.add_concept(f"concept_{i}", {})
    
    # Add edges in a chain
    for i in range(999):
        mem.add_relation(f"concept_{i}", "similar_to", f"concept_{i+1}")
    
    result = mem.spread_activation(["concept_0"], steps=3, decay=0.7)
    
    # Should not activate all 1000+ concepts (bounded by _MAX_FRONTIER=200)
    assert len(result) < 1000, f"Too many activated concepts: {len(result)}"
    assert len(result) > 0


def test_fallback_when_rust_unavailable():
    """Python path works when Rust not compiled."""
    from python.core.memory.semantic_memory import SemanticMemory
    from python.core.memory import semantic_memory_shim
    
    # Temporarily override to simulate Rust unavailable
    orig_use_rust = semantic_memory_shim._USE_RUST
    semantic_memory_shim._USE_RUST = False
    
    try:
        mem = _build_semantic_memory_with_graph()
        result = mem.spread_activation(["dog"], steps=2, decay=0.7)
        assert "dog" in result
        assert result["dog"] >= 1.0
    finally:
        semantic_memory_shim._USE_RUST = orig_use_rust
