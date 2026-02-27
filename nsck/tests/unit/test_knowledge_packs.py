"""Tests for KnowledgePack (WP-5)."""
from __future__ import annotations
import os
import tempfile
import pytest

import python.core.vsa.hypervec_shim as hv_mod


def _make_pack():
    from python.core.integration.knowledge_pack import KnowledgePack
    pack = KnowledgePack(name="test_pack")
    pack.add_concept("cat", {"type": "animal"})
    pack.add_concept("dog", {"type": "animal"})
    pack.add_relation("cat", "is_a", "animal")
    pack.add_causal_link("hunger", "eating", strength=0.9)
    return pack


def test_save_load_roundtrip():
    """Save and load produces equivalent pack."""
    pack = _make_pack()
    
    with tempfile.NamedTemporaryFile(suffix=".gz", delete=False) as f:
        path = f.name
    
    try:
        pack.save(path)
        loaded = _make_pack().__class__.load(path)
        assert loaded.name == "test_pack"
        assert len(loaded._concepts) == 2
        assert len(loaded._relations) == 1
        assert len(loaded._causal_links) == 1
    finally:
        os.unlink(path)


def test_inject_concepts_queryable():
    """Injected concepts can be queried from semantic memory."""
    from python.core.memory.semantic_memory import SemanticMemory
    
    pack = _make_pack()
    
    # Create fake engine
    class FakeEngine:
        semantic_memory = SemanticMemory(use_rust=False)
    
    engine = FakeEngine()
    counts = pack.inject_into(engine)
    
    assert counts["concepts"] == 2
    assert "cat" in engine.semantic_memory.concept_graph.nodes()
    assert "dog" in engine.semantic_memory.concept_graph.nodes()


def test_config_field():
    """NSCKConfig has knowledge_packs field."""
    from python.core.integration.config import NSCKConfig
    cfg = NSCKConfig()
    assert hasattr(cfg, "knowledge_packs")
    assert isinstance(cfg.knowledge_packs, list)
    assert len(cfg.knowledge_packs) == 0


def test_substrate_loads_on_init():
    """Substrate loads knowledge packs specified in config."""
    from python.core.integration.config import NSCKConfig
    from python.core.integration.knowledge_pack import KnowledgePack
    from python.core.substrate import NSCKSubstrate
    
    pack = _make_pack()
    with tempfile.NamedTemporaryFile(suffix=".gz", delete=False) as f:
        path = f.name
    
    try:
        pack.save(path)
        cfg = NSCKConfig(knowledge_packs=[path])
        sub = NSCKSubstrate(config=cfg)
        # Concepts from pack should be in semantic memory
        sem = sub._engine.semantic_memory
        assert "cat" in sem.concept_graph.nodes()
    finally:
        os.unlink(path)


def test_works_without_packs():
    """NSCKSubstrate works normally when no packs configured."""
    from python.core.integration.config import NSCKConfig
    from python.core.substrate import NSCKSubstrate
    
    cfg = NSCKConfig()  # no knowledge_packs
    sub = NSCKSubstrate(config=cfg)
    sub.register_task("test")
    result = sub.ingest("hello", "test")
    assert result is not None
