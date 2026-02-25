"""Tests for V8 memory lifecycle: decay, prune, access tracking."""
import pytest
import time

from python.core.memory.semantic_memory import SemanticMemory


@pytest.fixture()
def mem():
    m = SemanticMemory()
    for c in ["cat", "dog", "fish", "bird", "snake"]:
        m.add_concept(c, {"type": "animal"})
    return m


class TestAddConceptAttributes:
    def test_access_count_initialized(self, mem):
        assert mem.concept_graph.nodes["cat"].get("access_count", 0) == 0

    def test_last_accessed_initialized(self, mem):
        node = mem.concept_graph.nodes["cat"]
        assert "last_accessed" in node
        assert node["last_accessed"] > 0

    def test_importance_score_initialized(self, mem):
        node = mem.concept_graph.nodes["cat"]
        assert node.get("importance_score", 1.0) == 1.0

    def test_multiple_concepts_have_attrs(self, mem):
        for c in ["cat", "dog", "fish"]:
            assert "importance_score" in mem.concept_graph.nodes[c]


class TestGetConceptUpdatesAccess:
    def test_get_increments_access_count(self, mem):
        mem.get_concept("cat")
        assert mem.concept_graph.nodes["cat"]["access_count"] >= 1

    def test_get_updates_last_accessed(self, mem):
        t_before = time.time()
        mem.get_concept("dog")
        t_after = time.time()
        la = mem.concept_graph.nodes["dog"]["last_accessed"]
        assert t_before <= la <= t_after + 0.1

    def test_get_missing_returns_none(self, mem):
        assert mem.get_concept("nonexistent") is None

    def test_multiple_gets_increments_count(self, mem):
        for _ in range(5):
            mem.get_concept("fish")
        assert mem.concept_graph.nodes["fish"]["access_count"] >= 5

    def test_get_returns_hv(self, mem):
        hv = mem.get_concept("cat")
        assert hv is not None


class TestDecayConcepts:
    def test_decay_returns_count(self, mem):
        count = mem.decay_concepts(0.01)
        assert count == len(list(mem.concept_graph.nodes()))

    def test_decay_reduces_importance(self, mem):
        # Manually set last_accessed to 1 hour ago
        for node in mem.concept_graph.nodes():
            mem.concept_graph.nodes[node]["last_accessed"] = time.time() - 3600
        mem.decay_concepts(1.0)  # high lambda to force visible change
        for node in mem.concept_graph.nodes():
            score = mem.concept_graph.nodes[node]["importance_score"]
            assert score < 1.0

    def test_decay_zero_lambda_no_change(self, mem):
        orig = {n: mem.concept_graph.nodes[n].get("importance_score", 1.0)
                for n in mem.concept_graph.nodes()}
        mem.decay_concepts(0.0)
        for n, v in orig.items():
            assert mem.concept_graph.nodes[n]["importance_score"] == pytest.approx(v)

    def test_decay_empty_memory(self):
        m = SemanticMemory()
        count = m.decay_concepts()
        assert count == 0


class TestPruneBelow:
    def test_prune_removes_low_importance(self, mem):
        for n in list(mem.concept_graph.nodes()):
            mem.concept_graph.nodes[n]["importance_score"] = 0.0
        removed = mem.prune_below(0.1)
        assert removed == 5

    def test_prune_keeps_high_importance(self, mem):
        removed = mem.prune_below(0.1)
        assert removed == 0
        assert len(mem.concept_hvs) == 5

    def test_prune_removes_from_hvs_too(self, mem):
        for n in list(mem.concept_graph.nodes()):
            mem.concept_graph.nodes[n]["importance_score"] = 0.0
        mem.prune_below(0.1)
        assert len(mem.concept_hvs) == 0

    def test_prune_partial(self, mem):
        mem.concept_graph.nodes["snake"]["importance_score"] = 0.05
        removed = mem.prune_below(0.1)
        assert removed == 1
        assert "snake" not in mem.concept_hvs

    def test_prune_returns_zero_when_all_healthy(self, mem):
        assert mem.prune_below(0.001) == 0

    def test_prune_empty_memory(self):
        m = SemanticMemory()
        assert m.prune_below() == 0


class TestDecayAndPrunePipeline:
    def test_decay_then_prune(self, mem):
        for n in list(mem.concept_graph.nodes()):
            mem.concept_graph.nodes[n]["last_accessed"] = time.time() - 7200
        mem.decay_concepts(10.0)
        removed = mem.prune_below(0.001)
        assert removed >= 0

    def test_add_and_prune(self):
        m = SemanticMemory()
        m.add_concept("healthy", {"v": 1})
        m.concept_graph.nodes["healthy"]["importance_score"] = 0.5
        m.add_concept("sick", {"v": 2})
        m.concept_graph.nodes["sick"]["importance_score"] = 0.05
        removed = m.prune_below(0.1)
        assert removed == 1
        assert "healthy" in m.concept_hvs
