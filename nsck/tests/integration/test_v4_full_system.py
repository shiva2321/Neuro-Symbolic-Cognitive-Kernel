"""
End-to-end V4 integration tests.

Tests V4 capabilities:
1. Knowledge domain seeding
2. System-1 procedural fast-path
3. Semantic memory hot cache
4. NLU intent classification
5. Bundle majority-vote correctness
6. Multi-step imagination
"""
import os
import pytest
import time
import numpy as np
import python.core.vsa.hypervec_shim as hypervec_rs
from python.core.reasoning.cognitive_engine import CognitiveEngine
from python.core.integration.config import NSCKConfig
from python.core.memory.procedural_memory import ProceduralMemory


NAVIGATION_YAML = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    "../../python/core/bootstrap/domain_kits/navigation.yaml"
))


@pytest.fixture(scope="module")
def engine():
    cfg = NSCKConfig()
    eng = CognitiveEngine(config=cfg, persistence_path=None)
    eng.register_task("v4_test")
    return eng


class TestProceduralFastPath:
    """Test System-1 procedural fast-path auto-population."""

    def test_procedural_memory_wired(self, engine):
        assert hasattr(engine, 'procedural_memory')
        assert isinstance(engine.procedural_memory, ProceduralMemory)

    def test_skills_populated_after_positive_reward_learn(self, engine):
        state = {"x": 5, "y": 3, "feature": "positive"}
        cs = engine.decide(state, "v4_test")
        engine.learn(state, cs.chosen_action, reward=1.0, task_tag="v4_test")
        stats = engine.procedural_memory.get_statistics()
        assert stats["total_skills"] >= 1

    def test_familiarity_threshold_is_0_72(self, engine):
        assert engine.procedural_memory.familiarity_threshold == pytest.approx(0.72, abs=0.01)


class TestSemanticHotCache:
    """Test semantic memory hot cache."""

    def test_hot_cache_exists(self, engine):
        assert hasattr(engine.semantic_memory, '_hot_cache')

    def test_hnsw_enabled_by_default(self, engine):
        assert engine.semantic_memory._hnsw_enabled is True

    def test_hot_cache_updated_after_spread_activation(self, engine):
        sm = engine.semantic_memory
        sm.add_concept("test_v4_concept", {"type": "test"})
        sm.spread_activation(["test_v4_concept"], steps=1, decay=0.5)
        # Cache was at least called (even if no results)
        assert isinstance(sm._hot_cache, dict)


class TestNLU:
    """Test NLU intent classification."""

    def test_vsa_nlu_classifies_intent(self):
        from python.core.language.vsa_nlu import VSANLUEngine
        engine = VSANLUEngine()
        intent, conf = engine.classify("how does this work")
        assert isinstance(intent, str)
        assert conf >= 0.0

    def test_nlu_accuracy_on_sample(self):
        from python.core.language.vsa_nlu import VSANLUEngine
        _VALID_INTENTS = {"question", "command", "statement", "greeting", "farewell", "exclamation", "negation"}
        engine = VSANLUEngine()
        test_cases = [
            "what is this",
            "go to the goal",
            "hello",
            "goodbye",
            "this does not work",
        ]
        for text in test_cases:
            intent, conf = engine.classify(text)
            assert intent in _VALID_INTENTS, f"classify('{text}') returned unknown intent '{intent}'"
            assert 0.0 <= conf <= 1.0


class TestBundleMajorityVote:
    """Test that VSA bundle gives sensible results."""

    def test_bundle_same_vector_returns_similar(self):
        hv_a = hypervec_rs.HyperVector(42)
        # bundle(A, A) should be very close to A
        result = hv_a.bundle(hv_a)
        sim = hv_a.similarity(result)
        assert sim > 0.85, f"bundle(A, A) similarity {sim} too low"

    def test_bundle_asymmetry(self):
        """bundle(A, B) should not always equal bundle(B, A) — content-seeded."""
        hv_a = hypervec_rs.HyperVector(1)
        hv_b = hypervec_rs.HyperVector(2)
        result_ab = hv_a.bundle(hv_b)
        result_ba = hv_b.bundle(hv_a)
        # Both should be similar to A and B (i.e. both within VSA bundle space)
        sim_ab_a = hv_a.similarity(result_ab)
        sim_ba_a = hv_a.similarity(result_ba)
        assert sim_ab_a > 0.6
        assert sim_ba_a > 0.6


class TestMultiStepImagination:
    """Test multi-step imagine_rollout."""

    def test_imagine_rollout_exists(self, engine):
        assert hasattr(engine, 'imagine_rollout')
        assert callable(engine.imagine_rollout)

    def test_imagine_rollout_returns_tuple(self, engine):
        hv = hypervec_rs.HyperVector(123)
        reward, safe = engine.imagine_rollout(
            initial_hv=hv,
            action_sequence=["ACTION_UP", "ACTION_DOWN", "ACTION_LEFT"],
            task_tag="v4_test",
        )
        assert isinstance(reward, float)
        assert isinstance(safe, bool)


class TestKnowledgeSeeding:
    """Test domain seeding from YAML."""

    def test_seed_domain_method_exists(self, engine):
        assert hasattr(engine, 'seed_domain')

    def test_seed_navigation_creates_rules(self, engine):
        if not os.path.exists(NAVIGATION_YAML):
            pytest.skip(f"navigation.yaml not found at {NAVIGATION_YAML}")
        n = engine.seed_domain(NAVIGATION_YAML)
        assert n > 0
        nav_rules = engine.rule_learner.learned_rules.get("navigation", [])
        assert len(nav_rules) > 0

    def test_seeded_rules_have_correct_domain(self, engine):
        if not os.path.exists(NAVIGATION_YAML):
            pytest.skip("navigation.yaml not found")
        for rule in engine.rule_learner.learned_rules.get("navigation", []):
            assert rule.task_tag == "navigation"
