"""
Tests for Phase 2: Self-Learning Engine
"""
import pytest
import time
import python.core.vsa.hypervec_shim as hypervec_rs
from python.rule_learner import RuleLearner
from python.episodic_memory import EpisodicMemory, LiveEpisode
from python.curiosity import CuriosityModule
from python.grounding_verifier import create_snake_verifier


class TestRuleLearner:
    """Test automatic rule induction."""
    
    def test_observe_and_induce(self):
        """Test that rules are induced from observations."""
        print("\n--- Test: Rule Induction ---")
        
        verifier = create_snake_verifier()
        learner = RuleLearner(verifier, min_support=3, min_success_rate=0.6)
        
        # Simulate 5 observations of same pattern
        state = {"head": (5, 5), "food": (5, 3), "body": [(5, 6)]}  # Food is above
        
        for _ in range(5):
            learner.observe(state, "UP", reward=1.0, task_tag="snake", outcome="success")
        
        # Induce rules
        new_rules = learner.induce_rules("snake")
        
        assert len(new_rules) >= 1, "Should induce at least one rule"
        
        # Check rule properties
        rule = new_rules[0]
        assert rule.consequence == "ACTION_UP"
        assert rule.support_count >= 5
        assert rule.success_rate >= 0.6
        print(f"Induced rule: {set(rule.condition)} → {rule.consequence}")
    
    def test_applicable_rules(self):
        """Test getting applicable rules for a state."""
        print("\n--- Test: Applicable Rules ---")
        
        verifier = create_snake_verifier()
        learner = RuleLearner(verifier, min_support=2, min_success_rate=0.5)
        
        # Train on two patterns
        state1 = {"head": (5, 5), "food": (5, 2), "body": [(5, 6)]}  # Food above
        state2 = {"head": (5, 5), "food": (8, 5), "body": [(5, 6)]}  # Food right
        
        for _ in range(3):
            learner.observe(state1, "UP", reward=1.0, task_tag="snake")
            learner.observe(state2, "RIGHT", reward=1.0, task_tag="snake")
        
        learner.induce_rules("snake")
        
        # Query for state1 — get_applicable_rules expects active predicates, not state
        active_preds = verifier.get_active_predicates(state1, context="snake")
        matches = learner.get_applicable_rules(active_preds, "snake")
        assert len(matches) >= 1
        best_rule, score = matches[0]
        assert "UP" in best_rule.consequence
        print(f"Best match for state1: {best_rule.consequence} (score={score:.2f})")


class TestEpisodicMemory:
    """Test episodic memory storage and retrieval."""
    
    def test_record_and_recall(self):
        """Test recording and recalling similar episodes."""
        print("\n--- Test: Episodic Memory ---")
        
        memory = EpisodicMemory(recent_capacity=100)
        
        # Create a few episodes
        for i in range(10):
            hv = hypervec_rs.HyperVector(42 + i)
            episode = LiveEpisode(
                timestamp=time.time(),
                task_tag="snake",
                situation_hv=hv,
                state={"head": (5, 5), "food": (5, 3 + i)},
                action="ACTION_UP",
                outcome="success",
                reward=1.0
            )
            memory.record(episode)
        
        # Check recent
        recent = memory.recall_recent("snake", n=5)
        assert len(recent) == 5
        
        # Query similar
        query_hv = hypervec_rs.HyperVector(45)  # Similar to episode 3
        similar = memory.recall_similar(query_hv, "snake", k=3)
        assert len(similar) <= 3
        print(f"Recalled {len(similar)} similar episodes")
    
    def test_recall_by_outcome(self):
        """Test filtering by outcome."""
        print("\n--- Test: Recall by Outcome ---")
        
        memory = EpisodicMemory(recent_capacity=100)
        
        # Mix of successes and failures
        for i in range(10):
            hv = hypervec_rs.HyperVector(100 + i)
            outcome = "success" if i % 2 == 0 else "failure"
            episode = LiveEpisode(
                timestamp=time.time(),
                task_tag="snake",
                situation_hv=hv,
                state={"head": (5, 5)},
                action="ACTION_UP",
                outcome=outcome,
                reward=1.0 if outcome == "success" else -1.0
            )
            memory.record(episode)
        
        successes = memory.recall_by_outcome("snake", "success")
        failures = memory.recall_by_outcome("snake", "failure")
        
        assert len(successes) == 5
        assert len(failures) == 5
        print(f"Successes: {len(successes)}, Failures: {len(failures)}")


class TestCuriosity:
    """Test curiosity-driven exploration."""
    
    def test_novelty_detection(self):
        """Test that novel situations are detected."""
        print("\n--- Test: Novelty Detection ---")
        
        curiosity = CuriosityModule(novelty_threshold=0.7)
        
        # First situation should be novel
        hv1 = hypervec_rs.HyperVector(12345)
        novelty1 = curiosity.compute_novelty(hv1, "snake")
        assert novelty1 == 1.0, "First situation should be completely novel"
        
        # Add as prototype
        curiosity.update_prototype(hv1, "snake")
        
        # Same situation should no longer be novel
        novelty2 = curiosity.compute_novelty(hv1, "snake")
        assert novelty2 < 0.5, "Same situation should not be novel"
        
        # Different situation should be novel
        hv2 = hypervec_rs.HyperVector(99999)
        novelty3 = curiosity.compute_novelty(hv2, "snake")
        assert novelty3 > 0.5, "Different situation should be novel"
        print(f"Novelty scores: first={novelty1:.2f}, same={novelty2:.2f}, diff={novelty3:.2f}")
    
    def test_exploration_decision(self):
        """Test exploration vs exploitation decision."""
        print("\n--- Test: Exploration Decision ---")
        
        curiosity = CuriosityModule()
        
        hv = hypervec_rs.HyperVector(5555)
        
        # Novel + low confidence = explore
        decision = curiosity.should_explore(hv, "snake", confidence=0.3)
        assert decision.should_explore, "Should explore when novel and uncertain"
        assert decision.novelty_score > 0.7
        print(f"Decision: explore={decision.should_explore}, reason={decision.reason}")
        
        # Add prototype to make familiar
        curiosity.update_prototype(hv, "snake")
        
        # Familiar + high confidence = exploit
        decision2 = curiosity.should_explore(hv, "snake", confidence=0.9)
        assert not decision2.should_explore, "Should exploit when familiar and confident"
        print(f"Decision2: explore={decision2.should_explore}, reason={decision2.reason}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
