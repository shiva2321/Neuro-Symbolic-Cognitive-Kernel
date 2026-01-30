
import unittest
import time
import sys
import os
import shutil

# Add python folder to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../python')))

from rule_learner import RuleLearner, RuleCandidate

# Mock hypervec_rs if not available
try:
    import hypervec_rs
except ImportError:
    from unittest.mock import MagicMock
    hypervec_rs = MagicMock()
    # Mock the HyperVector class
    class MockHyperVector:
        def __init__(self, seed):
            self.bits = [0] * 1000 # Dummy bits
            self.seed = seed
    hypervec_rs.HyperVector = MockHyperVector

from persistence import Rule, Episode, BrainStore
from episodic_memory import LiveEpisode

class TestStability(unittest.TestCase):
    def setUp(self):
        # Create a temporary DB
        self.db_path = "test_stability.db"
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self.store = BrainStore(self.db_path)
        
        # Mock objects
        class MockVerifier:
            def get_active_predicates(self, state, context): return []
        self.learner = RuleLearner(MockVerifier(), self.store, min_support=10, min_success_rate=0.7)

    def tearDown(self):
        self.store.close()
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except PermissionError:
                pass # Windows file locking sometimes checks in

    def test_tenure_thresholds(self):
        """Verify that rules get the correct survival threshold based on tenure."""
        
        # 1. Bootstrap Rule (Should be protected -> 0.5 threshold)
        bootstrap_rule = Rule(
            id=1, condition=frozenset(["A"]), consequence="UP", 
            source="bootstrap", support_count=100
        )
        self.assertEqual(self.learner._get_tenure_threshold(bootstrap_rule), 0.5, 
                         "Bootstrap rules should have 0.5 threshold")

        # 2. High Support Rule (Should be protected -> 0.6 threshold)
        veteran_rule = Rule(
            id=2, condition=frozenset(["B"]), consequence="DOWN", 
            source="learned", support_count=1500
        )
        self.assertEqual(self.learner._get_tenure_threshold(veteran_rule), 0.6, 
                         "High-support rules should have 0.6 threshold")

        # 3. New Rule (Should be strict -> 0.7 threshold)
        newbie_rule = Rule(
            id=3, condition=frozenset(["C"]), consequence="LEFT", 
            source="learned", support_count=50
        )
        self.assertEqual(self.learner._get_tenure_threshold(newbie_rule), 0.7, 
                         "New rules should have default 0.7 threshold")

    def test_pruning_protection(self):
        """Verify that protected rules survive pruning even with low success rates."""
        
        # Add a bootstrap rule acting poorly (0.55 success rate)
        # 0.55 < 0.7 (default) BUT > 0.5 (bootstrap threshold) -> SHOULD SURVIVE
        bootstrap_rule = Rule(
            id=None, condition=frozenset(["REL_ABOVE"]), consequence="ACTION_UP",
            source="bootstrap", support_count=500, success_rate=0.55,
            task_tag="snake"
        )
        self.learner.learned_rules["snake"].append(bootstrap_rule)
        
        # Add a new learned rule acting poorly (0.65 success rate)
        # 0.65 < 0.7 (default) -> SHOULD DIE
        bad_learned_rule = Rule(
            id=None, condition=frozenset(["REL_LEFT"]), consequence="ACTION_RIGHT", # Bad rule
            source="learned", support_count=50, success_rate=0.65,
            task_tag="snake"
        )
        self.learner.learned_rules["snake"].append(bad_learned_rule)
        
        print(f"\nBefore Pruning: {len(self.learner.learned_rules['snake'])} rules")
        self.learner.prune_rules("snake")
        print(f"After Pruning: {len(self.learner.learned_rules['snake'])} rules")
        
        remaining = self.learner.learned_rules["snake"]
        self.assertEqual(len(remaining), 1, "Only one rule should survive")
        self.assertEqual(remaining[0].source, "bootstrap", "Bootstrap rule should survive")
        self.assertEqual(remaining[0].consequence, "ACTION_UP", "Correct rule preserved")

    def test_impact_score_persistence(self):
        """Verify that impact scores are saved and loaded correctly."""
        
        # Create a dummy hypervector
        hv = hypervec_rs.HyperVector(123)
        
        # Create live episode with impact score
        ep = LiveEpisode(
            timestamp=time.time(),
            task_tag="test_task",
            situation_hv=hv,
            state={"x": 1},
            action="UP",
            outcome="success",
            reward=1.0,
            impact_score=99.9 # Unique value
        )
        
        # Persist
        stored_ep = ep.to_stored()
        self.store.record_episode(stored_ep)
        self.store.flush_episodes()
        
        # Load back
        recent = self.store.load_recent_episodes("test_task", limit=10)
        loaded_ep = recent[0]
        
        self.assertEqual(loaded_ep.impact_score, 99.9, "Impact score was not persisted correctly")
        print(f"\nPersisted Impact Score: {loaded_ep.impact_score}")

if __name__ == '__main__':
    unittest.main()
