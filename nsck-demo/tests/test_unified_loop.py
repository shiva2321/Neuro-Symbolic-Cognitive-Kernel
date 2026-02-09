import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cognitive_engine import create_cognitive_engine

class TestUnifiedLoop(unittest.TestCase):
    def setUp(self):
        self.engine = create_cognitive_engine()
        
    def test_planner_priority(self):
        """Test that the Global Workspace competition produces a winner."""
        # SNN proposes something
        meta_result = {"action": "ACTION_LEFT", "confidence": 0.5}
        
        state = {"head": (5,5), "food": (5,7)}
        result = self.engine.decide(state, "snake", metacognition_result=meta_result)
        
        # Competition should select a winner from valid sources
        valid_winners = {"SNN", "PLANNER", "ACTIVE_INFERENCE", "RULES", "EXPLORATION", "DEFAULT", "IMAGINATION"}
        self.assertIn(result.trace["winner"], valid_winners)
        self.assertIsNotNone(result.chosen_action)
        
        print(f"[TEST] Winner: {result.trace['winner']}, Action: {result.chosen_action}")

    def test_snn_competes_in_workspace(self):
        """Test that SNN proposal enters the Global Workspace competition."""
        meta_result = {"action": "ACTION_LEFT", "confidence": 0.9} # High confidence
        
        state = {"head": (5,5), "food": (5,6)}
        result = self.engine.decide(state, "snake", metacognition_result=meta_result)
        
        # SNN should at least be a proposal (may or may not win)
        self.assertTrue(result.trace["proposals"] >= 2)  # SNN + at least one other
        self.assertIsNotNone(result.chosen_action)

    def test_competition_snn_vs_rules(self):
        """Test competition between SNN and Rules."""
        # SNN proposes LEFT with 0.6
        meta_result = {"action": "ACTION_LEFT", "confidence": 0.6}
        
        # Rule proposes RIGHT with 0.8
        mock_rule = MagicMock()
        mock_rule.consequence = "ACTION_RIGHT"
        self.engine.rule_learner.get_applicable_rules = MagicMock(return_value=[(mock_rule, 0.8)])
        
        state = {"head": (5,5), "food": (5,5)}
        result = self.engine.decide(state, "snake", metacognition_result=meta_result)
        
        # Rules have higher base_salience (0.8) than SNN (0.6), so RULES should win
        self.assertEqual(result.trace["winner"], "RULES")
        self.assertEqual(result.chosen_action, "ACTION_RIGHT")

if __name__ == "__main__":
    unittest.main()
