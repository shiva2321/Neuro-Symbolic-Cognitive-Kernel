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
        # Mock dependencies to control salience
        self.engine.episodic_memory = MagicMock()
        self.engine.world_model = MagicMock()
        self.engine.world_model.hv_to_numpy.return_value = np.zeros(10240)
        self.engine.world_model.predictor.predict.return_value = (np.zeros(10240), 0.0)
        
    def test_planner_priority(self):
        """Test that Planner wins when salience is 1.0."""
        self.engine.active_plan = ["ACTION_UP", "ACTION_UP"]
        
        # SNN proposes something else with lower salience
        meta_result = {"action": "ACTION_LEFT", "confidence": 0.5}
        
        state = {"head": (5,5), "food": (5,7)} # Default would move UP/DOWN
        result = self.engine.decide(state, "snake", metacognition_result=meta_result)
        
        self.assertEqual(result.chosen_action, "ACTION_UP")
        self.assertEqual(result.trace["winner"], "PLANNER")
        self.assertEqual(len(self.engine.active_plan), 1)

    def test_snn_veto_by_simulation(self):
        """Test that dangerous SNN advice loses salience and fallback wins."""
        # Ensure no accidental planning
        self.engine._try_spatial_planning = MagicMock(return_value=None)
        
        # Mock world model to predict death for LEFT
        # We need to mock imagine_rollout or the predictor
        self.engine.imagine_rollout = MagicMock(return_value=-1.0)
        
        meta_result = {"action": "ACTION_LEFT", "confidence": 0.9} # High confidence
        
        # No plan, no rules
        self.engine.rule_learner.get_applicable_rules = MagicMock(return_value=[])
        
        state = {"head": (5,5), "food": (5,6)} # Default move DOWN
        result = self.engine.decide(state, "snake", metacognition_result=meta_result)
        
        # SNN salience should be tiny (0.9 * 0.05 = 0.045), less than default attention threshold (0.5)
        # So it should hit DEFAULT
        self.assertEqual(result.trace["winner"], "DEFAULT")
        self.assertEqual(result.chosen_action, "ACTION_DOWN")

    def test_competition_snn_vs_rules(self):
        """Test competition between SNN and Rules."""
        # No plan
        self.engine.active_plan = []
        
        # SNN proposes LEFT with 0.6
        meta_result = {"action": "ACTION_LEFT", "confidence": 0.6}
        
        # Rule proposes RIGHT with 0.8
        mock_rule = MagicMock()
        mock_rule.consequence = "ACTION_RIGHT"
        self.engine.rule_learner.get_applicable_rules = MagicMock(return_value=[(mock_rule, 0.8)])
        
        # Safe world
        self.engine.imagine_rollout = MagicMock(return_value=0.0)
        
        state = {"head": (5,5), "food": (5,5)}
        result = self.engine.decide(state, "snake", metacognition_result=meta_result)
        
        self.assertEqual(result.trace["winner"], "RULES")
        self.assertEqual(result.chosen_action, "ACTION_RIGHT")

if __name__ == "__main__":
    unittest.main()
