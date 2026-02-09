import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cognitive_engine import create_cognitive_engine
from persistence import Rule

class TestEmergence(unittest.TestCase):
    def setUp(self):
        self.engine = create_cognitive_engine()
        # Mock world model to prevent slow simulations
        self.engine.imagine_rollout = MagicMock(return_value=0.0)
        
    def test_zero_shot_symbolic_transfer(self):
        """[AGI 6.3] Test that a global rule transfers from Snake logic to Pong."""
        # 1. Manually add a "Global Safety Rule"
        danger_rule = Rule(
            id=999,
            condition=frozenset(["DANGER_UP"]),
            consequence="ACTION_DOWN", # Move away from top boundary
            priority=10,
            source="learned",
            task_tag="global",
            scope="global",
            support_count=100,
            success_rate=1.0
        )
        self.engine.rule_learner.learned_rules["global"].append(danger_rule)
        
        # 2. Setup Pong state where DANGER_UP is true
        state = {"p1_y": 0, "ball_y": 10, "ball_dx": -1}
        
        # We patch the verifier to return DANGER_UP
        with patch.object(self.engine.verifiers["pong"], 'get_active_predicates', return_value=["DANGER_UP"]):
            # 3. Decision in Pong context
            result = self.engine.decide(state, "pong")
            
            print(f"[TEST] Transfer Result in Pong: {result.chosen_action} (Winner: {result.trace.get('winner')})")
            
            # Assert rule was applied
            self.assertEqual(result.chosen_action, "ACTION_DOWN")
            self.assertEqual(result.trace["winner"], "RULES")

if __name__ == "__main__":
    unittest.main()
