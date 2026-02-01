import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import time

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

    def test_rapid_causal_adaptation(self):
        """[AGI 6.3] Test that the system quickly adapts to a novel obstacle mid-run."""
        task = "snake"
        state_poison = {"head": (1,1), "food": (5,5), "on_poison": True}
        
        # Mock verifier
        def mock_get_active(state, context=None):
            preds = ["REL_RIGHT", "REL_BELOW"]
            if state.get("on_poison"):
                preds.append("POISON_TILE")
            return preds
            
        self.engine.verifiers[task].get_active_predicates = MagicMock(side_effect=mock_get_active)
        
        # [AVOIDANCE LOGIC]
        # Instead of learning a rule (which takes many samples of success), 
        # we check if Causal Discovery + Veto works with single/few failures.
        
        # 1. Inject a causal hypothesis that POISON_TILE leads to FAILURE
        self.engine.causal_discovery.get_hypotheses = MagicMock(return_value=[("POISON_TILE", "FAILURE")])
        
        # 2. Cogntive Engine should now Veto actions on POISON_TILE if they look dangerous
        # We mock imagine_rollout to be dangerous when POISON_TILE is active
        def mock_imagine(hv, acts, tag):
            # If we are in state_poison (which we are in the call below)
            return -0.9
            
        self.engine.imagine_rollout = MagicMock(side_effect=mock_imagine)
        
        # SNN wants to move RIGHT
        meta_result = {"action": "ACTION_RIGHT", "confidence": 0.9}
        
        # Decision
        result = self.engine.decide(state_poison, task, metacognition_result=meta_result)
        
        print(f"[TEST] Adaptation on Poison: {result.chosen_action} (Winner: {result.trace.get('winner')})")
        
        # ACTION_RIGHT should be vetoed (salience reduced to near-zero)
        # It should fall back to DEFAULT or another proposer
        self.assertNotEqual(result.chosen_action, "ACTION_RIGHT")
        self.assertIn("ACTION_RIGHT", result.trace.get("vetoes", []))

if __name__ == "__main__":
    unittest.main()
