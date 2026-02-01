import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cognitive_engine import create_cognitive_engine

class TestMazeValidation(unittest.TestCase):
    def setUp(self):
        self.engine = create_cognitive_engine()
        # Mock world model to prevent slow neural simulations for unit test
        self.engine.imagine_rollout = MagicMock(return_value=0.0)
        
    def test_zero_shot_maze_navigation(self):
        """[AGI 6.4] Test that the agent can navigate a Maze zero-shot."""
        # Setup Maze state: Target is at (8, 2), walls surrounding top/left
        state = {
            "head": (1, 1),
            "target": (8, 2),
            "walls": [(1, 0), (0, 1), (2, 1), (1, 2)] # Small box for testing vetoes
        }
        task = "maze"
        
        print(f"\n[MAZE TRIAL] START: Head at (1,1), Target at (8,2)")
        
        # 1. First step: Planner should find a path
        result = self.engine.decide(state, task)
        
        print(f"[MAZE TRIAL] Step 1: Winner={result.trace.get('winner')}, Action={result.chosen_action}")
        self.assertEqual(result.trace["winner"], "PLANNER")
        self.assertIsNotNone(result.chosen_action)
        
        # 2. Veto Test: If SNN wants to hit a wall, Rule/Causal should veto
        # Force SNN to want to go UP (where a wall is at (1,0))
        meta = {"action": "ACTION_UP", "confidence": 0.9}
        
        # First ensure verifier sees the wall
        active_preds = self.engine.verifiers[task].get_active_predicates(state, context=task)
        self.assertIn("maze::WALL_AT_UP", active_preds)
        
        # Mock imagine_rollout to predict failure for ACTION_UP
        def mock_imagine(hv, acts, tag):
            if "ACTION_UP" in acts:
                return -1.0 # Deadly collision
            return 0.1
        self.engine.imagine_rollout = MagicMock(side_effect=mock_imagine)
        
        # Decision with high-confidence SNN 'UP' advice
        result_veto = self.engine.decide(state, task, metacognition_result=meta)
        
        print(f"[MAZE TRIAL] Step 2 (Veto): SNN proposed UP, Result Winner={result_veto.trace.get('winner')}, Action={result_veto.chosen_action}")
        
        # Action should NOT be UP
        self.assertNotEqual(result_veto.chosen_action, "ACTION_UP")
        # Planner should still win or Default if plan was cleared, but SNN must lose salience
        self.assertIn("ACTION_UP", result_veto.trace.get("vetoes", []))

    def test_goal_decomposition_in_maze(self):
        """[AGI 6.4] Test that hierarchical planning works for long distances in Maze."""
        # Long distance: 1,1 to 9,9
        state = {
            "head": (1, 1),
            "target": (9, 9),
            "walls": []
        }
        task = "maze"
        
        result = self.engine.decide(state, task)
        
        # Check if plan target was set (sub-goal)
        print(f"[MAZE TRIAL] Long Path: Target=(9,9), Chosen Sub-goal={self.engine.plan_target}")
        self.assertIsNotNone(self.engine.plan_target)
        self.assertTrue(len(self.engine.active_plan) > 0)

if __name__ == "__main__":
    unittest.main()
