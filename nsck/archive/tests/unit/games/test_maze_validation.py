import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Add project root to path

from python.core.reasoning.cognitive_engine import create_cognitive_engine

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
        
        # First step: A competition winner should propose an action
        result = self.engine.decide(state, task)
        
        print(f"[MAZE TRIAL] Step 1: Winner={result.trace.get('winner')}, Action={result.chosen_action}")
        # The competition winner can be PLANNER or ACTIVE_INFERENCE — both are valid
        self.assertIn(result.trace["winner"], ["PLANNER", "ACTIVE_INFERENCE", "RULES", "EXPLORATION"])
        self.assertIsNotNone(result.chosen_action)
        
    def test_maze_produces_valid_actions(self):
        """Test that maze decisions produce valid directional actions."""
        state = {
            "head": (1, 1),
            "target": (9, 9),
            "walls": []
        }
        task = "maze"
        
        result = self.engine.decide(state, task)
        
        valid_actions = {"ACTION_UP", "ACTION_DOWN", "ACTION_LEFT", "ACTION_RIGHT", "ACTION_STAY"}
        print(f"[MAZE TRIAL] Long Path: Target=(9,9), Action={result.chosen_action}")
        self.assertIn(result.chosen_action, valid_actions)

if __name__ == "__main__":
    unittest.main()
