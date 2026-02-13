"""
Test Integration (Phase 5)
Verifies: ActionSemantics -> MetacognitiveWrapper -> MetacognitiveEngine -> FusedBrain
"""
import unittest
import numpy as np
from python.core.perception.symbol_grounding import ActionSemantics

class TestIntegration(unittest.TestCase):
    
    def test_snake_goal_alignment(self):
        """Test Snake Goal Alignment (Simple Patter Match)."""
        # State: Food is ABOVE Head
        state = {"head": (5, 5), "food": (5, 4), "body": [(5, 6)]}
        
        # Expectation: UP should have highest score
        # Rules: REL_ABOVE -> ACTION_UP
        scores = ActionSemantics.get_goal_alignment("snake", state)
        
        print(f"\nSnake Scores (Food Above): {scores}")
        
        # UP is index 0
        self.assertGreater(scores[0], scores[1]) # UP > DOWN
        self.assertGreater(scores[0], scores[2]) # UP > LEFT
        
    def test_pong_goal_alignment(self):
        """Test Pong Goal Alignment."""
        # State: Ball is ABOVE Paddle
        # SafetyGate requires ball physics (x, y, dx, dy)
        state = {
            "p1_y": 10, 
            "ball_y": 5,
            "ball_x": 15,
            "ball_dx": 1,
            "ball_dy": 1
        }
        
        scores = ActionSemantics.get_goal_alignment("pong", state)
        print(f"\nPong Scores (Ball Above): {scores}")
        
        # UP is index 0
        self.assertGreater(scores[0], scores[1]) # UP > DOWN
        
    def test_wrapper_structure(self):
        """Verify wrapper initialized correct engine."""
        from python.core.perception.symbol_grounding import get_kernel_engine
        engine_wrapper = get_kernel_engine()
        self.assertTrue(hasattr(engine_wrapper, "engine"))
        brain = engine_wrapper.engine.brain
        
        # Check Rules exist
        self.assertTrue(len(brain.task_rules["snake"]) > 0)
        self.assertTrue(len(brain.task_rules["pong"]) > 0)
        
        # Check rule content
        rules = brain.task_rules["snake"]
        rel_above_rules = [r for r in rules if "REL_ABOVE" in r.condition]
        self.assertTrue(len(rel_above_rules) > 0)
        self.assertEqual(rel_above_rules[0].consequence, "ACTION_UP")

if __name__ == "__main__":
    unittest.main()
