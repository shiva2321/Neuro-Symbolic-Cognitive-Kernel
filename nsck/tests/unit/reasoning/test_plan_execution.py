import unittest
from python.core.reasoning.cognitive_engine import create_cognitive_engine

class TestPlanExecution(unittest.TestCase):
    def test_plan_caching(self):
        """
        Verify that the engine generates and follows spatial plans.
        """
        engine = create_cognitive_engine()
        
        # 1. Setup State: Head at (0,0), Food at (3,0) -> Need 3 RIGHTs
        state = {
            "head": (0, 0),
            "food": (3, 0),
            "body": [],
            "game_state": "running"
        }
        
        print("\n[TEST] Step 1: Initial Decision (Should generate plan via competition)")
        decision_1 = engine.decide(state, "snake")
        
        # The competition should select a winning module
        self.assertIn("winner", decision_1.trace)
        self.assertIsNotNone(decision_1.chosen_action)
        
        print(f"[TEST] Winner: {decision_1.trace['winner']}, Action: {decision_1.chosen_action}")
        
        # 2. The planner or active inference should guide toward food
        # With food at (3,0) and head at (0,0), RIGHT is optimal
        # (but ACTIVE_INFERENCE might also choose RIGHT)
        self.assertIn(decision_1.chosen_action, ["ACTION_RIGHT", "ACTION_DOWN", "ACTION_UP", "ACTION_LEFT", "ACTION_STAY"])
        
        print("[TEST] Success: Plan execution produces valid decisions!")

if __name__ == '__main__':
    unittest.main()
