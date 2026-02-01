import unittest
from cognitive_engine import create_cognitive_engine

class TestPlanExecution(unittest.TestCase):
    def test_plan_caching(self):
        """
        Verify that the engine caches plans and executes them sequentially.
        """
        engine = create_cognitive_engine()
        
        # 1. Setup State: Head at (0,0), Food at (3,0) -> Need 3 RIGHTs
        state = {
            "head": (0, 0),
            "food": (3, 0),
            "body": [],
            "game_state": "running"
        }
        
        print("\n[TEST] Step 1: Initial Decision (Should generate plan)")
        # This call should generate a new plan of length 3
        decision_1 = engine.decide(state, "snake")
        self.assertEqual(decision_1.chosen_action, "ACTION_RIGHT")
        self.assertEqual(decision_1.trace["mode"], "new_plan_generated")
        self.assertEqual(len(engine.active_plan), 2) # Remaining steps
        
        # 2. Step 2: Simulate move to (1,0)
        # Food still at (3,0)
        state["head"] = (1, 0)
        
        print("[TEST] Step 2: Follow-up Decision (Should use cached plan)")
        decision_2 = engine.decide(state, "snake")
        self.assertEqual(decision_2.chosen_action, "ACTION_RIGHT")
        self.assertEqual(decision_2.trace["mode"], "continued_plan")
        self.assertEqual(decision_2.trace["steps_remaining"], 1)
        self.assertEqual(len(engine.active_plan), 1)
        
        # 3. Step 3: Simulate move to (2,0)
        state["head"] = (2, 0)
        
        print("[TEST] Step 3: Final Step (Should finish plan)")
        decision_3 = engine.decide(state, "snake")
        self.assertEqual(decision_3.chosen_action, "ACTION_RIGHT")
        self.assertEqual(decision_3.trace["mode"], "continued_plan")
        self.assertEqual(decision_3.trace["steps_remaining"], 0)
        self.assertEqual(len(engine.active_plan), 0)
        
        # 4. Step 4: Arrived at Food (3,0)
        # Should return None (plan done) or generic default action since plan is empty
        # If we are at food, logic in _try_spatial_planning says "Already there" -> returns None
        # So we fall back to other modules (Exploration or Rule)
        # Since we have no rules and explore is off (confidence?), default is move to food?
        # But we are ON food. Default action might fail/irrelevant.
        state["head"] = (3, 0)
        
        print("[TEST] Step 4: At Goal (Should not crash)")
        decision_4 = engine.decide(state, "snake")
        print(f"Action at goal: {decision_4.chosen_action}")
        self.assertNotEqual(decision_4.trace.get("mode"), "continued_plan")
        
        print("[TEST] Success: Plan cached and executed!")

if __name__ == '__main__':
    unittest.main()
