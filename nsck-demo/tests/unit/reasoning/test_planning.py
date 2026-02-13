import unittest
from python.core.reasoning.planner import STRIPSPlanner
from python.core.reasoning.causal_reasoning import CausalGraph, CausalReasoner

class MockCausalGraph(CausalGraph):
    """
    Mock graph that simulates grid movement for testing.
    Standard CausalGraph is propositional ("Head Moves Up"), 
    but for Planning we need State Transitions ("At 5,5" -> "At 5,4").
    """
    def get_immediate_effects(self, action: str) -> list:
        # This wrapper calls the planner's _predict logic which calls this.
        # But wait, planner calls reasoner.graph.get_immediate_effects(action)
        
        # This returns generic effects like "HEAD_MOVES_UP".
        # It does NOT return "AT_5_4" because the graph is static.
        return super().get_immediate_effects(action)

class TestPlanner(unittest.TestCase):
    def test_planning_bfs(self):
        # 1. Setup Planner
        planner = STRIPSPlanner()
        
        # 2. Mock Reasoner with specific transition logic for "AT_X_Y"
        # Since our static CausalGraph doesn't know about coordinates, 
        # we need to inject a smarter reasoner or mock the _predict method 
        # for this specific test case.
        
        class GridReasoner:
             def __init__(self):
                 self.graph = MockCausalGraph()
                 
             # We hack the graph accesor to return dynamic effects based on state?
             # No, Planner calls planner._predict(current_state, action)
             # _predict calls reasoner.graph.get_immediate_effects(action) ==> Static
             
             # ISSUE: The current implementation of _predict in planner.py is too simple.
             # It relies on static graph effects.
             # It ignores the input 'state' in _predict!
             pass
        
        # REDESIGN FOR TEST:
        # We will subclass Planner to override _predict for Grid World logic.
        class GridPlanner(STRIPSPlanner):
            def _predict(self, state, action):
                # Find current position predicate "AT_X_Y"
                x, y = 5, 5
                old_pred = None
                
                for pred in state:
                    if pred.startswith("AT_"):
                        parts = pred.split("_")
                        try:
                            x, y = int(parts[1]), int(parts[2])
                            old_pred = pred
                        except:
                            pass
                        # Important: Don't break immediately if multiple ATs exist ( shouldn't happen with correct logic)
                        # but we want the "latest"? Since it's a set, order is undefined.
                        # We rely on unique AT per state.
                
                new_x, new_y = x, y
                if action == "ACTION_UP": new_y -= 1
                if action == "ACTION_DOWN": new_y += 1
                if action == "ACTION_LEFT": new_x -= 1
                if action == "ACTION_RIGHT": new_x += 1
                
                add_effects = {f"AT_{new_x}_{new_y}"}
                del_effects = {old_pred} if old_pred else set()
                
                return add_effects, del_effects

        planner = GridPlanner()
        
        start_state = {"AT_5_5"}
        goal_state = {"AT_7_5"} # Right 2 steps
        
        print(f"Planning from {start_state} to {goal_state}...")
        plan = planner.plan(start_state, goal_state, max_depth=5)
        
        print(f"Plan found: {plan}")
        
        self.assertIsNotNone(plan)
        self.assertEqual(len(plan), 2)
        self.assertEqual(plan, ["ACTION_RIGHT", "ACTION_RIGHT"])

if __name__ == '__main__':
    unittest.main()
