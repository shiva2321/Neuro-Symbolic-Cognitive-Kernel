import unittest
from python.utilities.spatial_reasoning import GridPlanner

class TestHierarchicalPlanning(unittest.TestCase):
    def test_long_path_decomposition(self):
        """Test that long paths are decomposed and solved."""
        planner = GridPlanner(grid_size=20)
        
        start_state = {"AT_0_0"}
        goal_state = {"AT_8_0"} # Distance 8, exceeds threshold 6
        
        print(f"\n[TEST] Planning from {start_state} to {goal_state}...")
        
        # Verify plan_hierarchical exists
        self.assertTrue(hasattr(planner, 'plan_hierarchical'))
        
        # Run hierarchical plan
        # We assume max_depth default is plenty for subgoals
        plan = planner.plan_hierarchical(start_state, goal_state)
        
        print(f"[TEST] Plan Result: {plan}")
        
        self.assertIsNotNone(plan, "Plan should not be None")
        self.assertEqual(len(plan), 8, "Plan should be exactly 8 steps")
        self.assertEqual(plan, ["ACTION_RIGHT"] * 8)
        
    def test_diagonal_decomposition(self):
        """Test decomposition on diagonal path."""
        planner = GridPlanner(grid_size=20)
        start = {"AT_0_0"}
        goal = {"AT_6_6"} # Distance 12
        
        plan = planner.plan_hierarchical(start, goal)
        self.assertIsNotNone(plan)
        self.assertEqual(len(plan), 12)
        
        # Verify it reaches goal
        # Simulate manually or check last state? 
        # Plan is just actions.
        # We can implement a check.
        current_x, current_y = 0, 0
        for action in plan:
            if action == "ACTION_UP": current_y -= 1
            if action == "ACTION_DOWN": current_y += 1
            if action == "ACTION_LEFT": current_x -= 1
            if action == "ACTION_RIGHT": current_x += 1
            
        self.assertEqual(current_x, 6)
        self.assertEqual(current_y, 6)

if __name__ == '__main__':
    unittest.main()
