import sys
import os
import unittest
from unittest.mock import MagicMock
import numpy as np

# Add project root to path

from python.core.learning.curiosity import CuriosityModule
import python.core.vsa.hypervec_shim as hypervec_rs

class TestCrossModule(unittest.TestCase):
    def test_planner_guided_icm(self):
        """Test that sub-goals boost novelty."""
        curiosity = CuriosityModule(novelty_threshold=0.7)
        task = "snake"
        
        # Create a situation and a goal
        sit_hv = hypervec_rs.HyperVector(1)
        goal_hv = hypervec_rs.HyperVector(1) # Exact match for max bonus
        
        # Make the situation FAMILIAR (add it as prototype)
        curiosity.update_prototype(sit_hv, task)
        
        # Baseline novelty (should be ~0.0 now because it matches prototype)
        base_nov = curiosity.compute_novelty(sit_hv, task)
        decision_base = curiosity.should_explore(sit_hv, task, confidence=0.4)
        
        # Set subgoals
        curiosity.set_subgoals(task, [goal_hv])
        
        # should_explore with goals
        decision_goal = curiosity.should_explore(sit_hv, task, confidence=0.4)
        
        print(f"[TEST] Familiar State Novelty: {decision_base.novelty_score:.2f}")
        print(f"[TEST] Goal-Boosted Novelty: {decision_goal.novelty_score:.2f}")
        
        # Goal-boosted should be significantly higher
        self.assertGreater(decision_goal.novelty_score, decision_base.novelty_score)

if __name__ == "__main__":
    unittest.main()
