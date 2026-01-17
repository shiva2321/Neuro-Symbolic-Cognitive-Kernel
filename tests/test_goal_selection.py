"""
Tests for Phase 5 Goal Selection in System2

Tests the Goal API: setting goals, selecting by utility,
selecting with curiosity weighting, and progress tracking.
"""

import unittest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.system2 import System2, Goal


class TestGoalSelection(unittest.TestCase):
    """Tests for Phase 5 goal selection."""

    def setUp(self):
        """Set up System2 instance for each test."""
        self.system2 = System2()

    def test_set_goals(self):
        """Test setting goals in System2."""
        goals = [
            Goal(goal_id=1, description="Goal A", utility=0.8),
            Goal(goal_id=2, description="Goal B", utility=0.5),
            Goal(goal_id=3, description="Goal C", utility=0.3),
        ]

        self.system2.set_goals(goals)

        all_goals = self.system2.get_all_goals()
        self.assertEqual(len(all_goals), 3)
        self.assertIn(goals[0], all_goals)
        self.assertIn(goals[1], all_goals)
        self.assertIn(goals[2], all_goals)

    def test_select_goal_by_utility(self):
        """Test goal selection based on utility (default policy)."""
        goals = [
            Goal(goal_id=1, description="Low priority", utility=0.2),
            Goal(goal_id=2, description="High priority", utility=0.9),
            Goal(goal_id=3, description="Medium priority", utility=0.5),
        ]

        self.system2.set_goals(goals)

        # Select goal without curiosity (beta=0)
        selected = self.system2.select_goal(beta_curiosity=0.0)

        self.assertIsNotNone(selected)
        self.assertEqual(selected.goal_id, 2)  # Should select highest utility
        self.assertEqual(selected.state, "pursuing")

    def test_select_goal_with_curiosity(self):
        """Test goal selection with curiosity weighting."""
        goals = [
            Goal(goal_id=1, description="High utility, low novelty", utility=0.8),
            Goal(goal_id=2, description="Medium utility, high novelty", utility=0.5),
        ]

        self.system2.set_goals(goals)

        # Provide novelty scores
        novelty = {1: 0.1, 2: 0.9}

        # With high beta_curiosity, should prefer high novelty goal
        selected = self.system2.select_goal(
            beta_curiosity=0.5,
            novelty_scores=novelty
        )

        self.assertIsNotNone(selected)
        # Goal 2: 0.5 + 0.5*0.9 = 0.95
        # Goal 1: 0.8 + 0.5*0.1 = 0.85
        self.assertEqual(selected.goal_id, 2)

    def test_select_goal_custom_policy(self):
        """Test goal selection with custom policy."""
        goals = [
            Goal(goal_id=1, utility=0.8, progress=0.0),
            Goal(goal_id=2, utility=0.5, progress=0.7),
            Goal(goal_id=3, utility=0.3, progress=0.9),
        ]

        self.system2.set_goals(goals)

        # Custom policy: select goal closest to completion
        def policy(available_goals):
            return max(available_goals, key=lambda g: g.progress).goal_id

        selected = self.system2.select_goal(policy=policy)

        self.assertIsNotNone(selected)
        self.assertEqual(selected.goal_id, 3)  # Highest progress

    def test_evaluate_goal_progress(self):
        """Test goal progress evaluation."""
        def reward_model(state):
            return state.get("score", 0.0)

        goal = Goal(
            goal_id=1,
            description="Test goal",
            utility=0.7,
            reward_model=reward_model
        )

        self.system2.set_goals([goal])

        # Start with low progress
        current_state = {"score": 0.3}
        progress = self.system2.evaluate_goal_progress(1, current_state)
        self.assertAlmostEqual(progress, 0.3, places=2)

        # Increase progress
        current_state = {"score": 0.8}
        progress = self.system2.evaluate_goal_progress(1, current_state)
        self.assertAlmostEqual(progress, 0.8, places=2)

        # Nearly achieved
        current_state = {"score": 0.99}
        progress = self.system2.evaluate_goal_progress(1, current_state)
        self.assertAlmostEqual(progress, 0.99, places=2)
        self.assertEqual(goal.state, "achieved")

    def test_goal_state_transitions(self):
        """Test goal state transitions (idle -> pursuing -> achieved/failed)."""
        goal = Goal(goal_id=1, description="Test", utility=0.5, state="idle")
        self.system2.set_goals([goal])

        self.assertEqual(goal.state, "idle")

        # Select goal
        selected = self.system2.select_goal()
        self.assertEqual(selected.state, "pursuing")

        # Mark as failed
        self.system2.mark_goal_failed(1)
        self.assertEqual(goal.state, "failed")

        # Failed goals shouldn't be selected
        goal2 = Goal(goal_id=2, description="Active", utility=0.6)
        self.system2.set_goals([goal2])

        selected = self.system2.select_goal()
        self.assertEqual(selected.goal_id, 2)  # Should skip failed goal 1

    def test_no_available_goals(self):
        """Test behavior when no goals are available."""
        # Empty goals
        selected = self.system2.select_goal()
        self.assertIsNone(selected)

        # All goals achieved
        goals = [
            Goal(goal_id=1, state="achieved", utility=0.8),
            Goal(goal_id=2, state="failed", utility=0.5),
        ]
        self.system2.set_goals(goals)

        selected = self.system2.select_goal()
        self.assertIsNone(selected)


if __name__ == '__main__':
    unittest.main()
