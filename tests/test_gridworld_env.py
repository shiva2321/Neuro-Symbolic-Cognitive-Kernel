"""
Tests for Phase 5 Gridworld Environment

Tests environment mechanics, reward computation, and episode termination.
"""

import unittest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from experiments.gridworld_env import SimpleGridworld, GreedyAgent, RandomAgent


class TestGridworldEnv(unittest.TestCase):
    """Tests for Phase 5 gridworld environment."""

    def setUp(self):
        """Set up gridworld for each test."""
        self.env = SimpleGridworld(grid_size=5, max_steps=50)

    def test_reset_initializes_state(self):
        """Test that reset returns to initial state."""
        obs = self.env.reset()

        self.assertIsInstance(obs, list)
        self.assertEqual(len(obs), 5)  # [agent_row, agent_col, goal_row, goal_col, distance]
        self.assertEqual(self.env.agent_pos, (0, 0))
        self.assertEqual(self.env.steps, 0)
        self.assertFalse(self.env.done)

    def test_actions_move_agent(self):
        """Test that actions correctly move the agent."""
        self.env.reset()

        # Action 1: Down
        obs, _, _ = self.env.step(1)
        self.assertEqual(self.env.agent_pos[0], 1)  # Row increased

        # Action 3: Right
        obs, _, _ = self.env.step(3)
        self.assertEqual(self.env.agent_pos[1], 1)  # Col increased

        # Action 0: Up
        obs, _, _ = self.env.step(0)
        self.assertEqual(self.env.agent_pos[0], 0)  # Row decreased

        # Action 2: Left
        obs, _, _ = self.env.step(2)
        self.assertEqual(self.env.agent_pos[1], 0)  # Col decreased

    def test_walls_block_movement(self):
        """Test that agent cannot move outside grid."""
        self.env.reset()

        # Try to move up from (0, 0) - should stay at (0, 0)
        self.env.step(0)
        self.assertEqual(self.env.agent_pos, (0, 0))

        # Try to move left from (0, 0) - should stay at (0, 0)
        self.env.step(2)
        self.assertEqual(self.env.agent_pos, (0, 0))

    def test_reward_for_reaching_goal(self):
        """Test that reaching goal gives high reward and terminates episode."""
        self.env.reset()
        self.env.agent_pos = (4, 3)  # One step from goal

        obs, reward, done = self.env.step(3)  # Move right to goal

        self.assertEqual(self.env.agent_pos, (4, 4))
        self.assertEqual(reward, 1.0)  # High reward
        self.assertTrue(done)

    def test_reward_for_progress(self):
        """Test reward based on progress toward goal."""
        self.env.reset()

        # Move toward goal should give positive reward
        obs, reward, done = self.env.step(1)  # Down (toward goal)
        self.assertGreater(reward, 0.0)

        # Move away from goal should give negative reward
        obs, reward, done = self.env.step(0)  # Up (away from goal)
        self.assertLess(reward, 0.0)

    def test_episode_terminates_at_max_steps(self):
        """Test that episode terminates after max steps."""
        env = SimpleGridworld(grid_size=5, max_steps=10)
        env.reset()

        # Take 10 steps without reaching goal
        done = False
        for i in range(10):
            obs, reward, done = env.step(0)  # Keep trying to go up

        self.assertTrue(done)
        self.assertEqual(env.steps, 10)

    def test_greedy_agent_reaches_goal(self):
        """Test that greedy agent can reach goal."""
        env = SimpleGridworld(grid_size=5)
        agent = GreedyAgent()

        obs = env.reset()
        done = False
        steps = 0

        while not done and steps < 20:
            action = agent.select_action(obs)
            obs, reward, done = env.step(action)
            steps += 1

        # Greedy agent should reach goal in 8 steps (4 down + 4 right)
        self.assertTrue(done)
        self.assertEqual(env.agent_pos, env.goal_pos)
        self.assertLessEqual(steps, 10)  # Should be efficient

    def test_observation_normalized(self):
        """Test that observations are normalized to [0, 1]."""
        obs = self.env.reset()

        for value in obs:
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)


if __name__ == '__main__':
    unittest.main()
