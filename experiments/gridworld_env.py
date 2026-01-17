"""
Simple Gridworld Environment for Phase 5 Testing

A minimal mock environment for testing goal-directed behavior,
intrinsic motivation, and self-modeling without external dependencies.
"""

from typing import Tuple, List, Optional
import random


class SimpleGridworld:
    """
    Minimal 5x5 gridworld for Phase 5 demos (Phase 5).

    Agent can move in 4 directions, receives reward based on distance to goal.
    Episodes terminate when agent reaches goal or max steps exceeded.
    """

    def __init__(self, grid_size: int = 5, max_steps: int = 50):
        """
        Initialize gridworld.

        Args:
            grid_size: Size of square grid (default 5x5)
            max_steps: Maximum steps per episode
        """
        self.grid_size = grid_size
        self.max_steps = max_steps

        # Agent and goal positions
        self.agent_pos = (0, 0)
        self.goal_pos = (grid_size - 1, grid_size - 1)

        # Episode tracking
        self.steps = 0
        self.done = False
        self.total_reward = 0.0

    def reset(self) -> List[float]:
        """
        Reset environment to initial state.

        Returns:
            Initial observation (flattened state representation)
        """
        self.agent_pos = (0, 0)
        self.steps = 0
        self.done = False
        self.total_reward = 0.0

        return self._get_observation()

    def step(self, action: int) -> Tuple[List[float], float, bool]:
        """
        Take an action in the environment.

        Actions:
            0: Up (decrease row)
            1: Down (increase row)
            2: Left (decrease col)
            3: Right (increase col)

        Args:
            action: Action to take [0-3]

        Returns:
            (observation, reward, done)
            - observation: New state vector
            - reward: Reward signal [0, 1]
            - done: Whether episode has terminated
        """
        if self.done:
            return self._get_observation(), 0.0, True

        # Get distance before move
        dist_before = self._manhattan_distance(self.agent_pos, self.goal_pos)

        # Update position based on action
        row, col = self.agent_pos

        if action == 0:  # Up
            row = max(0, row - 1)
        elif action == 1:  # Down
            row = min(self.grid_size - 1, row + 1)
        elif action == 2:  # Left
            col = max(0, col - 1)
        elif action == 3:  # Right
            col = min(self.grid_size - 1, col + 1)

        self.agent_pos = (row, col)
        self.steps += 1

        # Get distance after move
        dist_after = self._manhattan_distance(self.agent_pos, self.goal_pos)

        # Reward based on progress toward goal
        if self.agent_pos == self.goal_pos:
            reward = 1.0  # Large reward for reaching goal
            self.done = True
        elif dist_after < dist_before:
            reward = 0.1  # Small reward for getting closer
        elif dist_after > dist_before:
            reward = -0.05  # Small penalty for moving away
        else:
            reward = -0.01  # Small penalty for staying same distance

        # Check for timeout
        if self.steps >= self.max_steps:
            self.done = True

        self.total_reward += reward

        return self._get_observation(), reward, self.done

    def _get_observation(self) -> List[float]:
        """
        Get current observation as normalized vector.

        Returns:
            [agent_row, agent_col, goal_row, goal_col, distance] (normalized)
        """
        agent_row, agent_col = self.agent_pos
        goal_row, goal_col = self.goal_pos
        distance = self._manhattan_distance(self.agent_pos, self.goal_pos)

        # Normalize to [0, 1]
        obs = [
            agent_row / self.grid_size,
            agent_col / self.grid_size,
            goal_row / self.grid_size,
            goal_col / self.grid_size,
            distance / (2 * self.grid_size)  # Max distance is 2*grid_size
        ]

        return obs

    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Compute Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def render(self) -> str:
        """
        Render gridworld as ASCII art.

        Returns:
            String representation of grid
        """
        lines = []
        for row in range(self.grid_size):
            line = []
            for col in range(self.grid_size):
                if (row, col) == self.agent_pos:
                    line.append('A')
                elif (row, col) == self.goal_pos:
                    line.append('G')
                else:
                    line.append('.')
            lines.append(' '.join(line))

        lines.append(f"Steps: {self.steps}/{self.max_steps}, Reward: {self.total_reward:.2f}")
        return '\n'.join(lines)

    def get_statistics(self) -> dict:
        """Get environment statistics."""
        return {
            "steps": self.steps,
            "total_reward": self.total_reward,
            "done": self.done,
            "agent_pos": self.agent_pos,
            "goal_pos": self.goal_pos,
            "distance_to_goal": self._manhattan_distance(self.agent_pos, self.goal_pos),
        }


class RandomAgent:
    """Random agent for baseline comparison."""

    def __init__(self, num_actions: int = 4):
        """Initialize random agent."""
        self.num_actions = num_actions

    def select_action(self, observation: List[float]) -> int:
        """Select random action."""
        return random.randint(0, self.num_actions - 1)


class GreedyAgent:
    """Greedy agent that always moves toward goal."""

    def select_action(self, observation: List[float]) -> int:
        """
        Select action greedily based on observation.

        Args:
            observation: [agent_row, agent_col, goal_row, goal_col, distance]

        Returns:
            Action that moves toward goal
        """
        agent_row, agent_col, goal_row, goal_col, _ = observation

        # Denormalize (assume grid_size = 5 for now)
        grid_size = 5
        agent_row = int(agent_row * grid_size)
        agent_col = int(agent_col * grid_size)
        goal_row = int(goal_row * grid_size)
        goal_col = int(goal_col * grid_size)

        # Move toward goal (row first, then col)
        if agent_row < goal_row:
            return 1  # Down
        elif agent_row > goal_row:
            return 0  # Up
        elif agent_col < goal_col:
            return 3  # Right
        elif agent_col > goal_col:
            return 2  # Left
        else:
            return 0  # At goal, any action
