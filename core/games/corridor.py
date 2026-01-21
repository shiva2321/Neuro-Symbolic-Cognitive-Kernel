"""
NCGN v6.0 Corridor Game - Simple 1D Learning Environment

A minimal environment for testing basic RL mechanics:
- Agent starts at position 0
- Goal is at position N
- Actions: move_left, move_right
- Reward: +1 at goal, small negative per step

This is the "Corridor Test" from the verification plan.
"""

from typing import Dict, List, Tuple, Optional
from .interface import GameInterface, SensoryInput, Direction, ValenceEstimator


class CorridorGame(GameInterface):
    """
    1D corridor environment for testing NCGN learning.
    
    The corridor is a simple line:
    [START] - - - - - [GOAL]
       0    1 2 3 4 5   6
    
    Agent must learn to consistently move right.
    
    This tests:
    - Basic credit assignment (reward at end)
    - Eligibility traces (delayed reward)
    - RPE calculation (expected vs actual)
    """
    
    # Action node IDs
    ACTION_LEFT = "action_move_left"
    ACTION_RIGHT = "action_move_right"
    
    def __init__(
        self,
        length: int = 7,
        step_penalty: float = -0.01,
        goal_reward: float = 1.0,
        wall_penalty: float = -0.1
    ):
        self.length = length
        self.step_penalty = step_penalty
        self.goal_reward = goal_reward
        self.wall_penalty = wall_penalty
        
        # State
        self.position: int = 0
        self.steps: int = 0
        self.max_steps: int = length * 3  # Timeout after too many steps
        
        # Episode tracking
        self.episode_count: int = 0
        self.success_count: int = 0
    
    def reset(self) -> SensoryInput:
        """Reset to starting position."""
        self.position = 0
        self.steps = 0
        self.episode_count += 1
        return self.get_sensory()
    
    def step(self, action: str) -> Tuple[SensoryInput, float, bool]:
        """
        Take a step in the corridor.
        
        Returns:
            (observation, reward, done)
        """
        self.steps += 1
        reward = self.step_penalty  # Small penalty for each step
        done = False
        
        if action == self.ACTION_RIGHT:
            if self.position < self.length - 1:
                self.position += 1
            else:
                reward += self.wall_penalty  # Hit right wall
        
        elif action == self.ACTION_LEFT:
            if self.position > 0:
                self.position -= 1
            else:
                reward += self.wall_penalty  # Hit left wall
        
        # Check goal
        if self.position == self.length - 1:
            reward += self.goal_reward
            done = True
            self.success_count += 1
        
        # Check timeout
        if self.steps >= self.max_steps:
            done = True
        
        return self.get_sensory(), reward, done
    
    def get_sensory(self) -> SensoryInput:
        """Get current observation."""
        vision = {}
        
        # What's ahead (right)
        if self.position == self.length - 1:
            vision[Direction.AHEAD] = "wall"
        elif self.position == self.length - 2:
            vision[Direction.AHEAD] = "goal"
        else:
            vision[Direction.AHEAD] = "empty"
        
        # What's behind (left)
        if self.position == 0:
            vision[Direction.BEHIND] = "wall"
        else:
            vision[Direction.BEHIND] = "empty"
        
        # Proprioception: how far from goal (normalized)
        distance_ratio = (self.length - 1 - self.position) / (self.length - 1)
        
        return SensoryInput(
            vision=vision,
            proprioception={"distance_to_goal": distance_ratio},
            recent_action=None,
            reward=0.0
        )
    
    def get_possible_actions(self) -> List[str]:
        """Get list of actions."""
        return [self.ACTION_LEFT, self.ACTION_RIGHT]
    
    def render(self) -> str:
        """Render corridor as ASCII."""
        corridor = ["-"] * self.length
        corridor[0] = "S"  # Start
        corridor[-1] = "G"  # Goal
        corridor[self.position] = "A"  # Agent
        
        return f"[{''.join(corridor)}] Step {self.steps}"
    
    def get_stats(self) -> Dict:
        """Get game statistics."""
        return {
            "episode_count": self.episode_count,
            "success_count": self.success_count,
            "success_rate": (
                self.success_count / self.episode_count 
                if self.episode_count > 0 else 0.0
            ),
            "current_position": self.position,
            "current_steps": self.steps
        }


def run_corridor_demo():
    """Simple demo of the corridor game."""
    game = CorridorGame(length=7)
    
    print("Corridor Demo")
    print("=" * 40)
    
    obs = game.reset()
    print(f"Start: {game.render()}")
    
    # Optimal policy: always go right
    actions = [CorridorGame.ACTION_RIGHT] * 10
    
    for action in actions:
        obs, reward, done = game.step(action)
        print(f"{game.render()} | Reward: {reward:.2f}")
        
        if done:
            print(f"\nEpisode complete! Stats: {game.get_stats()}")
            break


if __name__ == "__main__":
    run_corridor_demo()
