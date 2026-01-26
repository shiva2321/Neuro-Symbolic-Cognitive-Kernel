"""
NCGN v6.0 Corridor Game - Simple 1D Learning Environment
"""

from typing import Dict, List, Tuple, Optional
from .interface import GameInterface, SensoryInput, Direction, ValenceEstimator


class CorridorGame(GameInterface):
    """
    1D corridor environment.
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
        self.max_steps: int = length * 3
        
        # Episode tracking
        self.episode_count: int = 0
        self.success_count: int = 0
    
    def reset(self) -> SensoryInput:
        self.position = 0
        self.steps = 0
        self.episode_count += 1
        return self.get_sensory()
    
    def step(self, action: str) -> Tuple[SensoryInput, float, bool]:
        self.steps += 1
        reward = self.step_penalty
        done = False
        
        if action == self.ACTION_RIGHT:
            if self.position < self.length - 1:
                self.position += 1
            else:
                reward += self.wall_penalty
        
        elif action == self.ACTION_LEFT:
            if self.position > 0:
                self.position -= 1
            else:
                reward += self.wall_penalty
        
        # Check goal
        if self.position == self.length - 1:
            reward += self.goal_reward
            done = True
            self.success_count += 1
        
        if self.steps >= self.max_steps:
            done = True
        
        return self.get_sensory(), reward, done
    
    def get_sensory(self) -> SensoryInput:
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
        
        distance_ratio = (self.length - 1 - self.position) / (self.length - 1)
        
        return SensoryInput(
            vision=vision,
            proprioception={"distance_to_goal": distance_ratio},
            recent_action=None,
            reward=0.0
        )
    
    def get_possible_actions(self) -> List[str]:
        return [self.ACTION_LEFT, self.ACTION_RIGHT]
    
    def render(self) -> str:
        corridor = ["-"] * self.length
        corridor[0] = "S"
        corridor[-1] = "G"
        corridor[self.position] = "A"
        return f"[{''.join(corridor)}] Step {self.steps}"
    
    def get_stats(self) -> Dict:
        return {
            "episode_count": self.episode_count,
            "success_count": self.success_count,
            "success_rate": (self.success_count / self.episode_count if self.episode_count > 0 else 0.0),
            "current_position": self.position,
            "current_steps": self.steps
        }
