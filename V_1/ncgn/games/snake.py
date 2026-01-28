"""
NCGN v6.0 Snake Game - 2D Grid Environment
"""

import random
from typing import Dict, List, Tuple, Optional
from .interface import GameInterface, SensoryInput, Direction

class SnakeGame(GameInterface):
    """
    Classic Snake game on a grid.
    """
    
    ACTION_UP = "action_up"
    ACTION_DOWN = "action_down"
    ACTION_LEFT = "action_left"
    ACTION_RIGHT = "action_right"
    
    def __init__(self, width: int = 8, height: int = 8):
        self.width = width
        self.height = height
        
        # Game state
        self.snake: List[Tuple[int, int]] = []
        self.food: Tuple[int, int] = (0, 0)
        self.score: int = 0
        self.steps: int = 0
        self.max_steps: int = 200
        
        # Stats
        self.episode_count = 0
        self.success_count = 0  # Reaching certain score
        
        self.reset()
    
    def reset(self) -> SensoryInput:
        self.snake = [(self.width // 2, self.height // 2)]
        self._place_food()
        self.score = 0
        self.steps = 0
        self.episode_count += 1
        return self.get_sensory()
    
    def _place_food(self):
        while True:
            pos = (random.randint(0, self.width-1), random.randint(0, self.height-1))
            if pos not in self.snake:
                self.food = pos
                break
    
    def step(self, action: str) -> Tuple[SensoryInput, float, bool]:
        self.steps += 1
        head = self.snake[0]
        dx, dy = 0, 0
        
        if action == self.ACTION_UP: dy = -1
        elif action == self.ACTION_DOWN: dy = 1
        elif action == self.ACTION_LEFT: dx = -1
        elif action == self.ACTION_RIGHT: dx = 1
        
        new_head = (head[0] + dx, head[1] + dy)
        reward = -0.01  # Step penalty
        done = False
        
        # Check collision with walls
        if (new_head[0] < 0 or new_head[0] >= self.width or
            new_head[1] < 0 or new_head[1] >= self.height):
            reward = -1.0
            done = True
        # Check collision with self
        elif new_head in self.snake[:-1]:
            reward = -1.0
            done = True
        else:
            # Move snake
            self.snake.insert(0, new_head)
            
            # Check food
            if new_head == self.food:
                reward = 1.0
                self.score += 1
                self._place_food()
                if self.score >= 10:  # arbitrary win condition
                    self.success_count += 1
            else:
                self.snake.pop()
        
        if self.steps >= self.max_steps:
            done = True
            
        return self.get_sensory(), reward, done
    
    def get_sensory(self) -> SensoryInput:
        vision = {}
        head = self.snake[0]
        
        # Check 4 directions
        dirs = {
            Direction.UP: (0, -1),
            Direction.DOWN: (0, 1),
            Direction.LEFT: (-1, 0),
            Direction.RIGHT: (1, 0)
        }
        
        for d, (dx, dy) in dirs.items():
            pos = (head[0] + dx, head[1] + dy)
            obj = "empty"
            if (pos[0] < 0 or pos[0] >= self.width or 
                pos[1] < 0 or pos[1] >= self.height):
                obj = "wall"
            elif pos in self.snake:
                obj = "body"
            elif pos == self.food:
                obj = "food"
            vision[d] = obj
            
        return SensoryInput(
            vision=vision,
            proprioception={"score": self.score},
            recent_action=None,
            reward=0.0
        )
            
    def get_possible_actions(self) -> List[str]:
        return [self.ACTION_UP, self.ACTION_DOWN, self.ACTION_LEFT, self.ACTION_RIGHT]

    def render(self) -> str:
        grid = [['.' for _ in range(self.width)] for _ in range(self.height)]
        for x, y in self.snake:
            grid[y][x] = 'S'
        grid[self.snake[0][1]][self.snake[0][0]] = 'H'
        grid[self.food[1]][self.food[0]] = 'F'
        
        return "\n".join("".join(row) for row in grid) + f"\nScore: {self.score}"
        
    def get_stats(self) -> Dict:
        return {
            "episode_count": self.episode_count,
            "success_count": self.success_count,
            "high_score": self.score
        }
