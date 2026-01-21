"""
NCGN v6.0 Snake Game - 2D Grid Learning Environment

A classic Snake game for testing spatial reasoning and sequential decisions:
- Snake starts at center with length 1
- Food spawns randomly
- Actions: up, down, left, right
- Reward: +1 for food, -1 for death, small negative per step
- Game over: hit wall or self

This tests:
- Spatial navigation (2D instead of 1D)
- Longer-term planning (longer snake = harder to avoid)
- Dynamic environment (food moves when eaten)
"""

import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from .interface import GameInterface, SensoryInput, Direction


@dataclass
class Position:
    """2D position on the grid."""
    x: int
    y: int
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def copy(self):
        return Position(self.x, self.y)


class SnakeGame(GameInterface):
    """
    Classic Snake game environment for NCGN learning.
    
    Grid layout (10x10 default):
    ┌──────────────────┐
    │                  │
    │        F         │  F = Food
    │                  │
    │      █ S         │  S = Snake head, █ = body
    │                  │
    └──────────────────┘
    
    This tests:
    - Multi-directional navigation (4 actions)
    - Obstacle avoidance (self and walls)
    - Dynamic goals (moving food target)
    - Sequential credit assignment (delayed rewards)
    """
    
    # Action node IDs
    ACTION_UP = "action_move_up"
    ACTION_DOWN = "action_move_down"
    ACTION_LEFT = "action_move_left"
    ACTION_RIGHT = "action_move_right"
    
    # Direction vectors
    DIRECTIONS = {
        ACTION_UP: (0, -1),
        ACTION_DOWN: (0, 1),
        ACTION_LEFT: (-1, 0),
        ACTION_RIGHT: (1, 0)
    }

    def __init__(
        self,
        width: int = 10,
        height: int = 10,
        step_penalty: float = -0.01,
        food_reward: float = 1.0,
        death_penalty: float = -1.0,
        max_steps_without_food: int = 50
    ):
        self.width = width
        self.height = height
        self.step_penalty = step_penalty
        self.food_reward = food_reward
        self.death_penalty = death_penalty
        self.max_steps_without_food = max_steps_without_food
        
        # State
        self.snake: List[Position] = []
        self.food: Optional[Position] = None
        self.direction: str = self.ACTION_RIGHT
        self.steps: int = 0
        self.steps_since_food: int = 0
        self.score: int = 0
        
        # Statistics
        self.episode_count: int = 0
        self.total_food_eaten: int = 0
        self.max_score: int = 0
        
        self.reset()
    
    def reset(self) -> SensoryInput:
        """Reset to initial state."""
        # Start snake at center with length 1
        center_x = self.width // 2
        center_y = self.height // 2
        self.snake = [Position(center_x, center_y)]
        
        # Spawn food
        self._spawn_food()
        
        # Reset state
        self.direction = self.ACTION_RIGHT
        self.steps = 0
        self.steps_since_food = 0
        self.score = 0
        
        self.episode_count += 1
        
        return self.get_sensory()
    
    def _spawn_food(self):
        """Spawn food at random empty position."""
        empty_positions = []
        for x in range(self.width):
            for y in range(self.height):
                pos = Position(x, y)
                if pos not in self.snake:
                    empty_positions.append(pos)
        
        if empty_positions:
            self.food = random.choice(empty_positions)
        else:
            self.food = None  # Snake fills entire grid (win!)
    
    def step(self, action: str) -> Tuple[SensoryInput, float, bool]:
        """
        Take a step in the Snake game.
        
        Returns:
            (observation, reward, done)
        """
        self.steps += 1
        self.steps_since_food += 1
        reward = self.step_penalty
        done = False
        
        # Update direction (can't reverse)
        if action in self.DIRECTIONS:
            opposite = {
                self.ACTION_UP: self.ACTION_DOWN,
                self.ACTION_DOWN: self.ACTION_UP,
                self.ACTION_LEFT: self.ACTION_RIGHT,
                self.ACTION_RIGHT: self.ACTION_LEFT
            }
            if len(self.snake) == 1 or action != opposite.get(self.direction):
                self.direction = action
        
        # Move snake
        dx, dy = self.DIRECTIONS[self.direction]
        head = self.snake[0]
        new_head = Position(head.x + dx, head.y + dy)
        
        # Check wall collision
        if (new_head.x < 0 or new_head.x >= self.width or
            new_head.y < 0 or new_head.y >= self.height):
            reward += self.death_penalty
            done = True
            return self.get_sensory(), reward, done
        
        # Check self collision
        if new_head in self.snake[:-1]:
            reward += self.death_penalty
            done = True
            return self.get_sensory(), reward, done
        
        # Move snake
        self.snake.insert(0, new_head)
        
        # Check food
        if self.food and new_head == self.food:
            reward += self.food_reward
            self.score += 1
            self.total_food_eaten += 1
            self.steps_since_food = 0
            self._spawn_food()
            
            if self.score > self.max_score:
                self.max_score = self.score
        else:
            # Remove tail (didn't eat)
            self.snake.pop()
        
        # Check starvation (prevents infinite wandering)
        if self.steps_since_food >= self.max_steps_without_food:
            done = True
        
        # Win condition: snake fills grid
        if self.food is None:
            reward += self.food_reward * 10  # Big bonus for winning
            done = True
        
        return self.get_sensory(), reward, done
    
    def get_sensory(self) -> SensoryInput:
        """Get current egocentric observation."""
        if not self.snake:
            return SensoryInput({}, {}, None, 0.0)
        
        head = self.snake[0]
        vision = {}
        
        # Look in each direction relative to current heading
        # For simplicity, we use absolute directions
        
        # What's ahead (in current direction)
        dx, dy = self.DIRECTIONS[self.direction]
        ahead_pos = Position(head.x + dx, head.y + dy)
        vision[Direction.AHEAD] = self._classify_position(ahead_pos)
        
        # What's behind
        behind_pos = Position(head.x - dx, head.y - dy)
        vision[Direction.BEHIND] = self._classify_position(behind_pos)
        
        # What's left (rotate direction 90 degrees counter-clockwise)
        left_dx, left_dy = dy, -dx
        left_pos = Position(head.x + left_dx, head.y + left_dy)
        vision[Direction.LEFT] = self._classify_position(left_pos)
        
        # What's right (rotate direction 90 degrees clockwise)
        right_dx, right_dy = -dy, dx
        right_pos = Position(head.x + right_dx, head.y + right_dy)
        vision[Direction.RIGHT] = self._classify_position(right_pos)
        
        # Proprioception
        food_dx = 0
        food_dy = 0
        if self.food:
            food_dx = (self.food.x - head.x) / self.width
            food_dy = (self.food.y - head.y) / self.height
        
        proprio = {
            "food_direction_x": food_dx,
            "food_direction_y": food_dy,
            "snake_length": len(self.snake) / (self.width * self.height)
        }
        
        return SensoryInput(
            vision=vision,
            proprioception=proprio,
            recent_action=self.direction,
            reward=0.0
        )
    
    def _classify_position(self, pos: Position) -> str:
        """Classify what's at a position."""
        # Wall
        if pos.x < 0 or pos.x >= self.width or pos.y < 0 or pos.y >= self.height:
            return "wall"
        
        # Food
        if self.food and pos == self.food:
            return "food"
        
        # Snake body
        if pos in self.snake:
            return "danger"  # Hitting body = death
        
        return "empty"
    
    def get_possible_actions(self) -> List[str]:
        """Get list of possible actions."""
        return [self.ACTION_UP, self.ACTION_DOWN, self.ACTION_LEFT, self.ACTION_RIGHT]
    
    def render(self) -> str:
        """Render game as ASCII grid."""
        # Create grid
        grid = [["." for _ in range(self.width)] for _ in range(self.height)]
        
        # Draw snake
        for i, pos in enumerate(self.snake):
            if i == 0:
                grid[pos.y][pos.x] = "@"  # Head
            else:
                grid[pos.y][pos.x] = "█"  # Body
        
        # Draw food
        if self.food:
            grid[self.food.y][self.food.x] = "*"
        
        # Build string
        lines = ["+" + "-" * self.width + "+"]
        for row in grid:
            lines.append("|" + "".join(row) + "|")
        lines.append("+" + "-" * self.width + "+")
        lines.append(f"Score: {self.score} | Steps: {self.steps}")
        
        return "\n".join(lines)
    
    def render_compact(self) -> str:
        """Compact single-line render for dashboard."""
        return f"Snake: Score={self.score} Len={len(self.snake)} Steps={self.steps}"
    
    def get_stats(self) -> Dict:
        """Get game statistics."""
        return {
            "episode_count": self.episode_count,
            "total_food_eaten": self.total_food_eaten,
            "max_score": self.max_score,
            "current_score": self.score,
            "current_length": len(self.snake),
            "current_steps": self.steps,
            "success_rate": self.total_food_eaten / max(1, self.episode_count),
            "success_count": self.total_food_eaten  # For compatibility with corridor
        }


def run_snake_demo():
    """Demo of the snake game with random actions."""
    game = SnakeGame(width=8, height=8)
    
    print("Snake Game Demo")
    print("=" * 40)
    
    obs = game.reset()
    print(game.render())
    print()
    
    actions = game.get_possible_actions()
    
    for step in range(20):
        action = random.choice(actions)
        obs, reward, done = game.step(action)
        
        print(f"Step {step + 1}: {action.split('_')[-1]} -> Reward: {reward:.2f}")
        
        if done:
            print("\nGame Over!")
            print(game.render())
            print(f"Final Stats: {game.get_stats()}")
            break
    else:
        print(game.render())


if __name__ == "__main__":
    run_snake_demo()
