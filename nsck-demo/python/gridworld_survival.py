"""
NSCK GridWorld Survival Game Environment

A complex game environment for demonstrating advanced learning capabilities:
- Multiple resources to manage (food, water, energy)
- Dynamic enemies with patrol patterns
- Strategic decisions required (risk vs reward)
- Multiple objectives simultaneously
- Environmental hazards

This game is more complex than Snake/Maze/Pong to demonstrate:
1. Multi-objective decision making
2. Long-term planning and strategy
3. Risk assessment
4. Resource management
5. Adaptation to dynamic environments
"""

import random
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Set
from enum import Enum


class EntityType(Enum):
    """Types of entities in the gridworld."""
    EMPTY = 0
    WALL = 1
    PLAYER = 2
    FOOD = 3
    WATER = 4
    ENEMY = 5
    HAZARD = 6
    TREASURE = 7
    EXIT = 8


class Direction(Enum):
    """Movement directions."""
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)
    STAY = (0, 0)


@dataclass
class Enemy:
    """Enemy entity with patrol behavior."""
    position: Tuple[int, int]
    patrol_points: List[Tuple[int, int]]
    current_target_idx: int = 0
    damage: float = 0.3
    
    def get_next_position(self, grid_width: int, grid_height: int) -> Tuple[int, int]:
        """Calculate next position based on patrol pattern."""
        if not self.patrol_points:
            # Random movement if no patrol points
            dy, dx = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)])
            new_y = max(0, min(grid_height - 1, self.position[0] + dy))
            new_x = max(0, min(grid_width - 1, self.position[1] + dx))
            return (new_y, new_x)
        
        # Move towards current patrol target
        target = self.patrol_points[self.current_target_idx]
        dy = np.sign(target[0] - self.position[0])
        dx = np.sign(target[1] - self.position[1])
        
        new_pos = (self.position[0] + dy, self.position[1] + dx)
        
        # If reached target, move to next patrol point
        if new_pos == target:
            self.current_target_idx = (self.current_target_idx + 1) % len(self.patrol_points)
        
        return new_pos


@dataclass
class PlayerStats:
    """Player statistics and resources."""
    energy: float = 1.0  # 0-1, drains over time
    health: float = 1.0  # 0-1, damage from enemies/hazards
    hunger: float = 0.0  # 0-1, increases over time
    thirst: float = 0.0  # 0-1, increases over time
    score: int = 0
    treasures_collected: int = 0
    
    def is_alive(self) -> bool:
        """Check if player is still alive."""
        return self.health > 0 and self.energy > 0


@dataclass
class GridWorldState:
    """Complete state of the GridWorld Survival game."""
    grid: np.ndarray
    player_pos: Tuple[int, int]
    exit_pos: Tuple[int, int]
    enemies: List[Enemy]
    food_positions: Set[Tuple[int, int]]
    water_positions: Set[Tuple[int, int]]
    hazard_positions: Set[Tuple[int, int]]
    treasure_positions: Set[Tuple[int, int]]
    stats: PlayerStats
    steps: int = 0
    max_steps: int = 500
    width: int = 20
    height: int = 20
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for cognitive engine."""
        # Calculate distances to nearest objectives
        player_y, player_x = self.player_pos
        
        def nearest_distance(positions: Set[Tuple[int, int]]) -> float:
            if not positions:
                return float('inf')
            return min(abs(py - player_y) + abs(px - player_x) 
                      for py, px in positions)
        
        nearest_food = nearest_distance(self.food_positions)
        nearest_water = nearest_distance(self.water_positions)
        nearest_enemy = min((abs(e.position[0] - player_y) + abs(e.position[1] - player_x) 
                           for e in self.enemies), default=float('inf'))
        
        exit_dist = abs(self.exit_pos[0] - player_y) + abs(self.exit_pos[1] - player_x)
        
        # Generate predicates for rule learning
        predicates = []
        
        # Resource state predicates
        if self.stats.hunger > 0.7:
            predicates.append("CRITICAL_HUNGER")
        elif self.stats.hunger > 0.4:
            predicates.append("HUNGRY")
            
        if self.stats.thirst > 0.7:
            predicates.append("CRITICAL_THIRST")
        elif self.stats.thirst > 0.4:
            predicates.append("THIRSTY")
            
        if self.stats.energy < 0.3:
            predicates.append("LOW_ENERGY")
        elif self.stats.energy < 0.5:
            predicates.append("MODERATE_ENERGY")
            
        if self.stats.health < 0.5:
            predicates.append("LOW_HEALTH")
            
        # Proximity predicates
        if nearest_enemy < 3:
            predicates.append("ENEMY_NEARBY")
            if nearest_enemy < 2:
                predicates.append("ENEMY_VERY_CLOSE")
                
        if nearest_food < 5:
            predicates.append("FOOD_NEARBY")
            if nearest_food < 2:
                predicates.append("FOOD_CLOSE")
                
        if nearest_water < 5:
            predicates.append("WATER_NEARBY")
            if nearest_water < 2:
                predicates.append("WATER_CLOSE")
                
        if exit_dist < 5:
            predicates.append("EXIT_NEARBY")
            
        # Direction predicates (where are resources/threats?)
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                check_y, check_x = player_y + dy, player_x + dx
                if 0 <= check_y < self.height and 0 <= check_x < self.width:
                    cell = self.grid[check_y, check_x]
                    if cell == EntityType.FOOD.value:
                        predicates.append(f"FOOD_DIRECTION_{dy}_{dx}")
                    elif cell == EntityType.WATER.value:
                        predicates.append(f"WATER_DIRECTION_{dy}_{dx}")
                    elif cell == EntityType.ENEMY.value:
                        predicates.append(f"ENEMY_DIRECTION_{dy}_{dx}")
                    elif cell == EntityType.HAZARD.value:
                        predicates.append(f"HAZARD_DIRECTION_{dy}_{dx}")
                    elif cell == EntityType.WALL.value:
                        predicates.append(f"WALL_DIRECTION_{dy}_{dx}")
        
        return {
            "player_pos": self.player_pos,
            "exit_pos": self.exit_pos,
            "head": self.player_pos,  # Alias for compatibility
            "target": self.exit_pos,  # Alias for compatibility
            "width": self.width,
            "height": self.height,
            "steps": self.steps,
            "max_steps": self.max_steps,
            "predicates": predicates,
            "stats": {
                "energy": self.stats.energy,
                "health": self.stats.health,
                "hunger": self.stats.hunger,
                "thirst": self.stats.thirst,
                "score": self.stats.score,
                "treasures": self.stats.treasures_collected,
            },
            "nearest_food": nearest_food,
            "nearest_water": nearest_water,
            "nearest_enemy": nearest_enemy,
            "exit_distance": exit_dist,
            "done": not self.stats.is_alive() or self.steps >= self.max_steps,
        }


class GridWorldSurvival:
    """
    GridWorld Survival Game - A complex environment for demonstrating
    advanced cognitive capabilities.
    
    Game Mechanics:
    - Player must survive by managing energy, health, hunger, and thirst
    - Collect food to reduce hunger
    - Collect water to reduce thirst
    - Avoid enemies (patrol patterns)
    - Avoid hazards (static damage zones)
    - Collect treasures for bonus points
    - Reach the exit to win
    - Energy drains over time (more when moving)
    - Hunger and thirst increase over time
    
    This requires strategic decision-making:
    - When to prioritize survival (food/water) vs progress (exit)
    - Risk assessment (going near enemies for resources)
    - Resource management (conserve energy vs move faster)
    - Multi-objective optimization
    """
    
    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        num_enemies: int = 3,
        num_food: int = 8,
        num_water: int = 6,
        num_hazards: int = 5,
        num_treasures: int = 3,
        max_steps: int = 500,
    ):
        """
        Initialize GridWorld Survival game.
        
        Args:
            width: Grid width
            height: Grid height
            num_enemies: Number of enemy entities
            num_food: Number of food items
            num_water: Number of water sources
            num_hazards: Number of hazard zones
            num_treasures: Number of treasure items
            max_steps: Maximum steps before game over
        """
        self.width = width
        self.height = height
        self.num_enemies = num_enemies
        self.num_food = num_food
        self.num_water = num_water
        self.num_hazards = num_hazards
        self.num_treasures = num_treasures
        self.max_steps = max_steps
        self.state: Optional[GridWorldState] = None
        self.reset()
    
    def reset(self) -> GridWorldState:
        """Reset game to initial state with random layout."""
        # Initialize empty grid
        grid = np.full((self.height, self.width), EntityType.EMPTY.value, dtype=np.int32)
        
        # Add border walls
        grid[0, :] = EntityType.WALL.value
        grid[-1, :] = EntityType.WALL.value
        grid[:, 0] = EntityType.WALL.value
        grid[:, -1] = EntityType.WALL.value
        
        # Add some internal walls for complexity
        num_internal_walls = (self.width * self.height) // 20
        for _ in range(num_internal_walls):
            y = random.randint(2, self.height - 3)
            x = random.randint(2, self.width - 3)
            # Create small wall segments
            for dy, dx in [(0, 0), (0, 1), (1, 0)]:
                if 0 < y + dy < self.height - 1 and 0 < x + dx < self.width - 1:
                    grid[y + dy, x + dx] = EntityType.WALL.value
        
        # Place player (bottom-left area)
        player_pos = self._find_empty_position(grid, region="bottom-left")
        grid[player_pos] = EntityType.PLAYER.value
        
        # Place exit (top-right area)
        exit_pos = self._find_empty_position(grid, region="top-right")
        grid[exit_pos] = EntityType.EXIT.value
        
        # Place hazards
        hazard_positions = set()
        for _ in range(self.num_hazards):
            pos = self._find_empty_position(grid)
            grid[pos] = EntityType.HAZARD.value
            hazard_positions.add(pos)
        
        # Place enemies with patrol patterns
        enemies = []
        for _ in range(self.num_enemies):
            pos = self._find_empty_position(grid)
            grid[pos] = EntityType.ENEMY.value
            
            # Create patrol pattern (3-5 points)
            patrol_points = [pos]
            for _ in range(random.randint(2, 4)):
                patrol_y = max(1, min(self.height - 2, pos[0] + random.randint(-5, 5)))
                patrol_x = max(1, min(self.width - 2, pos[1] + random.randint(-5, 5)))
                patrol_points.append((patrol_y, patrol_x))
            
            enemies.append(Enemy(position=pos, patrol_points=patrol_points))
        
        # Place food
        food_positions = set()
        for _ in range(self.num_food):
            pos = self._find_empty_position(grid)
            grid[pos] = EntityType.FOOD.value
            food_positions.add(pos)
        
        # Place water
        water_positions = set()
        for _ in range(self.num_water):
            pos = self._find_empty_position(grid)
            grid[pos] = EntityType.WATER.value
            water_positions.add(pos)
        
        # Place treasures
        treasure_positions = set()
        for _ in range(self.num_treasures):
            pos = self._find_empty_position(grid)
            grid[pos] = EntityType.TREASURE.value
            treasure_positions.add(pos)
        
        # Initialize state
        self.state = GridWorldState(
            grid=grid,
            player_pos=player_pos,
            exit_pos=exit_pos,
            enemies=enemies,
            food_positions=food_positions,
            water_positions=water_positions,
            hazard_positions=hazard_positions,
            treasure_positions=treasure_positions,
            stats=PlayerStats(),
            steps=0,
            max_steps=self.max_steps,
            width=self.width,
            height=self.height,
        )
        
        return self.state
    
    def _find_empty_position(
        self, 
        grid: np.ndarray, 
        region: Optional[str] = None
    ) -> Tuple[int, int]:
        """Find an empty position in the grid, optionally in a specific region."""
        max_attempts = 1000
        for _ in range(max_attempts):
            if region == "bottom-left":
                y = random.randint(self.height // 2, self.height - 2)
                x = random.randint(1, self.width // 2)
            elif region == "top-right":
                y = random.randint(1, self.height // 2)
                x = random.randint(self.width // 2, self.width - 2)
            else:
                y = random.randint(1, self.height - 2)
                x = random.randint(1, self.width - 2)
            
            if grid[y, x] == EntityType.EMPTY.value:
                return (y, x)
        
        # Fallback: find any empty position
        empty_positions = np.argwhere(grid == EntityType.EMPTY.value)
        if len(empty_positions) > 0:
            idx = random.randint(0, len(empty_positions) - 1)
            return tuple(empty_positions[idx])
        
        raise RuntimeError("No empty positions available in grid")
    
    def step(self, action: str) -> Tuple[GridWorldState, float, bool, Dict[str, Any]]:
        """
        Execute one step of the game.
        
        Args:
            action: Action to take (ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_STAY)
        
        Returns:
            Tuple of (new_state, reward, done, info)
        """
        if self.state is None:
            raise RuntimeError("Game not initialized. Call reset() first.")
        
        # Parse action
        action_map = {
            "ACTION_UP": Direction.UP.value,
            "ACTION_DOWN": Direction.DOWN.value,
            "ACTION_LEFT": Direction.LEFT.value,
            "ACTION_RIGHT": Direction.RIGHT.value,
            "ACTION_STAY": Direction.STAY.value,
        }
        
        dy, dx = action_map.get(action, Direction.STAY.value)
        
        # Calculate new position
        old_y, old_x = self.state.player_pos
        new_y = old_y + dy
        new_x = old_x + dx
        
        # Initialize reward
        reward = -0.01  # Small negative reward per step (encourages efficiency)
        info = {"reason": "step"}
        
        # Check boundaries and walls
        if (new_y < 0 or new_y >= self.height or 
            new_x < 0 or new_x >= self.width or
            self.state.grid[new_y, new_x] == EntityType.WALL.value):
            # Invalid move - stay in place
            new_y, new_x = old_y, old_x
            reward -= 0.05  # Penalty for invalid move
            info["reason"] = "invalid_move"
        else:
            # Valid move - update position
            cell_type = self.state.grid[new_y, new_x]
            
            # Handle interactions
            if cell_type == EntityType.FOOD.value:
                self.state.stats.hunger = max(0, self.state.stats.hunger - 0.4)
                self.state.stats.score += 10
                reward += 1.0
                self.state.food_positions.discard((new_y, new_x))
                info["reason"] = "ate_food"
            
            elif cell_type == EntityType.WATER.value:
                self.state.stats.thirst = max(0, self.state.stats.thirst - 0.4)
                self.state.stats.score += 10
                reward += 1.0
                self.state.water_positions.discard((new_y, new_x))
                info["reason"] = "drank_water"
            
            elif cell_type == EntityType.TREASURE.value:
                self.state.stats.treasures_collected += 1
                self.state.stats.score += 50
                reward += 5.0
                self.state.treasure_positions.discard((new_y, new_x))
                info["reason"] = "collected_treasure"
            
            elif cell_type == EntityType.HAZARD.value:
                self.state.stats.health = max(0, self.state.stats.health - 0.2)
                reward -= 2.0
                info["reason"] = "hit_hazard"
            
            elif cell_type == EntityType.EXIT.value:
                # Reached exit - big reward!
                self.state.stats.score += 100
                reward += 10.0
                info["reason"] = "reached_exit"
                info["success"] = True
            
            # Update grid
            self.state.grid[old_y, old_x] = EntityType.EMPTY.value
            self.state.grid[new_y, new_x] = EntityType.PLAYER.value
            self.state.player_pos = (new_y, new_x)
        
        # Move enemies
        for enemy in self.state.enemies:
            old_enemy_pos = enemy.position
            new_enemy_pos = enemy.get_next_position(self.width, self.height)
            
            # Check if enemy collides with player
            if new_enemy_pos == self.state.player_pos:
                self.state.stats.health = max(0, self.state.stats.health - enemy.damage)
                reward -= 3.0
                info["reason"] = "hit_by_enemy"
                # Enemy doesn't move into player
                new_enemy_pos = old_enemy_pos
            elif self.state.grid[new_enemy_pos] == EntityType.WALL.value:
                # Enemy can't move through walls
                new_enemy_pos = old_enemy_pos
            else:
                # Update enemy position on grid
                self.state.grid[old_enemy_pos] = EntityType.EMPTY.value
                self.state.grid[new_enemy_pos] = EntityType.ENEMY.value
                enemy.position = new_enemy_pos
        
        # Update player stats
        self.state.stats.energy -= 0.002  # Energy drains per step
        if action != "ACTION_STAY":
            self.state.stats.energy -= 0.001  # Extra drain for movement
        
        self.state.stats.hunger += 0.003  # Hunger increases
        self.state.stats.thirst += 0.004  # Thirst increases faster
        
        # Damage from critical states
        if self.state.stats.hunger >= 1.0:
            self.state.stats.health -= 0.01
            reward -= 0.1
        if self.state.stats.thirst >= 1.0:
            self.state.stats.health -= 0.015
            reward -= 0.15
        
        # Clamp values
        self.state.stats.energy = max(0, min(1, self.state.stats.energy))
        self.state.stats.health = max(0, min(1, self.state.stats.health))
        self.state.stats.hunger = max(0, min(1, self.state.stats.hunger))
        self.state.stats.thirst = max(0, min(1, self.state.stats.thirst))
        
        # Increment steps
        self.state.steps += 1
        
        # Check if game is done
        done = (not self.state.stats.is_alive() or 
                self.state.steps >= self.max_steps or
                info.get("success", False))
        
        if not self.state.stats.is_alive():
            reward -= 10.0  # Penalty for dying
            info["reason"] = "died"
        elif self.state.steps >= self.max_steps:
            reward -= 5.0  # Penalty for timeout
            info["reason"] = "timeout"
        
        return self.state, reward, done, info
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get current state as dictionary."""
        if self.state is None:
            raise RuntimeError("Game not initialized")
        return self.state.to_dict()
    
    def render_ascii(self) -> str:
        """Render game state as ASCII art."""
        if self.state is None:
            return "Game not initialized"
        
        # Entity symbols
        symbols = {
            EntityType.EMPTY.value: '.',
            EntityType.WALL.value: '#',
            EntityType.PLAYER.value: '@',
            EntityType.FOOD.value: 'F',
            EntityType.WATER.value: 'W',
            EntityType.ENEMY.value: 'E',
            EntityType.HAZARD.value: 'X',
            EntityType.TREASURE.value: 'T',
            EntityType.EXIT.value: '★',
        }
        
        lines = []
        for y in range(self.height):
            line = ''.join(symbols.get(self.state.grid[y, x], '?') 
                          for x in range(self.width))
            lines.append(line)
        
        # Add stats
        stats_lines = [
            f"Steps: {self.state.steps}/{self.max_steps}  Score: {self.state.stats.score}",
            f"Energy: {self.state.stats.energy:.2f}  Health: {self.state.stats.health:.2f}",
            f"Hunger: {self.state.stats.hunger:.2f}  Thirst: {self.state.stats.thirst:.2f}",
            f"Treasures: {self.state.stats.treasures_collected}",
        ]
        
        return '\n'.join(lines) + '\n\n' + '\n'.join(stats_lines)


def create_gridworld_causal_graph():
    """
    Create causal graph for GridWorld Survival game.
    
    Defines cause-effect relationships in the game:
    - Actions affect player position
    - Position affects resource collection
    - Resources affect player stats
    - Stats affect survival
    """
    from causal_reasoning import CausalGraph
    
    graph = CausalGraph()
    
    # Action -> Position
    graph.add_edge("ACTION_UP", "PLAYER_MOVES_UP")
    graph.add_edge("ACTION_DOWN", "PLAYER_MOVES_DOWN")
    graph.add_edge("ACTION_LEFT", "PLAYER_MOVES_LEFT")
    graph.add_edge("ACTION_RIGHT", "PLAYER_MOVES_RIGHT")
    
    # Position -> Resource Collection
    graph.add_edge("PLAYER_MOVES_UP", "POSITION_CHANGED")
    graph.add_edge("PLAYER_MOVES_DOWN", "POSITION_CHANGED")
    graph.add_edge("PLAYER_MOVES_LEFT", "POSITION_CHANGED")
    graph.add_edge("PLAYER_MOVES_RIGHT", "POSITION_CHANGED")
    
    graph.add_edge("POSITION_CHANGED", "MAY_COLLECT_RESOURCE")
    graph.add_edge("FOOD_CLOSE", "MAY_COLLECT_RESOURCE")
    graph.add_edge("WATER_CLOSE", "MAY_COLLECT_RESOURCE")
    
    # Resource Collection -> Stats
    graph.add_edge("MAY_COLLECT_RESOURCE", "HUNGER_DECREASES")
    graph.add_edge("MAY_COLLECT_RESOURCE", "THIRST_DECREASES")
    graph.add_edge("HUNGER_DECREASES", "HEALTH_INCREASES")
    graph.add_edge("THIRST_DECREASES", "HEALTH_INCREASES")
    
    # Time -> Resource Drain
    graph.add_edge("STEP_TAKEN", "ENERGY_DRAINS")
    graph.add_edge("STEP_TAKEN", "HUNGER_INCREASES")
    graph.add_edge("STEP_TAKEN", "THIRST_INCREASES")
    
    # Critical States -> Damage
    graph.add_edge("CRITICAL_HUNGER", "HEALTH_DECREASES")
    graph.add_edge("CRITICAL_THIRST", "HEALTH_DECREASES")
    graph.add_edge("ENEMY_VERY_CLOSE", "HEALTH_DECREASES")
    
    # Stats -> Survival
    graph.add_edge("HEALTH_DECREASES", "DEATH_RISK")
    graph.add_edge("ENERGY_DRAINS", "DEATH_RISK")
    
    return graph


if __name__ == "__main__":
    # Demo the game
    print("=== GridWorld Survival Demo ===\n")
    
    game = GridWorldSurvival(width=15, height=15, num_enemies=2)
    state = game.reset()
    
    print("Initial State:")
    print(game.render_ascii())
    print("\nPress Enter to take random actions...")
    
    done = False
    step_count = 0
    
    while not done and step_count < 50:
        input()  # Wait for Enter
        
        # Random action
        action = random.choice([
            "ACTION_UP", "ACTION_DOWN", "ACTION_LEFT", "ACTION_RIGHT", "ACTION_STAY"
        ])
        
        state, reward, done, info = game.step(action)
        step_count += 1
        
        print(f"\nAction: {action}, Reward: {reward:.2f}, Reason: {info['reason']}")
        print(game.render_ascii())
        
        if done:
            print(f"\n=== Game Over: {info['reason']} ===")
            print(f"Final Score: {state.stats.score}")
