"""
Warehouse Robot Game - A Different Game Type

A puzzle/logistics game where a robot must collect packages and deliver them
to shipping zones while managing battery and avoiding obstacles.

This is VERY different from GridWorld Survival:
- GridWorld: Survival, resource management, dynamic enemies
- Warehouse: Logistics, planning, task completion

Yet the SAME agent can learn both!
"""

import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from universal_game_interface import (
    GameEnvironment, GameState, GameAction, GameResult,
    register_game, create_spatial_predicates
)


@dataclass
class Package:
    """A package to deliver."""
    id: int
    position: Tuple[int, int]
    destination: str  # "ZONE_A", "ZONE_B", etc.
    priority: int = 1  # Higher = more urgent
    collected: bool = False


@dataclass
class WarehouseRobotState:
    """Complete warehouse game state."""
    robot_pos: Tuple[int, int]
    battery: float  # 0-1
    packages_held: List[int]  # Package IDs
    packages: Dict[int, Package]  # All packages
    shipping_zones: Dict[str, List[Tuple[int, int]]]  # Zone -> positions
    obstacles: Set[Tuple[int, int]]
    charging_stations: Set[Tuple[int, int]]
    grid: np.ndarray
    width: int
    height: int
    steps: int
    max_steps: int
    score: int
    packages_delivered: int
    
    def to_game_state(self) -> GameState:
        """Convert to universal GameState."""
        # Generate predicates
        predicates = []
        
        # Battery state
        if self.battery < 0.2:
            predicates.append("CRITICAL_BATTERY")
        elif self.battery < 0.5:
            predicates.append("LOW_BATTERY")
        
        # Package holding
        if len(self.packages_held) > 0:
            predicates.append("CARRYING_PACKAGE")
            if len(self.packages_held) >= 3:
                predicates.append("FULL_CAPACITY")
        else:
            predicates.append("EMPTY_HANDS")
        
        # Proximity to important locations
        ry, rx = self.robot_pos
        
        # Find uncollected packages
        uncollected_packages = [p for p in self.packages.values() if not p.collected]
        if uncollected_packages:
            nearest_package_dist = min(
                abs(p.position[0] - ry) + abs(p.position[1] - rx)
                for p in uncollected_packages
            )
            if nearest_package_dist <= 2:
                predicates.append("PACKAGE_CLOSE")
            elif nearest_package_dist <= 5:
                predicates.append("PACKAGE_NEARBY")
        
        # Charging stations
        if self.charging_stations:
            nearest_charger_dist = min(
                abs(cy - ry) + abs(cx - rx)
                for cy, cx in self.charging_stations
            )
            if nearest_charger_dist <= 2:
                predicates.append("CHARGER_CLOSE")
            elif nearest_charger_dist <= 5:
                predicates.append("CHARGER_NEARBY")
        
        # Shipping zones (if carrying packages)
        if self.packages_held:
            for zone_name, zone_positions in self.shipping_zones.items():
                nearest_zone_dist = min(
                    abs(zy - ry) + abs(zx - rx)
                    for zy, zx in zone_positions
                )
                if nearest_zone_dist <= 2:
                    predicates.append(f"ZONE_{zone_name}_CLOSE")
                elif nearest_zone_dist <= 5:
                    predicates.append(f"ZONE_{zone_name}_NEARBY")
        
        # Task urgency
        if uncollected_packages:
            high_priority = [p for p in uncollected_packages if p.priority > 1]
            if high_priority:
                predicates.append("HIGH_PRIORITY_PACKAGES")
        
        # Spatial awareness
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                check_y, check_x = ry + dy, rx + dx
                if (check_y, check_x) in self.obstacles:
                    predicates.append(f"OBSTACLE_DIRECTION_{dy}_{dx}")
        
        # Features
        features = {
            'battery': self.battery,
            'packages_held': len(self.packages_held),
            'packages_remaining': len([p for p in self.packages.values() if not p.collected]),
            'packages_delivered': self.packages_delivered,
            'robot_x': rx,
            'robot_y': ry,
        }
        
        # Visual representation
        visual = self._render_ascii()
        
        return GameState(
            done=(self.steps >= self.max_steps or 
                  all(p.collected for p in self.packages.values())),
            score=float(self.score),
            steps=self.steps,
            predicates=predicates,
            features=features,
            visual_repr=visual
        )
    
    def _render_ascii(self) -> str:
        """Render warehouse as ASCII."""
        # Create visual grid
        visual = np.full((self.height, self.width), '.', dtype=str)
        
        # Walls/boundaries
        visual[0, :] = '#'
        visual[-1, :] = '#'
        visual[:, 0] = '#'
        visual[:, -1] = '#'
        
        # Obstacles
        for oy, ox in self.obstacles:
            visual[oy, ox] = 'X'
        
        # Charging stations
        for cy, cx in self.charging_stations:
            visual[cy, cx] = 'C'
        
        # Shipping zones
        for zone_name, positions in self.shipping_zones.items():
            letter = zone_name[-1]  # A, B, C, etc.
            for zy, zx in positions:
                visual[zy, zx] = letter
        
        # Uncollected packages
        for pkg in self.packages.values():
            if not pkg.collected:
                py, px = pkg.position
                priority_symbol = str(pkg.priority) if pkg.priority > 1 else 'P'
                visual[py, px] = priority_symbol
        
        # Robot
        ry, rx = self.robot_pos
        if len(self.packages_held) > 0:
            visual[ry, rx] = 'R'  # Robot with packages
        else:
            visual[ry, rx] = 'r'  # Empty robot
        
        # Convert to string
        lines = [''.join(row) for row in visual]
        
        # Add status
        status = [
            '',
            f"Steps: {self.steps}/{self.max_steps}  Score: {self.score}",
            f"Battery: {self.battery:.2f}  Carrying: {len(self.packages_held)} packages",
            f"Delivered: {self.packages_delivered}  Remaining: {len([p for p in self.packages.values() if not p.collected])}",
        ]
        
        return '\n'.join(lines + status)


@register_game("warehouse")
class WarehouseRobotGame(GameEnvironment):
    """
    Warehouse Robot Game - Logistics and planning.
    
    Objective: Collect packages and deliver them to correct shipping zones
    Challenges:
    - Battery management (charging vs delivery)
    - Route planning (efficiency)
    - Priority handling (urgent packages first)
    - Capacity limits (3 packages max)
    - Obstacles to navigate around
    
    This is completely different from GridWorld Survival, yet the same
    agent can learn both games!
    """
    
    def __init__(
        self,
        width: int = 15,
        height: int = 15,
        num_packages: int = 8,
        num_obstacles: int = 10,
        num_zones: int = 3,
        max_steps: int = 300,
    ):
        self.width = width
        self.height = height
        self.num_packages = num_packages
        self.num_obstacles = num_obstacles
        self.num_zones = num_zones
        self.max_steps_value = max_steps
        self.state: Optional[WarehouseRobotState] = None
        self.reset()
    
    def reset(self) -> GameState:
        """Reset warehouse to initial state."""
        # Create empty grid
        grid = np.zeros((self.height, self.width), dtype=np.int32)
        
        # Walls
        grid[0, :] = 1
        grid[-1, :] = 1
        grid[:, 0] = 1
        grid[:, -1] = 1
        
        # Random obstacles
        obstacles = set()
        for _ in range(self.num_obstacles):
            ox = random.randint(2, self.width - 3)
            oy = random.randint(2, self.height - 3)
            obstacles.add((oy, ox))
            grid[oy, ox] = 1
        
        # Robot starting position (bottom-left)
        robot_pos = (self.height - 2, 2)
        
        # Charging stations (2-3)
        charging_stations = set()
        for _ in range(2):
            cx = random.randint(2, self.width - 3)
            cy = random.randint(2, self.height - 3)
            if (cy, cx) not in obstacles and (cy, cx) != robot_pos:
                charging_stations.add((cy, cx))
        
        # Shipping zones (top area, separate regions)
        zones = ['A', 'B', 'C']
        shipping_zones = {}
        zone_width = (self.width - 4) // self.num_zones
        
        for i in range(self.num_zones):
            zone_name = f"ZONE_{zones[i]}"
            zone_positions = []
            start_x = 2 + i * zone_width
            for x in range(start_x, min(start_x + zone_width, self.width - 1)):
                for y in range(2, 4):
                    if (y, x) not in obstacles:
                        zone_positions.append((y, x))
            shipping_zones[zone_name] = zone_positions
        
        # Create packages
        packages = {}
        for i in range(self.num_packages):
            # Find empty spot
            while True:
                px = random.randint(3, self.width - 4)
                py = random.randint(4, self.height - 4)
                if ((py, px) not in obstacles and 
                    (py, px) != robot_pos and
                    (py, px) not in charging_stations):
                    break
            
            destination = f"ZONE_{zones[i % self.num_zones]}"
            priority = 2 if i < 2 else 1  # First 2 are high priority
            
            packages[i] = Package(
                id=i,
                position=(py, px),
                destination=destination,
                priority=priority,
            )
        
        # Create state
        self.state = WarehouseRobotState(
            robot_pos=robot_pos,
            battery=1.0,
            packages_held=[],
            packages=packages,
            shipping_zones=shipping_zones,
            obstacles=obstacles,
            charging_stations=charging_stations,
            grid=grid,
            width=self.width,
            height=self.height,
            steps=0,
            max_steps=self.max_steps_value,
            score=0,
            packages_delivered=0,
        )
        
        return self.state.to_game_state()
    
    def get_available_actions(self, state: Optional[GameState] = None) -> List[GameAction]:
        """Get available actions."""
        actions = [
            GameAction("MOVE_UP"),
            GameAction("MOVE_DOWN"),
            GameAction("MOVE_LEFT"),
            GameAction("MOVE_RIGHT"),
            GameAction("PICKUP"),
            GameAction("DELIVER"),
            GameAction("CHARGE"),
            GameAction("WAIT"),
        ]
        return actions
    
    def step(self, action: GameAction) -> GameResult:
        """Execute action."""
        if self.state is None:
            raise RuntimeError("Game not initialized")
        
        reward = -0.01  # Small penalty per step
        info = {"reason": "step"}
        
        ry, rx = self.state.robot_pos
        
        # Movement actions
        if action.name == "MOVE_UP":
            new_y, new_x = ry - 1, rx
        elif action.name == "MOVE_DOWN":
            new_y, new_x = ry + 1, rx
        elif action.name == "MOVE_LEFT":
            new_y, new_x = ry, rx - 1
        elif action.name == "MOVE_RIGHT":
            new_y, new_x = ry, rx + 1
        else:
            new_y, new_x = ry, rx
        
        # Validate movement
        if action.name.startswith("MOVE"):
            if (0 <= new_y < self.height and 0 <= new_x < self.width and
                (new_y, new_x) not in self.state.obstacles and
                self.state.grid[new_y, new_x] == 0):
                self.state.robot_pos = (new_y, new_x)
                self.state.battery -= 0.01  # Movement costs battery
            else:
                reward -= 0.05  # Invalid move penalty
                info["reason"] = "invalid_move"
        
        # Pickup action
        elif action.name == "PICKUP":
            if len(self.state.packages_held) < 3:  # Capacity limit
                for pkg in self.state.packages.values():
                    if (not pkg.collected and 
                        pkg.position == self.state.robot_pos):
                        pkg.collected = True
                        self.state.packages_held.append(pkg.id)
                        reward += 2.0 * pkg.priority  # Reward for pickup
                        self.state.score += 10 * pkg.priority
                        info["reason"] = "picked_up_package"
                        break
            else:
                reward -= 0.1
                info["reason"] = "capacity_full"
        
        # Deliver action
        elif action.name == "DELIVER":
            if self.state.packages_held:
                # Check if in any shipping zone
                delivered = False
                for zone_name, zone_positions in self.state.shipping_zones.items():
                    if self.state.robot_pos in zone_positions:
                        # Deliver packages for this zone
                        delivered_ids = []
                        for pkg_id in self.state.packages_held:
                            pkg = self.state.packages[pkg_id]
                            if pkg.destination == zone_name:
                                delivered_ids.append(pkg_id)
                                self.state.packages_delivered += 1
                                reward += 10.0 * pkg.priority
                                self.state.score += 50 * pkg.priority
                                delivered = True
                        
                        # Remove delivered packages
                        for pkg_id in delivered_ids:
                            self.state.packages_held.remove(pkg_id)
                        
                        if delivered:
                            info["reason"] = "delivered_packages"
                        else:
                            reward -= 0.5  # Wrong zone
                            info["reason"] = "wrong_zone"
                        break
            else:
                reward -= 0.1
                info["reason"] = "nothing_to_deliver"
        
        # Charge action
        elif action.name == "CHARGE":
            if self.state.robot_pos in self.state.charging_stations:
                self.state.battery = min(1.0, self.state.battery + 0.2)
                reward += 0.5
                info["reason"] = "charged"
            else:
                reward -= 0.1
                info["reason"] = "not_at_charger"
        
        # Battery drain
        self.state.battery -= 0.002
        if self.state.battery <= 0:
            reward -= 20.0
            info["reason"] = "battery_dead"
            self.state.battery = 0
        
        # Step increment
        self.state.steps += 1
        
        # Check completion
        done = (self.state.steps >= self.state.max_steps or
                self.state.battery <= 0 or
                all(p.collected for p in self.state.packages.values()))
        
        if done and all(p.collected for p in self.state.packages.values()):
            reward += 50.0  # Completion bonus
            info["success"] = True
            info["reason"] = "all_delivered"
        
        new_game_state = self.state.to_game_state()
        
        return GameResult(
            new_state=new_game_state,
            reward=reward,
            done=done,
            info=info
        )
    
    def get_current_state(self) -> GameState:
        """Get current state."""
        if self.state is None:
            raise RuntimeError("Game not initialized")
        return self.state.to_game_state()
    
    def get_abstract_concepts(self) -> Dict[str, List[str]]:
        """Get abstract concept mappings for transfer learning."""
        return {
            "AGENT": ["ROBOT"],
            "RESOURCE": ["BATTERY", "PACKAGE"],
            "RESOURCE_LOW": ["LOW_BATTERY", "CRITICAL_BATTERY"],
            "GOAL": ["ZONE_A", "ZONE_B", "ZONE_C", "CHARGER"],
            "GOAL_NEARBY": ["ZONE_A_NEARBY", "ZONE_B_NEARBY", "ZONE_C_NEARBY", "CHARGER_NEARBY"],
            "GOAL_CLOSE": ["ZONE_A_CLOSE", "ZONE_B_CLOSE", "ZONE_C_CLOSE", "CHARGER_CLOSE"],
            "THREAT": ["OBSTACLE"],
            "COLLECTIBLE": ["PACKAGE"],
            "COLLECTIBLE_NEARBY": ["PACKAGE_NEARBY"],
            "COLLECTIBLE_CLOSE": ["PACKAGE_CLOSE"],
        }
    
    def get_game_name(self) -> str:
        """Get game name."""
        return "warehouse"
    
    def get_max_steps(self) -> Optional[int]:
        """Get max steps."""
        return self.max_steps_value


if __name__ == "__main__":
    print("=" * 60)
    print("WAREHOUSE ROBOT GAME")
    print("=" * 60)
    print()
    print("A logistics/planning game completely different from GridWorld!")
    print()
    print("Objective: Collect packages and deliver to shipping zones")
    print("Challenges:")
    print("  - Battery management")
    print("  - Route planning")
    print("  - Priority handling")
    print("  - Capacity limits")
    print()
    
    game = WarehouseRobotGame(width=12, height=12, num_packages=5)
    state = game.reset()
    
    print(state.visual_repr)
    print()
    print("Legend:")
    print("  r = Robot (empty)")
    print("  R = Robot (carrying packages)")
    print("  P = Package")
    print("  2 = High-priority package")
    print("  A/B/C = Shipping zones")
    print("  C = Charging station")
    print("  X = Obstacle")
    print("  # = Wall")
    print()
