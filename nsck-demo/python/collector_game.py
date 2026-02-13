"""
NSCK Collector Game Environment
================================
A grid-world game where the agent collects numbered items in sequence
while avoiding static obstacles.

Designed to test transfer learning from Snake/Maze — shares the same
action space and predicate vocabulary but introduces multi-target
sequential collection as a novel mechanic.
"""
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum


class CellType(Enum):
    """Cell types for the Collector grid."""
    EMPTY = 0
    OBSTACLE = 1
    PLAYER = 2
    ITEM = 3


@dataclass
class CollectorState:
    """State of the collector game."""
    player_pos: Tuple[int, int]
    items: List[Tuple[int, int]]        # Positions of items in collection order
    collected: int                       # Number of items collected so far
    obstacles: set                       # Set of (x, y) obstacle positions
    width: int
    height: int
    steps: int = 0
    score: int = 0
    done: bool = False

    @property
    def next_target(self) -> Optional[Tuple[int, int]]:
        """Position of the next item to collect, or None if all collected."""
        if self.collected < len(self.items):
            return self.items[self.collected]
        return None

    @property
    def total_items(self) -> int:
        return len(self.items)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for transmission / cognitive engine."""
        target = self.next_target
        return {
            "player_pos": self.player_pos,
            "head": self.player_pos,           # Snake/Maze compatibility
            "items": list(self.items),
            "collected": self.collected,
            "total_items": self.total_items,
            "next_target": target,
            "target": target,                  # Planner compatibility
            "food": target,                    # Snake compatibility
            "obstacles": list(self.obstacles),
            "walls": list(self.obstacles),     # Maze compatibility
            "body": [],                        # Snake compatibility (no body)
            "width": self.width,
            "height": self.height,
            "steps": self.steps,
            "score": self.score,
            "done": self.done,
        }


class CollectorGame:
    """
    Grid-world game: collect numbered items in order while avoiding obstacles.

    - 10x10 grid (default)
    - 3-5 items placed at random empty cells
    - ~15% obstacle density
    - Same action space as Snake/Maze: UP, DOWN, LEFT, RIGHT
    - Agent must collect items in numerical order (1, 2, 3, ...)
    """

    MAX_STEPS = 300

    def __init__(
        self,
        width: int = 10,
        height: int = 10,
        num_items: int = 4,
        obstacle_density: float = 0.12,
    ):
        self.width = width
        self.height = height
        self.num_items = num_items
        self.obstacle_density = obstacle_density
        self.state: Optional[CollectorState] = None
        self.reset()

    def reset(self) -> CollectorState:
        """Reset game to a fresh board."""
        grid, obstacles = self._generate_grid()
        player_pos = self._place_on_empty(grid, obstacles)
        items = []
        occupied = obstacles | {player_pos}
        for _ in range(self.num_items):
            pos = self._place_on_empty(grid, occupied)
            items.append(pos)
            occupied.add(pos)

        # Ensure a path exists from player to first item
        # (and between consecutive items) by carving if needed
        all_targets = [player_pos] + items
        for i in range(len(all_targets) - 1):
            if not self._has_path(all_targets[i], all_targets[i + 1], obstacles):
                self._carve_path(all_targets[i], all_targets[i + 1], obstacles)

        self.state = CollectorState(
            player_pos=player_pos,
            items=items,
            collected=0,
            obstacles=obstacles,
            width=self.width,
            height=self.height,
            steps=0,
            score=0,
            done=False,
        )
        return self.state

    def _generate_grid(self) -> Tuple[List[List[int]], set]:
        """Generate a grid with scattered obstacles."""
        grid = [[CellType.EMPTY.value] * self.width for _ in range(self.height)]
        obstacles = set()

        num_obstacles = int(self.width * self.height * self.obstacle_density)
        for _ in range(num_obstacles):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            grid[y][x] = CellType.OBSTACLE.value
            obstacles.add((x, y))

        return grid, obstacles

    def _place_on_empty(self, grid, occupied: set) -> Tuple[int, int]:
        """Find a random empty cell not in the occupied set."""
        for _ in range(1000):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            if (x, y) not in occupied:
                return (x, y)
        # Fallback: linear scan
        for y in range(self.height):
            for x in range(self.width):
                if (x, y) not in occupied:
                    return (x, y)
        return (0, 0)

    def _has_path(self, start, end, obstacles) -> bool:
        """BFS path check."""
        from collections import deque
        if start == end:
            return True
        visited = {start}
        queue = deque([start])
        while queue:
            cx, cy = queue.popleft()
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if (nx, ny) not in visited and (nx, ny) not in obstacles:
                        if (nx, ny) == end:
                            return True
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        return False

    def _carve_path(self, start, end, obstacles: set):
        """Remove obstacles along a straight-ish path between two points."""
        x, y = start
        ex, ey = end
        while (x, y) != (ex, ey):
            obstacles.discard((x, y))
            if x < ex:
                x += 1
            elif x > ex:
                x -= 1
            elif y < ey:
                y += 1
            elif y > ey:
                y -= 1
        obstacles.discard((ex, ey))

    def step(self, action: str) -> Tuple[CollectorState, float, bool]:
        """
        Take a step in the collector game.

        Args:
            action: UP, DOWN, LEFT, RIGHT (or ACTION_UP etc.)

        Returns:
            Tuple of (state, reward, done)
        """
        if self.state is None or self.state.done:
            return self.state, 0.0, True

        action = action.upper().replace("ACTION_", "")

        dx, dy = 0, 0
        if action == "UP":
            dy = -1
        elif action == "DOWN":
            dy = 1
        elif action == "LEFT":
            dx = -1
        elif action == "RIGHT":
            dx = 1

        px, py = self.state.player_pos
        nx, ny = px + dx, py + dy

        reward = -0.01  # Small step penalty to encourage efficiency

        # Bounds check
        if not (0 <= nx < self.width and 0 <= ny < self.height):
            self.state.steps += 1
            if self.state.steps >= self.MAX_STEPS:
                self.state.done = True
            return self.state, -0.1, self.state.done

        # Obstacle check
        if (nx, ny) in self.state.obstacles:
            self.state.steps += 1
            if self.state.steps >= self.MAX_STEPS:
                self.state.done = True
            return self.state, -0.1, self.state.done

        # Move
        self.state.player_pos = (nx, ny)
        self.state.steps += 1

        # Check if we collected the next item
        target = self.state.next_target
        if target and (nx, ny) == target:
            self.state.collected += 1
            self.state.score += 1
            reward = 1.0

            # Check if all items collected
            if self.state.collected >= self.state.total_items:
                self.state.done = True
                reward = 5.0  # Bonus for completing all

        # Max steps
        if self.state.steps >= self.MAX_STEPS:
            self.state.done = True

        return self.state, reward, self.state.done

    def get_state_dict(self) -> Dict[str, Any]:
        """Get state for cognitive engine."""
        if self.state is None:
            return {}
        return self.state.to_dict()

    def render_ascii(self) -> str:
        """Render the game as ASCII for debugging."""
        if self.state is None:
            return "No game in progress"

        # Build item position map
        item_map = {}
        for i, pos in enumerate(self.state.items):
            if i >= self.state.collected:
                item_map[pos] = str(i + 1)

        lines = []
        for y in range(self.height):
            row = ""
            for x in range(self.width):
                if (x, y) == self.state.player_pos:
                    row += "P"
                elif (x, y) in item_map:
                    row += item_map[(x, y)]
                elif (x, y) in self.state.obstacles:
                    row += "#"
                else:
                    row += "."
                row += " "
            lines.append(row)

        lines.append(f"Score: {self.state.score}/{self.state.total_items}  "
                     f"Steps: {self.state.steps}/{self.MAX_STEPS}  "
                     f"Next: item {self.state.collected + 1}")
        return "\n".join(lines)


# =========================================================================
# Simulation function — lightweight, compatible with sim_snake/sim_pong
# =========================================================================

def sim_collector(state: Dict, action: str) -> Tuple[Dict, bool]:
    """
    Simulate one step of Collector (lightweight).

    Compatible with sim_snake / sim_pong interface.

    Args:
        state: Current state dict with player_pos, next_target, obstacles, items, collected
        action: UP, DOWN, LEFT, RIGHT

    Returns:
        Tuple of (next_state, terminal)
    """
    player = state.get("player_pos") or state.get("head", (5, 5))
    obstacles = set(tuple(o) for o in state.get("obstacles", state.get("walls", [])))
    items = [tuple(i) for i in state.get("items", [])]
    collected = state.get("collected", 0)
    width = state.get("width", 10)
    height = state.get("height", 10)

    x, y = player
    action = action.upper().replace("ACTION_", "")

    if action == "UP":
        y = max(0, y - 1)
    elif action == "DOWN":
        y = min(height - 1, y + 1)
    elif action == "LEFT":
        x = max(0, x - 1)
    elif action == "RIGHT":
        x = min(width - 1, x + 1)

    new_pos = (x, y)

    # Check obstacle — don't move if blocked
    if new_pos in obstacles:
        new_pos = player

    # Check item collection
    new_collected = collected
    if new_collected < len(items) and new_pos == items[new_collected]:
        new_collected += 1

    terminal = new_collected >= len(items)
    next_target = items[new_collected] if new_collected < len(items) else None

    return {
        "player_pos": new_pos,
        "head": new_pos,
        "items": items,
        "collected": new_collected,
        "next_target": next_target,
        "target": next_target,
        "food": next_target,
        "obstacles": list(obstacles),
        "walls": list(obstacles),
        "body": [],
        "width": width,
        "height": height,
    }, terminal
