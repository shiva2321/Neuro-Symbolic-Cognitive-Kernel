"""
NSCK Maze Game Environment
A simple maze game for testing cross-task transfer from Snake.

The player navigates through a maze to reach an exit while avoiding walls.
"""
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum


class Cell(Enum):
    """Maze cell types."""
    EMPTY = 0
    WALL = 1
    PLAYER = 2
    EXIT = 3
    VISITED = 4


@dataclass
class MazeState:
    """State of the maze game."""
    player_pos: Tuple[int, int]
    exit_pos: Tuple[int, int]
    maze: List[List[int]]
    width: int
    height: int
    steps: int = 0
    score: int = 0
    done: bool = False
    visited: List[Tuple[int, int]] = None
    
    def __post_init__(self):
        if self.visited is None:
            self.visited = [self.player_pos]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for ZMQ transmission."""
        # Extract wall positions for spatial reasoning
        walls = []
        for y in range(self.height):
            for x in range(self.width):
                if self.maze[y][x] == Cell.WALL.value:
                    walls.append((x, y))

        return {
            "player_pos": self.player_pos,
            "exit_pos": self.exit_pos,
            "head": self.player_pos,   # Alias for Snake/Planner compatibility
            "target": self.exit_pos,   # Alias for Snake/Planner compatibility
            "walls": walls,            # Obstacles for Planner
            "width": self.width,
            "height": self.height,
            "steps": self.steps,
            "score": self.score,
            "done": self.done,
            "visited_count": len(self.visited),
        }


class MazeGame:
    """
    Simple maze game with procedural generation.
    
    Structure is similar to Snake but navigating through corridors
    instead of an open field.
    """
    
    def __init__(self, width: int = 15, height: int = 15, wall_density: float = 0.2):
        """
        Initialize maze game.
        
        Args:
            width: Maze width
            height: Maze height
            wall_density: Probability of a cell being a wall
        """
        self.width = width
        self.height = height
        self.wall_density = wall_density
        self.state: Optional[MazeState] = None
        self.reset()
    
    def reset(self) -> MazeState:
        """Reset game to initial state."""
        # Generate maze
        maze = self._generate_maze()
        
        # Place player in top-left area
        player_pos = self._find_empty_cell(maze, prefer_area="top_left")
        
        # Place exit in bottom-right area
        exit_pos = self._find_empty_cell(maze, prefer_area="bottom_right")
        
        # Ensure path exists
        if not self._has_path(maze, player_pos, exit_pos):
            # Clear a path if needed
            self._clear_path(maze, player_pos, exit_pos)
        
        self.state = MazeState(
            player_pos=player_pos,
            exit_pos=exit_pos,
            maze=maze,
            width=self.width,
            height=self.height,
            steps=0,
            score=0,
            done=False,
            visited=[player_pos]
        )
        
        return self.state
    
    def _generate_maze(self) -> List[List[int]]:
        """Generate a random maze with walls."""
        maze = [[Cell.EMPTY.value for _ in range(self.width)] for _ in range(self.height)]
        
        # Add random walls
        for y in range(self.height):
            for x in range(self.width):
                # Keep borders clear for easier navigation
                if x == 0 or y == 0 or x == self.width - 1 or y == self.height - 1:
                    continue
                if random.random() < self.wall_density:
                    maze[y][x] = Cell.WALL.value
        
        return maze
    
    def _find_empty_cell(
        self,
        maze: List[List[int]],
        prefer_area: str = "any"
    ) -> Tuple[int, int]:
        """Find an empty cell, preferring a specific area."""
        candidates = []
        
        for y in range(self.height):
            for x in range(self.width):
                if maze[y][x] == Cell.EMPTY.value:
                    # Apply area preference
                    if prefer_area == "top_left":
                        if x < self.width // 3 and y < self.height // 3:
                            candidates.append((x, y))
                    elif prefer_area == "bottom_right":
                        if x > 2 * self.width // 3 and y > 2 * self.height // 3:
                            candidates.append((x, y))
                    else:
                        candidates.append((x, y))
        
        if not candidates:
            # Fallback: any empty cell
            for y in range(self.height):
                for x in range(self.width):
                    if maze[y][x] == Cell.EMPTY.value:
                        return (x, y)
            # Last resort
            return (1, 1)
        
        return random.choice(candidates)
    
    def _has_path(
        self,
        maze: List[List[int]],
        start: Tuple[int, int],
        end: Tuple[int, int]
    ) -> bool:
        """Check if a path exists using BFS."""
        from collections import deque
        
        if start == end:
            return True
        
        visited = set()
        queue = deque([start])
        
        while queue:
            x, y = queue.popleft()
            
            if (x, y) == end:
                return True
            
            if (x, y) in visited:
                continue
            visited.add((x, y))
            
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if maze[ny][nx] != Cell.WALL.value:
                        queue.append((nx, ny))
        
        return False
    
    def _clear_path(
        self,
        maze: List[List[int]],
        start: Tuple[int, int],
        end: Tuple[int, int]
    ):
        """Clear a straight-ish path between two points."""
        x, y = start
        ex, ey = end
        
        while (x, y) != (ex, ey):
            maze[y][x] = Cell.EMPTY.value
            
            if x < ex:
                x += 1
            elif x > ex:
                x -= 1
            elif y < ey:
                y += 1
            elif y > ey:
                y -= 1
        
        maze[ey][ex] = Cell.EMPTY.value
    
    def step(self, action: str) -> Tuple[MazeState, float, bool]:
        """
        Take a step in the maze.
        
        Args:
            action: ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT
            
        Returns:
            Tuple of (new_state, reward, done)
        """
        if self.state is None or self.state.done:
            return self.state, 0.0, True
        
        # Parse action
        print(f"[MAZE_GAME] Received action: {action}")
        action = action.upper().replace("ACTION_", "")
        print(f"[MAZE_GAME] Processed action: {action}")
        
        dx, dy = 0, 0
        if action == "UP":
            dy = -1
        elif action == "DOWN":
            dy = 1
        elif action == "LEFT":
            dx = -1
        elif action == "RIGHT":
            dx = 1
        
        # Calculate new position
        x, y = self.state.player_pos
        nx, ny = x + dx, y + dy
        
        # Check bounds
        if not (0 <= nx < self.width and 0 <= ny < self.height):
            # Hit boundary
            return self.state, -0.1, False
        
        # Check walls
        if self.state.maze[ny][nx] == Cell.WALL.value:
            # Hit wall
            return self.state, -0.5, False
        
        # Move successful
        old_pos = self.state.player_pos
        self.state.player_pos = (nx, ny)
        self.state.steps += 1
        print(f"[MAZE_GAME] Moved from {old_pos} to {self.state.player_pos}")
        
        # Track visited
        if (nx, ny) not in self.state.visited:
            self.state.visited.append((nx, ny))
        
        # Check if reached exit
        if (nx, ny) == self.state.exit_pos:
            self.state.done = True
            self.state.score = max(100 - self.state.steps, 10)
            return self.state, 10.0, True
        
        # Small reward for exploring new cells
        if (nx, ny) not in self.state.visited[:-1]:
            reward = 0.1
        else:
            reward = -0.05  # Discourage revisiting
        
        return self.state, reward, False
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get state for cognitive engine."""
        if self.state is None:
            return {}
        
        # Extract wall positions for spatial reasoning
        walls = []
        for y in range(self.height):
            for x in range(self.width):
                if self.state.maze[y][x] == Cell.WALL.value:
                    walls.append((x, y))

        return {
            "player_pos": self.state.player_pos,
            "exit_pos": self.state.exit_pos,
            "width": self.width,
            "height": self.height,
            "steps": self.state.steps,
            "visited": self.state.visited,
            "walls": walls,
            # Similar to Snake's state for transfer
            "head": self.state.player_pos,
            "food": self.state.exit_pos,
            "target": self.state.exit_pos, # Common alias
            "body": [], # DO NOT treat visited as body/obstacles in Maze
        }
    
    def render_ascii(self) -> str:
        """Render maze as ASCII for debugging."""
        if self.state is None:
            return "No game in progress"
        
        lines = []
        for y in range(self.height):
            row = ""
            for x in range(self.width):
                if (x, y) == self.state.player_pos:
                    row += "P"
                elif (x, y) == self.state.exit_pos:
                    row += "E"
                elif self.state.maze[y][x] == Cell.WALL.value:
                    row += "#"
                elif (x, y) in self.state.visited:
                    row += "."
                else:
                    row += " "
            lines.append(row)
        
        return "\n".join(lines)


def create_maze_verifier():
    """Create grounding verifier for Maze game."""
    from .grounding_verifier import GroundingVerifier
    
    verifier = GroundingVerifier()
    
    # Directional predicates (similar to Snake)
    def exit_above(state):
        player = state.get("player_pos") or state.get("head")
        exit_pos = state.get("exit_pos") or state.get("food")
        if player and exit_pos:
            return exit_pos[1] < player[1]
        return False
    
    def exit_below(state):
        player = state.get("player_pos") or state.get("head")
        exit_pos = state.get("exit_pos") or state.get("food")
        if player and exit_pos:
            return exit_pos[1] > player[1]
        return False
    
    def exit_left(state):
        player = state.get("player_pos") or state.get("head")
        exit_pos = state.get("exit_pos") or state.get("food")
        if player and exit_pos:
            return exit_pos[0] < player[0]
        return False
    
    def exit_right(state):
        player = state.get("player_pos") or state.get("head")
        exit_pos = state.get("exit_pos") or state.get("food")
        if player and exit_pos:
            return exit_pos[0] > player[0]
        return False
    
    verifier.register_predicate("EXIT_ABOVE", exit_above, context="maze")
    verifier.register_predicate("EXIT_BELOW", exit_below, context="maze")
    verifier.register_predicate("EXIT_LEFT", exit_left, context="maze")
    verifier.register_predicate("EXIT_RIGHT", exit_right, context="maze")
    
    return verifier


def create_maze_causal_graph():
    """Create causal graph for Maze game."""
    from .causal_reasoning import CausalGraph
    
    graph = CausalGraph()
    
    # Movement causes position changes
    graph.add_causes("ACTION_UP", "PLAYER_MOVES_UP", context="maze")
    graph.add_causes("ACTION_DOWN", "PLAYER_MOVES_DOWN", context="maze")
    graph.add_causes("ACTION_LEFT", "PLAYER_MOVES_LEFT", context="maze")
    graph.add_causes("ACTION_RIGHT", "PLAYER_MOVES_RIGHT", context="maze")
    
    # Position changes can cause wall collisions
    graph.add_causes("PLAYER_MOVES_UP", "WALL_HIT", strength=0.2, context="maze")
    graph.add_causes("PLAYER_MOVES_DOWN", "WALL_HIT", strength=0.2, context="maze")
    graph.add_causes("PLAYER_MOVES_LEFT", "WALL_HIT", strength=0.2, context="maze")
    graph.add_causes("PLAYER_MOVES_RIGHT", "WALL_HIT", strength=0.2, context="maze")
    
    # Reaching exit
    graph.add_causes("PLAYER_AT_EXIT", "WIN", context="maze")
    graph.add_causes("WIN", "SCORE_UP", context="maze")
    
    return graph


# Simulation function for integration with python_server.py
def sim_maze(state: Dict, action: str) -> Tuple[Dict, bool]:
    """
    Simulate maze game step.
    
    Compatible with sim_snake and sim_pong interface.
    
    Args:
        state: Current state dict
        action: Action to take
        
    Returns:
        Tuple of (next_state, terminal)
    """
    # This is a simplified simulation for physics verification
    player_pos = state.get("player_pos") or state.get("head", (7, 7))
    exit_pos = state.get("exit_pos") or state.get("food", (14, 14))
    
    x, y = player_pos
    action = action.upper().replace("ACTION_", "")
    
    if action == "UP":
        y = max(0, y - 1)
    elif action == "DOWN":
        y = min(14, y + 1)
    elif action == "LEFT":
        x = max(0, x - 1)
    elif action == "RIGHT":
        x = min(14, x + 1)
    
    new_pos = (x, y)
    terminal = (new_pos == exit_pos)
    
    return {
        "player_pos": new_pos,
        "exit_pos": exit_pos,
        "head": new_pos,
        "food": exit_pos,
    }, terminal
