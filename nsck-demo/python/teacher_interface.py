from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import numpy as np

class TeacherInterface(ABC):
    """
    Abstract Base Class for all Teachers (Human, Heuristic, AI).
    Standardizes how the agent receives advice and feedback.
    """
    
    @abstractmethod
    def get_advice(self, state: Dict[str, Any], task_name: str) -> Optional[int]:
        """
        Ask teacher for the best action index in the current state.
        Returns None if teacher has no advice (abstains).
        """
        pass
    
    @abstractmethod
    def give_feedback(self, state: Dict[str, Any], action: int, reward: float, next_state: Dict[str, Any]) -> float:
        """
        Teacher provides additional reward/penalty signal.
        Returns a scalar modifier to add to the environment reward.
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass


class NullTeacher(TeacherInterface):
    """
    The 'No Teacher' implementation for autonomous mode.
    Silent observer that provides no advice and no extra feedback.
    """
    def get_advice(self, state, task_name):
        return None
        
    def give_feedback(self, state, action, reward, next_state):
        return 0.0
        
    @property
    def name(self):
        return "NullTeacher (Autonomous)"


class HeuristicTeacher(TeacherInterface):
    """
    Traditional hardcoded oracle logic (Snake/Pong/Maze).
    Moves logic out of python_server.py into this modular class.
    """
    
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        
        # Primitive relationship mapping for 'explanations'
        # (Could be expanded later)
        
    def get_advice(self, state, task_name):
        if task_name == "snake":
            return self._solve_snake(state)
        elif task_name == "pong":
            return self._solve_pong(state)
        elif task_name == "maze":
            return self._solve_maze(state)
        return None

    def give_feedback(self, state, action, reward, next_state):
        # Heuristic teacher could penalize "obvious" mistakes here
        return 0.0

    @property
    def name(self):
        return "HeuristicOracle"
        
    # --- LOGIC MOVED FROM SERVER ---
    
    def _solve_snake(self, state):
        # Re-implementing the robust oracle from python_server.py
        hx, hy = state["head"]
        fx, fy = state["food"]
        
        # Simple BFS or Manhattan logic?
        # Server used a "manhattan_snake_move" with obstacle avoidance.
        # We need the 'image' or obstacle list.
        # STATE usually contains 'obstacles' key if we parsed it, 
        # but raw state dict from game usually has head/food.
        # We might need to access the Global Game State or pass 'img'
        
        # LIMITATION: Standard 'state' dict might lack full grid usage without the image.
        # For now, implementing the simple greedy logic:
        
        dx = fx - hx
        dy = fy - hy
        
        # Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        # Note: Check coordinate system! Usually Y grows DOWN in images?
        # Server code: 0=UP (dy=-1), 1=DOWN (dy=1)
        
        obs = set()
        if "body" in state:
            for b in state["body"]: obs.add(tuple(b))
            
        # Try preferred moves
        moves = []
        if abs(dx) > abs(dy):
            moves.append(3 if dx > 0 else 2) # Horizontal
            moves.append(1 if dy > 0 else 0) # Vertical
        else:
            moves.append(1 if dy > 0 else 0)
            moves.append(3 if dx > 0 else 2)
            
        # Add the remaining 2 moves as backups
        all_moves = {0, 1, 2, 3}
        for m in moves: all_moves.discard(m)
        moves.extend(list(all_moves))
        
        # Return first safe move
        for m in moves:
            nx, ny = hx, hy
            if m == 0: ny -= 1
            elif m == 1: ny += 1
            elif m == 2: nx -= 1
            elif m == 3: nx += 1
            
            # Boundary
            if nx < 0 or nx >= self.grid_size or ny < 0 or ny >= self.grid_size:
                continue
            # Body collision
            if (nx, ny) in obs:
                continue
                
            return m
            
        return 0 # Fallback (death likely)

    def _solve_pong(self, state):
        by = state["ball_y"]
        py = state["p1_y"]
        # Actions: 0=UP, 1=DOWN
        if py + 3 < by - 1: return 1 # DOWN
        if py + 3 > by + 1: return 0 # UP
        return 0 # STAY (default to UP if aligned? or jitter)

    def _solve_maze(self, state):
        # Similar to snake but target is exit
        px, py = state["player_pos"]
        tx, ty = state["exit_pos"]
        walls = set(tuple(w) for w in state.get("walls", []))
        width = state.get("width", 15)
        height = state.get("height", 15)
        
        # BFS for wall-aware greedy guidance
        from collections import deque
        queue = deque([(px, py, [])])
        visited = {(px, py)}
        
        while queue:
            cx, cy, path = queue.popleft()
            if (cx, cy) == (tx, ty):
                if not path: return 0
                dx, dy = path[0]
                if dy == -1: return 0 # UP
                if dy == 1: return 1  # DOWN
                if dx == -1: return 2 # LEFT
                if dx == 1: return 3  # RIGHT
                return 0
            
            # Neighbors
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < width and 0 <= ny < height:
                    if (nx, ny) not in walls and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny, path + [(dx, dy)]))
        
        # Fallback to greedy if no path found
        dx, dy = tx - px, ty - py
        if abs(dx) > abs(dy):
             return 3 if dx > 0 else 2
        else:
             return 1 if dy > 0 else 0
