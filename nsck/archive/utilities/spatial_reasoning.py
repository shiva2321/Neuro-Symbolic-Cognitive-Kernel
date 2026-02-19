"""
NSCK Spatial Reasoning Module
Provides coordinate-based reasoning for planning.
"""
from typing import Set, Tuple, Dict, Any, List, Optional

class CoordinateReasoner:
    """
    Simulates spatial transitions for the Planner.
    Used to predict 'AT_X_Y' state changes based on actions.
    """
    
    def __init__(self, grid_size: int = 10):
        self.grid_size = grid_size
        
        # Mock graph property for planner compatibility
        # Planner calls: reasoner.graph.get_immediate_effects(action)
        # But planner ALSO calls _predict which we can override if we use a subclass of Planner
        # OR we can just allow planner to call methods on us if we wrap it.
        
        # Actually, planner.py default _predict calls self.causal_reasoner.graph.get_immediate_effects(action).
        # This returns a set of strings.
        # It does NOT take 'state' as input effectively.
        # This is a limitation of the current Planner implementation.
        
        # However, Phase 2.2 verified the planner with a subclass `GridPlanner`.
        # So we should use a `GridPlanner` instance for Snake, not the generic `STRIPSPlanner`!
        pass

# We will actually define the GridPlanner here for use in CognitiveEngine
from python.core.reasoning.planner import STRIPSPlanner

class GridPlanner(STRIPSPlanner):
    """
    Spatial planner for grid worlds.
    Predicts coordinate changes dynamically.
    """
    def __init__(self, grid_size: int = 10, obstacles: List[Tuple[int, int]] = None):
        super().__init__()
        self.grid_size = grid_size
        self.obstacles = set(obstacles) if obstacles else set()
        
    def get_next_action(self, state: Dict[str, Any], task_tag: str) -> Optional[str]:
        """High-level API: Get next move for Snake/Maze."""
        self.current_task = task_tag
        # Update grid size if available in state
        if "width" in state:
            self.grid_size = state["width"]
        
        # 1. Parse state to predicates
        preds = set()
        head = state.get("head") # Tuple (x,y)
        food = state.get("food") or state.get("target") # Support both Snake and Maze
        
        # Dynamically update obstacles (walls in Maze, body in Snake)
        self.obstacles = set()
        if "walls" in state:
            self.obstacles.update(tuple(w) for w in state["walls"])
        if "body" in state:
            # Body segments are obstacles
            self.obstacles.update(tuple(b) for b in state["body"])
        
        if head:
            preds.add(f"AT_{head[0]}_{head[1]}")
        
        goal_preds = set()
        if food:
            goal_preds.add(f"AT_{food[0]}_{food[1]}")
            
        if not preds or not goal_preds:
            return None
            
        # 2. Plan (Direct BFS is robust for 15x15 grids)
        plan = self.plan(preds, goal_preds, max_depth=200)
        
        # 3. Return first step
        if plan and len(plan) > 0:
            if task_tag == "maze":
                print(f"[DEBUG] GridPlanner: Found plan for Maze (len={len(plan)}), next: {plan[0]}")
            return plan[0]
            
        return None

    def _predict(self, state: frozenset, action: str) -> Tuple[Set[str], Set[str]]:
        """
        Predict (add, del) effects for movement on grid.
        """
        x, y = -1, -1
        old_pred = None
        
        # Extract current position
        for pred in state:
            if pred.startswith("AT_"):
                parts = pred.split("_")
                try:
                    x, y = int(parts[1]), int(parts[2])
                    old_pred = pred
                    break
                except:
                    pass
        
        if x == -1:
            # No location found, return empty effects (can't move if we don't exist)
            return set(), set()
            
        new_x, new_y = x, y
        
        if action == "ACTION_UP": new_y -= 1
        elif action == "ACTION_DOWN": new_y += 1
        elif action == "ACTION_LEFT": new_x -= 1
        elif action == "ACTION_RIGHT": new_x += 1
        else:
            return set(), set()
            
        # Obstacle Check
        if (new_x, new_y) in self.obstacles:
            return set(), set() # Cannot move there
            
        # Wrap or Bound? Snake usually wraps, Maze does NOT.
        is_maze = getattr(self, "current_task", "") == "maze"
        
        if is_maze:
            # Check hard bounds for Maze
            if not (0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size):
                return set(), set()
        else:
            # Wrap for Snake
            new_x = new_x % self.grid_size
            new_y = new_y % self.grid_size
        
        add_effects = {f"AT_{new_x}_{new_y}"}
        del_effects = {old_pred} if old_pred else set()
        
        return add_effects, del_effects

    def decompose_goal(self, state: frozenset[str], goal: Set[str]) -> List[Set[str]]:
        """
        Decomposes long spatial paths into subgoals via midpoints.
        """
        start_pos = None
        goal_pos = None
        
        # 1. Extract positions
        for pred in state:
            if pred.startswith("AT_"):
                try:
                    parts = pred.split("_")
                    if len(parts) >= 3:
                        start_pos = (int(parts[1]), int(parts[2]))
                        break
                except:
                    pass
                
        for pred in goal:
            if pred.startswith("AT_"):
                try:
                    parts = pred.split("_")
                    if len(parts) >= 3:
                        goal_pos = (int(parts[1]), int(parts[2]))
                        break
                except:
                    pass

        if not start_pos or not goal_pos:
            return []

        # 2. Check distance (Manhattan)
        # Note: This simple heuristic ignores wrapping for decomposition logic,
        # relying on the sub-planner to handle the actual wrap-around path if shorter.
        dx = abs(goal_pos[0] - start_pos[0])
        dy = abs(goal_pos[1] - start_pos[1])
        dist = dx + dy
        
        # Threshold: plans > 6 steps should be decomposed
        if dist > 6:
            # 3. Create midpoint
            mid_x = (start_pos[0] + goal_pos[0]) // 2
            mid_y = (start_pos[1] + goal_pos[1]) // 2
            
            subgoal = {f"AT_{mid_x}_{mid_y}"}
            # Return list of subgoals: [Midpoint, OriginalGoal]
            return [subgoal, goal]
            
        return []
