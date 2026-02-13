"""
NSCK Grounding Verifier Module
Verifies that predicates and actions match physical reality.
"""
from typing import Callable, Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class GroundingResult:
    """Result of a grounding verification."""
    predicate: str
    is_grounded: bool
    expected: bool
    actual: bool
    context: str = ""


class GroundingVerifier:
    """
    Verifies that symbolic predicates correspond to physical reality.
    
    This is critical for ensuring learned rules are meaningful.
    Must be initialized BEFORE rule learning to prevent learning
    from misgrounded predicates.
    """
    
    def __init__(self):
        """Initialize verifier with empty grounding checks."""
        self.predicate_checks: Dict[str, Callable[[Dict], bool]] = {}
        self.action_checks: Dict[str, Callable[[Dict, Dict], bool]] = {}
        self.context_predicates: Dict[str, Dict[str, Callable[[Dict], bool]]] = {}
    
    def register_predicate(
        self,
        name: str,
        check: Callable[[Dict], bool],
        context: Optional[str] = None
    ):
        """
        Define what a predicate means.
        
        Args:
            name: Predicate name (e.g., "REL_ABOVE")
            check: Function that takes state dict and returns bool
            context: Optional context for context-specific grounding
        """
        if context:
            if context not in self.context_predicates:
                self.context_predicates[context] = {}
            self.context_predicates[context][name] = check
        else:
            self.predicate_checks[name] = check
    
    def register_action(
        self,
        name: str,
        effect_check: Callable[[Dict, Dict], bool],
        context: Optional[str] = None
    ):
        """
        Define what an action's effect should be.
        
        Args:
            name: Action name (e.g., "ACTION_UP")
            effect_check: Function(before_state, after_state) -> bool
            context: Optional context for context-specific grounding
        """
        key = f"{context}::{name}" if context else name
        self.action_checks[key] = effect_check
    
    def verify_predicate(
        self,
        name: str,
        state: Dict,
        context: Optional[str] = None
    ) -> bool:
        """
        Check if a predicate is true in given state.
        
        Args:
            name: Predicate name
            state: Current game state
            context: Optional context for context-specific lookup
            
        Returns:
            True if predicate holds, False if not or unknown
        """
        # Try context-specific first
        if context and context in self.context_predicates:
            if name in self.context_predicates[context]:
                try:
                    return self.context_predicates[context][name](state)
                except Exception:
                    return False
        
        # Fall back to global
        if name not in self.predicate_checks:
            return False  # Unknown predicate = not grounded
        
        try:
            return self.predicate_checks[name](state)
        except Exception:
            return False
    
    def verify_action(
        self,
        name: str,
        before_state: Dict,
        after_state: Dict,
        context: Optional[str] = None
    ) -> bool:
        """
        Verify that an action had its expected effect.
        
        Args:
            name: Action name
            before_state: State before action
            after_state: State after action
            context: Optional context
            
        Returns:
            True if action effect matches expectation
        """
        key = f"{context}::{name}" if context else name
        
        if key not in self.action_checks:
            # Try without context
            if name in self.action_checks:
                key = name
            else:
                return False
        
        try:
            return self.action_checks[key](before_state, after_state)
        except Exception:
            return False
    
    def verify_rule(
        self,
        rule_condition: frozenset,
        episodes: List[Any],
        context: Optional[str] = None
    ) -> float:
        """
        Check rule accuracy against episode history.
        
        Args:
            rule_condition: Set of predicates in rule condition
            episodes: List of episodes with 'state' and 'action' attributes
            context: Optional context
            
        Returns:
            Accuracy (0.0 to 1.0) of predicate grounding
        """
        if not episodes:
            return 0.0
        
        correct = 0
        total = 0
        
        for ep in episodes:
            state = getattr(ep, 'state_sketch', {}) or getattr(ep, 'state', {})
            if not state:
                continue
            
            # Check if all predicates in condition are grounded correctly
            all_grounded = all(
                self.verify_predicate(pred, state, context)
                for pred in rule_condition
            )
            
            if all_grounded:
                correct += 1
            total += 1
        
        return correct / total if total > 0 else 0.0
    
    def get_active_predicates(self, state: Dict, context: Optional[str] = None) -> List[str]:
        """
        Get all predicates that are true in given state.
        
        Args:
            state: Current game state
            context: Optional context
            
        Returns:
            List of predicate names that are true
        """
        active = []
        
        # Check global predicates
        for name, check in self.predicate_checks.items():
            try:
                if check(state):
                    active.append(name)
            except Exception:
                pass
        
        # Check context-specific predicates
        if context and context in self.context_predicates:
            for name, check in self.context_predicates[context].items():
                try:
                    if check(state):
                        active.append(f"{context}::{name}")
                except Exception:
                    pass
        
        return active


def create_snake_verifier() -> GroundingVerifier:
    """
    Create a grounding verifier with Snake-specific predicates.
    
    Returns:
        Configured GroundingVerifier for Snake game
    """
    verifier = GroundingVerifier()
    
    # Directional predicates
    verifier.register_predicate(
        "REL_ABOVE",
        lambda s: s.get("food", (0, 0))[1] < s.get("head", (0, 0))[1],
        context="snake"
    )
    verifier.register_predicate(
        "REL_BELOW",
        lambda s: s.get("food", (0, 0))[1] > s.get("head", (0, 0))[1],
        context="snake"
    )
    verifier.register_predicate(
        "REL_LEFT",
        lambda s: s.get("food", (0, 0))[0] < s.get("head", (0, 0))[0],
        context="snake"
    )
    verifier.register_predicate(
        "REL_RIGHT",
        lambda s: s.get("food", (0, 0))[0] > s.get("head", (0, 0))[0],
        context="snake"
    )
    
    # Danger predicates
    verifier.register_predicate(
        "DANGER_UP",
        lambda s: _wall_or_body_at(s, 0, -1),
        context="snake"
    )
    verifier.register_predicate(
        "DANGER_DOWN",
        lambda s: _wall_or_body_at(s, 0, 1),
        context="snake"
    )
    verifier.register_predicate(
        "DANGER_LEFT",
        lambda s: _wall_or_body_at(s, -1, 0),
        context="snake"
    )
    verifier.register_predicate(
        "DANGER_RIGHT",
        lambda s: _wall_or_body_at(s, 1, 0),
        context="snake"
    )
    
    # Action effects
    verifier.register_action(
        "ACTION_UP",
        lambda b, a: a.get("head", (0, 0))[1] < b.get("head", (0, 0))[1],
        context="snake"
    )
    verifier.register_action(
        "ACTION_DOWN",
        lambda b, a: a.get("head", (0, 0))[1] > b.get("head", (0, 0))[1],
        context="snake"
    )
    verifier.register_action(
        "ACTION_LEFT",
        lambda b, a: a.get("head", (0, 0))[0] < b.get("head", (0, 0))[0],
        context="snake"
    )
    verifier.register_action(
        "ACTION_RIGHT",
        lambda b, a: a.get("head", (0, 0))[0] > b.get("head", (0, 0))[0],
        context="snake"
    )
    
    return verifier


def _wall_or_body_at(state: Dict, dx: int, dy: int, grid_size: int = 10) -> bool:
    """Check if there's a wall or body segment at offset from head."""
    head = state.get("head", (5, 5))
    body = state.get("body", [])
    
    nx = (head[0] + dx) % grid_size
    ny = (head[1] + dy) % grid_size
    
    # Check body collision (wrapping handled by mod)
    return (nx, ny) in set(body)


def create_pong_verifier() -> GroundingVerifier:
    """
    Create a grounding verifier with Pong-specific predicates.
    
    Returns:
        Configured GroundingVerifier for Pong game
    """
    verifier = GroundingVerifier()
    
    # Ball position relative to paddle
    verifier.register_predicate(
        "BALL_ABOVE",
        lambda s: s.get("ball_y", 15) < s.get("p1_y", 10) + 3,
        context="pong"
    )
    verifier.register_predicate(
        "BALL_BELOW",
        lambda s: s.get("ball_y", 15) > s.get("p1_y", 10) + 3,
        context="pong"
    )
    verifier.register_predicate(
        "BALL_ALIGNED",
        lambda s: abs(s.get("ball_y", 15) - (s.get("p1_y", 10) + 3)) <= 1,
        context="pong"
    )
    
    # Ball approaching
    verifier.register_predicate(
        "BALL_APPROACHING",
        lambda s: s.get("ball_dx", 0) < 0,
        context="pong"
    )
    
    # Action effects
    verifier.register_action(
        "ACTION_UP",
        lambda b, a: a.get("p1_y", 10) < b.get("p1_y", 10),
        context="pong"
    )
    verifier.register_action(
        "ACTION_DOWN",
        lambda b, a: a.get("p1_y", 10) > b.get("p1_y", 10),
        context="pong"
    )
    
    return verifier


def create_maze_verifier() -> GroundingVerifier:
    """
    Create a grounding verifier with Maze-specific predicates.
    
    Returns:
        Configured GroundingVerifier for Maze game
    """
    verifier = GroundingVerifier()
    
    # Target relative positions
    verifier.register_predicate(
        "REL_ABOVE",
        lambda s: s.get("target", (0, 0))[1] < s.get("head", (0, 0))[1],
        context="maze"
    )
    verifier.register_predicate(
        "REL_BELOW",
        lambda s: s.get("target", (0, 0))[1] > s.get("head", (0, 0))[1],
        context="maze"
    )
    verifier.register_predicate(
        "REL_LEFT",
        lambda s: s.get("target", (0, 0))[0] < s.get("head", (0, 0))[0],
        context="maze"
    )
    verifier.register_predicate(
        "REL_RIGHT",
        lambda s: s.get("target", (0, 0))[0] > s.get("head", (0, 0))[0],
        context="maze"
    )
    
    # Wall detection
    def _is_wall(state, dx, dy):
        head = state.get("head", (5,5))
        walls = set(state.get("walls", []))
        return (head[0]+dx, head[1]+dy) in walls

    verifier.register_predicate("WALL_AT_UP", lambda s: _is_wall(s, 0, -1), context="maze")
    verifier.register_predicate("WALL_AT_DOWN", lambda s: _is_wall(s, 0, 1), context="maze")
    verifier.register_predicate("WALL_AT_LEFT", lambda s: _is_wall(s, -1, 0), context="maze")
    verifier.register_predicate("WALL_AT_RIGHT", lambda s: _is_wall(s, 1, 0), context="maze")
    
    # Action effects
    verifier.register_action(
        "ACTION_UP", 
        lambda b, a: a.get("head", (0, 0))[1] < b.get("head", (0, 0))[1],
        context="maze"
    )
    verifier.register_action(
        "ACTION_DOWN",
        lambda b, a: a.get("head", (0, 0))[1] > b.get("head", (0, 0))[1],
        context="maze"
    )
    verifier.register_action(
        "ACTION_LEFT",
        lambda b, a: a.get("head", (0, 0))[0] < b.get("head", (0, 0))[0],
        context="maze"
    )
    verifier.register_action(
        "ACTION_RIGHT",
        lambda b, a: a.get("head", (0, 0))[0] > b.get("head", (0, 0))[0],
        context="maze"
    )
    
    return verifier


def _obstacle_at(state: Dict, dx: int, dy: int) -> bool:
    """Check if there's an obstacle at offset from head/player."""
    head = state.get("head", (5, 5))
    obstacles = state.get("obstacles", state.get("walls", []))
    obs_set = set(tuple(o) for o in obstacles)
    target = (head[0] + dx, head[1] + dy)
    width = state.get("width", 10)
    height = state.get("height", 10)
    # Out of bounds counts as obstacle
    if not (0 <= target[0] < width and 0 <= target[1] < height):
        return True
    return target in obs_set


def create_collector_verifier() -> GroundingVerifier:
    """
    Create a grounding verifier with Collector-specific predicates.

    Uses the same predicate vocabulary as Snake/Maze to enable transfer:
    - REL_ABOVE/BELOW/LEFT/RIGHT -> relative to next target item
    - OBSTACLE_UP/DOWN/LEFT/RIGHT -> static obstacle detection
    - ACTION_UP/DOWN/LEFT/RIGHT -> movement effects

    Returns:
        Configured GroundingVerifier for Collector game
    """
    verifier = GroundingVerifier()

    # Target relative positions (next item to collect)
    verifier.register_predicate(
        "REL_ABOVE",
        lambda s: (s.get("target") or s.get("food") or (0, 0))[1] < s.get("head", (0, 0))[1],
        context="collector"
    )
    verifier.register_predicate(
        "REL_BELOW",
        lambda s: (s.get("target") or s.get("food") or (0, 0))[1] > s.get("head", (0, 0))[1],
        context="collector"
    )
    verifier.register_predicate(
        "REL_LEFT",
        lambda s: (s.get("target") or s.get("food") or (0, 0))[0] < s.get("head", (0, 0))[0],
        context="collector"
    )
    verifier.register_predicate(
        "REL_RIGHT",
        lambda s: (s.get("target") or s.get("food") or (0, 0))[0] > s.get("head", (0, 0))[0],
        context="collector"
    )

    # Obstacle detection (maps to Snake's DANGER_* and Maze's WALL_AT_*)
    verifier.register_predicate("OBSTACLE_UP", lambda s: _obstacle_at(s, 0, -1), context="collector")
    verifier.register_predicate("OBSTACLE_DOWN", lambda s: _obstacle_at(s, 0, 1), context="collector")
    verifier.register_predicate("OBSTACLE_LEFT", lambda s: _obstacle_at(s, -1, 0), context="collector")
    verifier.register_predicate("OBSTACLE_RIGHT", lambda s: _obstacle_at(s, 1, 0), context="collector")

    # Action effects
    verifier.register_action(
        "ACTION_UP",
        lambda b, a: a.get("head", (0, 0))[1] < b.get("head", (0, 0))[1],
        context="collector"
    )
    verifier.register_action(
        "ACTION_DOWN",
        lambda b, a: a.get("head", (0, 0))[1] > b.get("head", (0, 0))[1],
        context="collector"
    )
    verifier.register_action(
        "ACTION_LEFT",
        lambda b, a: a.get("head", (0, 0))[0] < b.get("head", (0, 0))[0],
        context="collector"
    )
    verifier.register_action(
        "ACTION_RIGHT",
        lambda b, a: a.get("head", (0, 0))[0] > b.get("head", (0, 0))[0],
        context="collector"
    )

    return verifier


def create_balancer_verifier() -> GroundingVerifier:
    """
    Create verifier for Balancer game.

    Shared predicates with Catcher for transfer:
    - OBJECT_LEFT/RIGHT -> ball position relative to center
    - MOVING_LEFT/RIGHT -> ball velocity direction
    - DANGER_LEFT/RIGHT -> ball near edge
    """
    verifier = GroundingVerifier()

    verifier.register_predicate(
        "OBJECT_LEFT",
        lambda s: s.get("object_x", 0.0) < -0.1,
        context="balancer"
    )
    verifier.register_predicate(
        "OBJECT_RIGHT",
        lambda s: s.get("object_x", 0.0) > 0.1,
        context="balancer"
    )
    verifier.register_predicate(
        "MOVING_LEFT",
        lambda s: s.get("object_vel", 0.0) < -0.005,
        context="balancer"
    )
    verifier.register_predicate(
        "MOVING_RIGHT",
        lambda s: s.get("object_vel", 0.0) > 0.005,
        context="balancer"
    )
    verifier.register_predicate(
        "DANGER_LEFT",
        lambda s: s.get("object_x", 0.0) < -0.7,
        context="balancer"
    )
    verifier.register_predicate(
        "DANGER_RIGHT",
        lambda s: s.get("object_x", 0.0) > 0.7,
        context="balancer"
    )

    return verifier


def create_catcher_verifier() -> GroundingVerifier:
    """
    Create verifier for Catcher game.

    Shared predicates with Balancer for transfer:
    - OBJECT_LEFT/RIGHT -> nearest object relative to paddle
    - MOVING_LEFT/RIGHT -> object horizontal drift
    - DANGER_LEFT/RIGHT -> paddle near wall
    """
    verifier = GroundingVerifier()

    verifier.register_predicate(
        "OBJECT_LEFT",
        lambda s: s.get("object_x", 0.5) < s.get("paddle_x", 0.5) - 0.05,
        context="catcher"
    )
    verifier.register_predicate(
        "OBJECT_RIGHT",
        lambda s: s.get("object_x", 0.5) > s.get("paddle_x", 0.5) + 0.05,
        context="catcher"
    )
    verifier.register_predicate(
        "MOVING_LEFT",
        lambda s: s.get("object_vel", 0.0) < -0.002,
        context="catcher"
    )
    verifier.register_predicate(
        "MOVING_RIGHT",
        lambda s: s.get("object_vel", 0.0) > 0.002,
        context="catcher"
    )
    verifier.register_predicate(
        "DANGER_LEFT",
        lambda s: s.get("paddle_x", 0.5) < 0.15,
        context="catcher"
    )
    verifier.register_predicate(
        "DANGER_RIGHT",
        lambda s: s.get("paddle_x", 0.5) > 0.85,
        context="catcher"
    )

    return verifier
