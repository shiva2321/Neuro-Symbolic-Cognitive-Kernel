"""
Universal Game Interface for NSCK Cognitive Agent

This module defines a standard interface that ANY game can implement
to work with the NSCK cognitive learning agent. 

The interface ensures that:
1. Any game can be plugged into the learning system
2. Transfer learning works across all games
3. Logging and analysis are consistent
4. Adding new games requires minimal code

To add a new game:
1. Implement the GameEnvironment interface
2. Register abstract concepts for transfer learning
3. That's it! The agent can now learn your game.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum


@dataclass
class GameState:
    """
    Universal game state representation.
    
    All games must provide this information to the agent.
    """
    # Core state info
    done: bool  # Is the game over?
    score: float  # Current score/reward
    steps: int  # Steps taken so far
    
    # State representation for learning
    predicates: List[str]  # Logical predicates describing state
    features: Dict[str, Any]  # Numeric/categorical features
    
    # For visualization/debugging
    visual_repr: Optional[str] = None  # ASCII/text representation
    
    # Additional game-specific data
    extra: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.extra is None:
            self.extra = {}


@dataclass
class GameAction:
    """Universal action representation."""
    name: str  # Action identifier (e.g., "MOVE_UP", "BUILD_TOWER")
    params: Dict[str, Any] = None  # Optional parameters
    
    def __post_init__(self):
        if self.params is None:
            self.params = {}
    
    def __str__(self):
        if self.params:
            return f"{self.name}({self.params})"
        return self.name


@dataclass
class GameResult:
    """Result of taking an action in the game."""
    new_state: GameState
    reward: float
    done: bool
    info: Dict[str, Any]  # Additional information (reason, etc.)


class GameEnvironment(ABC):
    """
    Abstract base class for all game environments.
    
    Any game that implements this interface can be learned by the
    NSCK cognitive agent with full transfer learning support.
    """
    
    @abstractmethod
    def reset(self) -> GameState:
        """
        Reset the game to initial state.
        
        Returns:
            Initial game state
        """
        pass
    
    @abstractmethod
    def get_available_actions(self, state: Optional[GameState] = None) -> List[GameAction]:
        """
        Get list of actions available in current state.
        
        Args:
            state: Optional state to query (uses current state if None)
        
        Returns:
            List of available actions
        """
        pass
    
    @abstractmethod
    def step(self, action: GameAction) -> GameResult:
        """
        Execute an action and return the result.
        
        Args:
            action: Action to execute
        
        Returns:
            GameResult with new state, reward, done flag, and info
        """
        pass
    
    @abstractmethod
    def get_current_state(self) -> GameState:
        """
        Get the current game state.
        
        Returns:
            Current state
        """
        pass
    
    @abstractmethod
    def get_abstract_concepts(self) -> Dict[str, List[str]]:
        """
        Get abstract concept mappings for transfer learning.
        
        Returns:
            Dictionary mapping abstract concepts to concrete game concepts
            Example: {
                "AGENT": ["PLAYER", "CHARACTER"],
                "RESOURCE": ["FOOD", "GOLD", "ENERGY"],
                "THREAT": ["ENEMY", "OBSTACLE"],
                "GOAL": ["EXIT", "TREASURE", "TARGET"]
            }
        """
        pass
    
    @abstractmethod
    def get_game_name(self) -> str:
        """Get the name of this game (for logging/identification)."""
        pass
    
    # Optional methods with default implementations
    
    def render(self) -> str:
        """
        Render the game state as text/ASCII.
        
        Returns:
            Visual representation of current state
        """
        state = self.get_current_state()
        if state.visual_repr:
            return state.visual_repr
        return f"Game: {self.get_game_name()}\nScore: {state.score}\nSteps: {state.steps}"
    
    def get_state_description(self) -> str:
        """
        Get human-readable description of current state.
        
        Returns:
            Text description
        """
        state = self.get_current_state()
        return f"State with {len(state.predicates)} predicates, score {state.score}"
    
    def clone(self) -> 'GameEnvironment':
        """
        Create a copy of the game environment (for planning/simulation).
        
        Returns:
            New instance of the game
        """
        raise NotImplementedError("This game does not support cloning")
    
    def get_max_steps(self) -> Optional[int]:
        """
        Get maximum steps before timeout (if applicable).
        
        Returns:
            Max steps or None if unlimited
        """
        return None
    
    def seed(self, seed: int):
        """
        Set random seed for reproducibility.
        
        Args:
            seed: Random seed value
        """
        pass  # Optional implementation


class GameRegistry:
    """
    Registry for all available game environments.
    
    Allows dynamic discovery and instantiation of games.
    """
    
    _games: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, game_class: type):
        """Register a game environment class."""
        if not issubclass(game_class, GameEnvironment):
            raise ValueError(f"{game_class} must inherit from GameEnvironment")
        cls._games[name] = game_class
    
    @classmethod
    def get(cls, name: str) -> type:
        """Get a registered game class by name."""
        if name not in cls._games:
            raise ValueError(f"Game '{name}' not registered. Available: {list(cls._games.keys())}")
        return cls._games[name]
    
    @classmethod
    def list_games(cls) -> List[str]:
        """List all registered games."""
        return list(cls._games.keys())
    
    @classmethod
    def create(cls, name: str, **kwargs) -> GameEnvironment:
        """Create an instance of a registered game."""
        game_class = cls.get(name)
        return game_class(**kwargs)


# Decorator for easy game registration
def register_game(name: str):
    """Decorator to register a game environment."""
    def decorator(game_class):
        GameRegistry.register(name, game_class)
        return game_class
    return decorator


# Common abstract concepts for transfer learning
COMMON_ABSTRACTIONS = {
    # Agent/Player concepts
    "AGENT": ["PLAYER", "CHARACTER", "AVATAR", "HERO"],
    "AGENT_POSITION": ["PLAYER_POS", "CHARACTER_POS", "LOCATION"],
    
    # Resource concepts
    "RESOURCE": ["FOOD", "WATER", "ENERGY", "GOLD", "MANA", "HEALTH_PACK"],
    "RESOURCE_LOW": ["LOW_FOOD", "LOW_WATER", "LOW_ENERGY", "LOW_GOLD"],
    "RESOURCE_CRITICAL": ["CRITICAL_HUNGER", "CRITICAL_THIRST", "CRITICAL_ENERGY"],
    
    # Threat concepts
    "THREAT": ["ENEMY", "OBSTACLE", "HAZARD", "DANGER", "TRAP"],
    "THREAT_NEARBY": ["ENEMY_NEARBY", "DANGER_CLOSE", "HAZARD_NEAR"],
    "THREAT_IMMEDIATE": ["ENEMY_VERY_CLOSE", "DANGER_IMMINENT"],
    
    # Goal concepts
    "GOAL": ["EXIT", "TARGET", "OBJECTIVE", "TREASURE", "CHECKPOINT"],
    "GOAL_NEARBY": ["EXIT_NEARBY", "TARGET_CLOSE", "OBJECTIVE_NEAR"],
    
    # Spatial concepts
    "ABOVE": ["UP", "NORTH", "TOP"],
    "BELOW": ["DOWN", "SOUTH", "BOTTOM"],
    "LEFT": ["WEST", "LEFT_SIDE"],
    "RIGHT": ["EAST", "RIGHT_SIDE"],
    
    # Action concepts
    "MOVE": ["WALK", "GO", "NAVIGATE", "TRAVEL"],
    "COLLECT": ["GATHER", "PICKUP", "TAKE", "ACQUIRE"],
    "ATTACK": ["FIGHT", "DESTROY", "ELIMINATE", "DAMAGE"],
    "BUILD": ["CREATE", "CONSTRUCT", "MAKE", "CRAFT"],
    "WAIT": ["STAY", "REST", "IDLE", "PAUSE"],
}


def get_standard_predicate(concept: str, game_specific: str) -> str:
    """
    Create a standard predicate name that can be mapped for transfer learning.
    
    Args:
        concept: Abstract concept (e.g., "RESOURCE", "THREAT")
        game_specific: Game-specific name (e.g., "FOOD", "ENEMY")
    
    Returns:
        Standardized predicate name
    """
    return f"{concept}_{game_specific}"


def create_spatial_predicates(
    agent_pos: Tuple[int, int],
    entity_positions: Dict[str, List[Tuple[int, int]]],
    max_distance: int = 5
) -> List[str]:
    """
    Helper to create spatial relationship predicates.
    
    Args:
        agent_pos: Agent position (y, x)
        entity_positions: Dict of entity type -> list of positions
        max_distance: Maximum distance to consider "nearby"
    
    Returns:
        List of spatial predicates
    """
    predicates = []
    ay, ax = agent_pos
    
    for entity_type, positions in entity_positions.items():
        for ey, ex in positions:
            dist = abs(ey - ay) + abs(ex - ax)  # Manhattan distance
            
            # Distance predicates
            if dist <= 1:
                predicates.append(f"{entity_type}_ADJACENT")
            if dist <= 3:
                predicates.append(f"{entity_type}_CLOSE")
            if dist <= max_distance:
                predicates.append(f"{entity_type}_NEARBY")
            
            # Directional predicates (for closest entity)
            if dist <= max_distance:
                dy = ey - ay
                dx = ex - ax
                
                if abs(dy) > abs(dx):  # Vertical is dominant
                    if dy < 0:
                        predicates.append(f"{entity_type}_ABOVE")
                    else:
                        predicates.append(f"{entity_type}_BELOW")
                else:  # Horizontal is dominant
                    if dx < 0:
                        predicates.append(f"{entity_type}_LEFT")
                    else:
                        predicates.append(f"{entity_type}_RIGHT")
    
    return predicates


# Example helper for common game patterns
class TurnBasedGame(GameEnvironment):
    """Base class for turn-based games with discrete steps."""
    
    def __init__(self, max_steps: int = 1000):
        self.max_steps_value = max_steps
        self.current_step = 0
    
    def get_max_steps(self) -> Optional[int]:
        return self.max_steps_value
    
    def is_timeout(self) -> bool:
        """Check if game has timed out."""
        return self.current_step >= self.max_steps_value


class RealtimeGame(GameEnvironment):
    """Base class for real-time games with continuous time."""
    
    def __init__(self, time_limit: float = 60.0):
        self.time_limit = time_limit
        self.elapsed_time = 0.0
    
    def get_max_steps(self) -> Optional[int]:
        return None  # Real-time games don't have discrete steps
    
    def is_timeout(self) -> bool:
        """Check if time limit exceeded."""
        return self.elapsed_time >= self.time_limit


if __name__ == "__main__":
    print("Universal Game Interface for NSCK Cognitive Agent")
    print("=" * 60)
    print()
    print("This module provides a standard interface that ANY game")
    print("can implement to work with the NSCK learning agent.")
    print()
    print("Features:")
    print("  ✓ Standard state/action representation")
    print("  ✓ Transfer learning support")
    print("  ✓ Easy game registration")
    print("  ✓ Consistent logging")
    print()
    print("To add a new game:")
    print("  1. Inherit from GameEnvironment")
    print("  2. Implement required methods")
    print("  3. Use @register_game decorator")
    print("  4. Done! Agent can now learn your game.")
    print()
    print(f"Registered games: {GameRegistry.list_games()}")
