"""
NCGN v6.0 Game Interface - Egocentric Sensory Encoding

Implements the bridge between game environments and the graph:
- EgocentricEncoder: Converts game state to sparse relative activations
- GameInterface: Abstract base class for game environments
- ValenceEstimator: Converts game events to valence signals

Design principle: Agent-relative sparse encoding for generalization.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

from ..memory import GraphMemory, ClusterType


class Direction(Enum):
    """Relative directions for egocentric encoding."""
    AHEAD = "ahead"
    LEFT = "left"
    RIGHT = "right"
    BEHIND = "behind"


@dataclass
class SensoryInput:
    """Structured sensory input from a game."""
    vision: Dict[Direction, str]  # What's in each direction
    proprioception: Dict[str, float]  # Internal state (hunger, health, etc.)
    recent_action: Optional[str]
    reward: float


class EgocentricEncoder:
    """
    Converts absolute game state to agent-relative sparse activations.
    
    SPARSE ENCODING RATIONALE (explicit comment as required):
    --------------------------------------------------------
    Rather than encoding absolute positions (x=5, y=3), we encode
    relative observations (food_ahead, wall_left). This provides:
    
    1. GENERALIZATION: Pattern "food_ahead → move_forward" works
       regardless of where food actually is on the grid.
    
    2. SPARSITY: Only encode what's relevant. A 100x100 grid would
       need 10,000 position nodes with absolute encoding. With
       relative encoding, we need ~20 direction×object nodes.
    
    3. EFFICIENCY: Sparse activation = less computation per tick.
    
    The encoding creates nodes like:
    - "vision_food_ahead" (food visible ahead)
    - "vision_wall_left" (wall to the left)
    - "proprioception_hungry" (internal hunger state)
    """
    
    # Sensory node prefixes
    VISION_PREFIX = "vision"
    PROPRIO_PREFIX = "proprio"
    
    def __init__(self, memory: GraphMemory):
        self.memory = memory
        
        # Cache of created sensory nodes
        self.sensory_nodes: Dict[str, str] = {}
    
    def encode(self, sensory_input: SensoryInput) -> Dict[str, float]:
        """
        Encode game state into sparse node activations.
        
        Args:
            sensory_input: Structured input from game
        
        Returns:
            Dict mapping node_id → energy to inject
        """
        activations: Dict[str, float] = {}
        
        # Encode vision (what's in each direction)
        for direction, object_type in sensory_input.vision.items():
            if object_type:
                node_id = self._get_or_create_node(
                    f"{self.VISION_PREFIX}_{object_type}_{direction.value}",
                    ClusterType.SENSORY
                )
                activations[node_id] = 1.0
        
        # Encode proprioception (internal states)
        for state_name, intensity in sensory_input.proprioception.items():
            if intensity > 0.1:
                node_id = self._get_or_create_node(
                    f"{self.PROPRIO_PREFIX}_{state_name}",
                    ClusterType.SENSORY
                )
                activations[node_id] = intensity
        
        return activations
    
    def _get_or_create_node(self, node_id: str, cluster: ClusterType) -> str:
        """Get or create a sensory node."""
        if node_id not in self.sensory_nodes:
            self.memory.add_node(node_id, cluster=cluster)
            self.sensory_nodes[node_id] = node_id
        return node_id


class GameInterface(ABC):
    """
    Abstract base class for game environments.
    
    Each game implements:
    - reset(): Start a new episode
    - step(action): Take an action, return reward
    - get_sensory(): Get current state as SensoryInput
    - get_possible_actions(): List of valid action node IDs
    """
    
    @abstractmethod
    def reset(self) -> SensoryInput:
        """Reset the game and return initial observation."""
        pass
    
    @abstractmethod
    def step(self, action: str) -> Tuple[SensoryInput, float, bool]:
        """
        Take an action in the game.
        
        Args:
            action: Action node ID
        
        Returns:
            (observation, reward, done)
        """
        pass
    
    @abstractmethod
    def get_sensory(self) -> SensoryInput:
        """Get current sensory state."""
        pass
    
    @abstractmethod
    def get_possible_actions(self) -> List[str]:
        """Get list of possible action node IDs."""
        pass
    
    @abstractmethod
    def render(self) -> str:
        """Render current state as ASCII for debugging."""
        pass


class ValenceEstimator:
    """
    Converts game events to valence (reward) signals.
    
    This is the bridge between game rewards and the dopamine
    modulator. It can transform raw rewards with:
    - Scaling
    - Clipping
    - Shaping (intermediate rewards)
    """
    
    def __init__(
        self,
        reward_scale: float = 1.0,
        survival_bonus: float = 0.01,
        death_penalty: float = -1.0
    ):
        self.reward_scale = reward_scale
        self.survival_bonus = survival_bonus
        self.death_penalty = death_penalty
    
    def estimate(
        self,
        game_reward: float,
        done: bool,
        survived: bool = True
    ) -> float:
        """
        Estimate valence from game state.
        
        Args:
            game_reward: Raw reward from game
            done: Whether episode ended
            survived: Whether agent survived (False = death)
        
        Returns:
            Valence signal for dopamine modulator
        """
        valence = game_reward * self.reward_scale
        
        if done and not survived:
            valence += self.death_penalty
        elif not done:
            valence += self.survival_bonus
        
        return valence
