from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

class Direction(str, Enum):
    AHEAD = "ahead"
    BEHIND = "behind"
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"

@dataclass
class SensoryInput:
    vision: Dict[str, str]  # Direction -> Object
    proprioception: Dict[str, float]
    recent_action: Optional[str]
    reward: float

class ValenceEstimator:
    """Helper for estimating valence/reward."""
    pass

class GameInterface:
    """Base interface for games."""
    
    def reset(self) -> SensoryInput:
        raise NotImplementedError
    
    def step(self, action: str) -> Tuple[SensoryInput, float, bool]:
        raise NotImplementedError
    
    def get_sensory(self) -> SensoryInput:
        raise NotImplementedError
        
    def get_possible_actions(self) -> List[str]:
        raise NotImplementedError
        
    def render(self) -> str:
        return ""
    
    def get_stats(self) -> Dict:
        return {}
