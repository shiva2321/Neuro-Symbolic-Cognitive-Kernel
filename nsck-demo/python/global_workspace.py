"""
NSCK Global Workspace Module
Phase 3.1: Global Workspace Architecture

Implements the Global Workspace Theory (GWT) architecture where specialized modules
compete for access to a global broadcast channel (consciousness).
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
import logging

# Configure logging
logger = logging.getLogger(__name__)

class WorkspaceModule(ABC):
    """Abstract base class for modules that connect to the Global Workspace."""
    
    @abstractmethod
    def receive_broadcast(self, content: Any):
        """Receive content broadcast from the global workspace."""
        pass

@dataclass
class Coalition:
    """A bundle of information competing for consciousness."""
    source: str
    content: Any
    base_salience: float      # Intrinsic loudness (0.0 - 1.0)
    relevance: float = 0.0    # Match with current context/goal
    affect_match: float = 0.0 # Match with current Drives (e.g. "Food" matches "Hunger")
    
    @property
    def activation(self) -> float:
        return self.base_salience + self.relevance + self.affect_match

class GlobalWorkspace:
    """
    The central executive that manages the 'stream of consciousness'.
    Implementation: LIDA-Lite.
    """
    
    def __init__(self, attention_threshold: float = 0.5):
        self.modules: Dict[str, WorkspaceModule] = {}
        self.workspace_content: Optional[Any] = None
        self.current_winner: Optional[str] = None
        self.attention_threshold = attention_threshold
        self.history: List[Tuple[str, Any, float]] = [] 
        
    def register_module(self, name: str, module: WorkspaceModule):
        self.modules[name] = module
        logger.info(f"[GWT] Registered module: {name}")
        
    def compete(self, proposals: List[Coalition]) -> Optional[Coalition]:
        """
        LIDA Competition Cycle.
        Calculates Activation = Salience + Relevance + Affect.
        """
        if not proposals:
            return None
            
        # 1. Score all coalitions
        ranked = sorted(proposals, key=lambda c: c.activation, reverse=True)
        winner = ranked[0]
        
        # 2. Check threshold
        if winner.activation < self.attention_threshold:
            return None
            
        # 3. Broadcast
        self.current_winner = winner.source
        self.workspace_content = winner.content
        self.broadcast(winner.content)
        
        # Log history
        self.history.append((winner.source, winner.content, winner.activation))
        if len(self.history) > 100: self.history.pop(0)
            
        return winner
        
    def broadcast(self, content: Any):
        """Send content to all registered modules."""
        for name, module in self.modules.items():
            try:
                module.receive_broadcast(content)
            except Exception as e:
                logger.error(f"[GWT] Error broadcasting to {name}: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get current workspace status."""
        return {
            "current_winner": self.current_winner,
            "current_content_type": type(self.workspace_content).__name__ if self.workspace_content else "None",
            "history_len": len(self.history)
        }


