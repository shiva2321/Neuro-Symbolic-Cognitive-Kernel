"""
NSCK Global Workspace Module
Phase 3.1: Global Workspace Architecture

Implements the Global Workspace Theory (GWT) architecture where specialized modules
compete for access to a global broadcast channel (consciousness).
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List
import logging

# Configure logging
logger = logging.getLogger(__name__)

class WorkspaceModule(ABC):
    """Abstract base class for modules that connect to the Global Workspace."""
    
    @abstractmethod
    def receive_broadcast(self, content: Any):
        """Receive content broadcast from the global workspace."""
        pass

class GlobalWorkspace:
    """
    The central executive that manages the 'stream of consciousness'.
    
    Mechanism:
    1. Specialized modules propose content with a 'salience' (importance) score.
    2. Winner-take-all competition selects the most salient content.
    3. Winner's content is 'broadcast' to ALL modules.
    """
    
    def __init__(self, attention_threshold: float = 0.5):
        self.modules: Dict[str, WorkspaceModule] = {}
        self.workspace_content: Optional[Any] = None
        self.current_winner: Optional[str] = None
        self.attention_threshold = attention_threshold
        self.history: List[Tuple[str, Any, float]] = [] # (source, content, salience)
        
    def register_module(self, name: str, module: WorkspaceModule):
        """Register a module to participate in the workspace."""
        self.modules[name] = module
        logger.info(f"[GWT] Registered module: {name}")
        
    def compete(self, proposals: Dict[str, Tuple[Any, float]]) -> Optional[str]:
        """
        Modules compete for workspace access.
        
        Args:
            proposals: Dict maps 'module_name' -> (content, salience_score)
            
        Returns:
            Name of the winning module, or None if no one crossed threshold.
        """
        if not proposals:
            return None
            
        # 1. Find winner (highest salience)
        winner_name, (content, salience) = max(
            proposals.items(), 
            key=lambda item: item[1][1]
        )
        
        # 2. Check threshold
        if salience >= self.attention_threshold:
            self.workspace_content = content
            self.current_winner = winner_name
            
            # Record history
            self.history.append((winner_name, content, salience))
            if len(self.history) > 100:
                self.history.pop(0)
                
            # 3. Broadcast
            self.broadcast()
            return winner_name
            
        return None
    
    def broadcast(self):
        """Broadcast current workspace content to all registered modules."""
        if self.workspace_content is None:
            return
            
        for name, module in self.modules.items():
            try:
                module.receive_broadcast(self.workspace_content)
            except Exception as e:
                logger.error(f"[GWT] Error broadcasting to {name}: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get current workspace status."""
        return {
            "current_winner": self.current_winner,
            "current_content_type": type(self.workspace_content).__name__ if self.workspace_content else "None",
            "history_len": len(self.history)
        }
