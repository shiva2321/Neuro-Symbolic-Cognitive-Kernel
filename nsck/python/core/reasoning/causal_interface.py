"""
NSCK Causal Query Interface
===========================
Defines the standard interface for causal reasoning queries.
Decouples the Dialogue Manager from the concrete CausalReasoner.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class Explanation:
    """Standard format for a causal explanation."""
    query: str
    cause: str
    effect: str
    chain: List[str]  # Human-readable chain (e.g. "A -> B -> C")
    confidence: float
    text: str # Final realized text

@dataclass
class Prediction:
    """Standard format for a causal prediction."""
    query: str
    action: str
    predicted_outcome: str
    confidence: float
    text: str

class CausalQueryService(ABC):
    """Abstract interface for causal reasoning capabilities."""
    
    @abstractmethod
    def explain_why(self, effect: str, context: Optional[str] = None) -> Explanation:
        """Explain why an effect occurred."""
        pass
        
    @abstractmethod
    def predict_what_if(self, action: str, state: Dict[str, Any], context: Optional[str] = None) -> Prediction:
        """Predict what would happen if an action were taken."""
        pass

class MockCausalService(CausalQueryService):
    """Mock implementation for testing and fallback."""
    
    def explain_why(self, effect: str, context: Optional[str] = None) -> Explanation:
        return Explanation(
            query=f"Why {effect}?",
            cause="unknown_cause",
            effect=effect,
            chain=[],
            confidence=0.0,
            text=f"I cannot explain why {effect} happened yet."
        )
        
    def predict_what_if(self, action: str, state: Dict[str, Any], context: Optional[str] = None) -> Prediction:
        return Prediction(
            query=f"What if {action}?",
            action=action,
            predicted_outcome="unknown_outcome",
            confidence=0.0,
            text=f"I don't know what would happen if I {action}."
        )
