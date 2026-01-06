"""
Base Module Interface
Defines the interface that all specialized modules must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from enum import Enum

from core.graph_network import GraphNetwork, DomainType


class ModuleType(Enum):
    """Types of specialized modules"""
    TEXT = "text"
    MATH = "math"
    CODE = "code"


class BaseModule(ABC):
    """
    Abstract base class for all specialized modules.
    Each module manages its own graph network and provides specialized processing.
    """

    def __init__(self, module_type: ModuleType, name: str):
        self.module_type = module_type
        self.name = name
        self.graph = GraphNetwork(name=f"{name}_graph")
        self.is_trained = False
        self.confidence_threshold = 0.5

    @abstractmethod
    def can_handle(self, query: str) -> float:
        """
        Determine if this module can handle the query.

        Args:
            query: Input query string

        Returns:
            Confidence score (0.0 to 1.0) indicating ability to handle
        """
        pass

    @abstractmethod
    def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a query and return results.

        Args:
            query: Input query
            context: Optional context information

        Returns:
            Dictionary containing response and metadata
        """
        pass

    @abstractmethod
    def train(self, training_data: List[str], **kwargs) -> Dict[str, Any]:
        """
        Train the module on provided data.

        Args:
            training_data: List of training examples
            **kwargs: Additional training parameters

        Returns:
            Training statistics
        """
        pass

    def get_statistics(self) -> Dict[str, Any]:
        """Get module statistics"""
        return {
            'module_type': self.module_type.value,
            'name': self.name,
            'is_trained': self.is_trained,
            'graph_stats': self.graph.get_statistics()
        }

    def save(self, filepath: str):
        """Save module state"""
        self.graph.save(filepath, format='pickle')

    def load(self, filepath: str):
        """Load module state"""
        self.graph = GraphNetwork.load(filepath, format='pickle')
        self.is_trained = True

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', trained={self.is_trained})"

