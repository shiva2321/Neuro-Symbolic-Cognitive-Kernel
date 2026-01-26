"""
NCGN v7: Data-Oriented Neuro-Symbolic Cognitive Architecture

A high-performance cognitive engine using:
- Rustworkx for graph topology
- Sparse CSR matrices for propagation dynamics
- Local LLMs for structured reasoning
- 3-Factor Hebbian learning with reward modulation

Components:
- topology: Graph structure with integer indexing
- state: Structure-of-Arrays cognitive state
- engine: Vectorized propagation dynamics
- learner: 3-Factor Hebbian plasticity
- embeddings: Semantic vector layer
- reasoner: LLM-based System 2
- brain: Unified coordinator
- persistence: Save/load brain state
- logger: Activity logging
"""

from .config import Config
from .topology import GraphTopology, IndexRegistry
from .state import CognitiveState
from .engine import PropagationEngine
from .learner import HebbianLearner
from .brain import Brain
from .persistence import BrainPersistence
from .logger import BrainLogger

__version__ = "7.0.0"
__all__ = [
    "Config",
    "GraphTopology",
    "IndexRegistry", 
    "CognitiveState",
    "PropagationEngine",
    "HebbianLearner",
    "Brain",
    "BrainPersistence",
    "BrainLogger",
]

