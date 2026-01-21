"""
NCGN Core Module - Unified Neuromorphic Cognitive Graph Network

Combines:
- v6 Learning System (3-Factor Hebbian, eligibility traces, homeostatic normalization)
- v5 Knowledge System (dialogue, queries, staging buffer)

Core Components:
- memory.py: GraphMemory, ConceptNode, Synapse (with trace/stability)
- system1.py: Physics engine with Softmax inhibition
- learning.py: DopamineModulator, ThreeFactorLearner
- planner.py: System 2 Monte Carlo simulation
- runner.py: NCGNAgent for game training
- games/: Corridor and Snake environments
- dialogue.py: DialogueManager for conversation
- ingestion.py: DocumentReader for file ingestion
"""

from .memory import ConceptNode, Synapse, GraphMemory, ClusterType
from .system1 import System1Engine
from .learning import DopamineModulator, ThreeFactorLearner, EpisodicBuffer
from .planner import SimulationSandbox, EpisodicMonteCarlo, System2Controller
from .runner import NCGNAgent, TrainingConfig
from .dialogue import DialogueManager, DialogueState
from .ingestion import DocumentReader
from .staging import StagingBuffer

__all__ = [
    # Data Structures
    'ConceptNode',
    'Synapse', 
    'GraphMemory',
    'ClusterType',
    # Physics
    'System1Engine',
    # Learning
    'DopamineModulator',
    'ThreeFactorLearner',
    'EpisodicBuffer',
    # Planning
    'SimulationSandbox',
    'EpisodicMonteCarlo',
    'System2Controller',
    # Agent
    'NCGNAgent',
    'TrainingConfig',
    # Knowledge & Dialogue
    'DialogueManager',
    'DialogueState',
    'DocumentReader',
    'StagingBuffer',
]
