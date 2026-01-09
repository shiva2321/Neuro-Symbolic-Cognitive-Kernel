"""
Neuromorphic Cognitive Graph Network (NCGN)

A next-generation graph-based neural network combining:
- Linguistic graph substrate with semantic connectivity
- Spiking neural network dynamics
- Dual-system cognitive architecture
- Continual graph learning
- Hardware-optimized for edge deployment
"""

__version__ = "0.1.0"
__author__ = "NCGN Development Team"

# Phase 1: Linguistic Graph Substrate ✓
from .linguistic_graph import LinguisticGraph, GraphBuilder
from .graph_embeddings import GraphEmbedding, StructuralEncoder

# Phase 2: Neuromorphic Core ✓
from .spiking_neurons import (
    LIFNeuron, IzhikevichNeuron, SpikingLayer,
    PoissonEncoder, RateEncoder, TemporalEncoder,
    SpikingGraphConvolution
)
from .stdp_learning import (
    STDPLearning, TripleSTDP, RewardModulatedSTDP,
    HomeostaticSTDP, WeightNormalization
)

# Phase 3: Dual-System Architecture ✓
from .graph_transformer import (
    GraphTransformer, SparseGraphTransformer,
    GraphMultiHeadAttention, GraphPooling
)
from .symbolic_reasoner import (
    SymbolicReasoner, KnowledgeBase,
    Fact, Rule, ForwardChaining
)
from .dual_system import (
    DualSystemArchitecture,
    System1Module, System2Module,
    IntegrationLayer
)

# Phase 4: Continual Learning (to be implemented)
# from .continual_learning import ContinualLearner
# from .memory_management import MemoryManager

# Phase 5: Hardware Optimization (to be implemented)
# from .memory_optimizer import MemoryOptimizer
# from .training_scheduler import TrainingScheduler

__all__ = [
    # Phase 1
    'LinguisticGraph',
    'GraphBuilder',
    'GraphEmbedding',
    'StructuralEncoder',
    # Phase 2
    'LIFNeuron',
    'IzhikevichNeuron',
    'SpikingLayer',
    'PoissonEncoder',
    'RateEncoder',
    'TemporalEncoder',
    'SpikingGraphConvolution',
    'STDPLearning',
    'TripleSTDP',
    'RewardModulatedSTDP',
    'HomeostaticSTDP',
    'WeightNormalization',
    # Phase 3
    'GraphTransformer',
    'SparseGraphTransformer',
    'GraphMultiHeadAttention',
    'GraphPooling',
    'SymbolicReasoner',
    'KnowledgeBase',
    'Fact',
    'Rule',
    'ForwardChaining',
    'DualSystemArchitecture',
    'System1Module',
    'System2Module',
    'IntegrationLayer',
]

