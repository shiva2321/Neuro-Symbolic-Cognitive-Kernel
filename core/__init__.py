"""
Core Neuromorphic Engine
Pure Python implementation of biologically-inspired neural network.

LEGACY COMPONENTS (being phased out):
- BioNode: Old monolithic neuron model
- NeuromorphicNetwork: Old synchronous network

NEW COMPONENTS (Phase 1 refactor):
- Spike, WeightedSpike: Interface contracts for spike events
- EventQueue: Temporal ordering with causality guarantees
- Dispatcher: Fan-out routing of spikes
- Neuron: Refactored neuron with separated concerns
- Plasticity: Learning rules and neuromodulation
- GraphStore: Topology persistence
- Synapse: Updated with transmit() interface
"""

# Legacy imports (for backward compatibility)
from .bionode import BioNode
from .synapse import Synapse
from .network import NeuromorphicNetwork

# New architecture components
from .spike import Spike, WeightedSpike
from .event_queue import EventQueue
from .dispatcher import Dispatcher
from .neuron import Neuron
from .plasticity import PlasticityController, STDPRule, HomeostasisRule, LearningGate
from .graph_store import GraphStore
from .orchestrator import Orchestrator, SimulationState
from .metrics import MetricsCollector, SpikeProbe, WeightProbe, StateProbe, ProbeSnapshot

__all__ = [
    # Legacy
    'BioNode',
    'NeuromorphicNetwork',
    # New
    'Spike',
    'WeightedSpike',
    'EventQueue',
    'Dispatcher',
    'Neuron',
    'PlasticityController',
    'STDPRule',
    'HomeostasisRule',
    'LearningGate',
    'GraphStore',
    'Synapse',
    'Orchestrator',
    'SimulationState',
    'MetricsCollector',
    'SpikeProbe',
    'WeightProbe',
    'StateProbe',
    'ProbeSnapshot',
]
