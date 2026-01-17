"""
Core Neuromorphic Engine
Pure Python implementation of biologically-inspired neural network.

Components:
- BioNode: Leaky Integrate-and-Fire neuron with STDP learning
- Synapse: Synaptic connections with eligibility traces
- NeuromorphicNetwork: Sparse graph of interconnected neurons
"""

from .bionode import BioNode
from .synapse import Synapse
from .network import NeuromorphicNetwork

__all__ = ['BioNode', 'Synapse', 'NeuromorphicNetwork']

