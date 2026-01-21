"""
NCGN Core Module - Neuromorphic Cognitive Graph Network

A cognitive control system operating on sparse, event-driven graph dynamics.
"""

from .memory import ConceptNode, Synapse, GraphMemory, EventSchema
from .system1 import System1Engine
from .bridge import SurpriseMonitor
from .system2 import System2Controller

__all__ = [
    'ConceptNode',
    'Synapse', 
    'GraphMemory',
    'EventSchema',
    'System1Engine',
    'SurpriseMonitor',
    'System2Controller',
]
