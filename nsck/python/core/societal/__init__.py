"""
Societal Hypervector Knowledge Representation — NSCK V26.

This package implements the Societal HyperVector (SHV) architecture, where
individual knowledge atoms (LivingHyperVectors) form dynamic communities,
bonds, and hierarchies analogous to societal structures.

Key components
--------------
LivingHyperVector
    An activated hypervector with lifecycle metadata (birth epoch, activation
    level, bond strengths) and societal relationships (peers, roles).

SocietyManager
    Orchestrates a collection of LivingHyperVectors: maintains the domain
    hierarchy, forms/dissolves bonds based on similarity thresholds, runs
    Leiden-style community detection, and exposes multi-resolution clusters.

SocietalContextRouter
    Integrates societal knowledge into NSCKSubstrate.process() by routing
    each incoming query through the active society and returning a
    ``societal_context`` dict attached to every SubstrateResult.

Usage
-----
>>> from python.core.societal import LivingHyperVector, SocietyManager
>>> from python.core.societal import SocietalContextRouter
"""

from python.core.societal.living_hypervector import LivingHyperVector
from python.core.societal.society_manager import SocietyManager
from python.core.societal.societal_context_router import SocietalContextRouter

__all__ = [
    "LivingHyperVector",
    "SocietyManager",
    "SocietalContextRouter",
]
