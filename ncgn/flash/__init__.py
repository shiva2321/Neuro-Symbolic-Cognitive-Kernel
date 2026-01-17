"""
Flash-native binary neuromorphic implementation.

Memory-mapped, production-grade system for large-scale networks.
"""

from ncgn.flash.block_manager import BlockManager
from ncgn.flash.flash_manager import FlashManager
from ncgn.flash.flash_colony import FlashColony
from ncgn.flash.ncgn_anatomy import NcgnAnatomy, BrainRegion, Projection
from ncgn.flash.neuro_kernel import NeuroKernel
from ncgn.flash.flash_dynamic import FlashDynamic

__all__ = [
    "BlockManager",
    "FlashManager",
    "FlashColony",
    "NcgnAnatomy",
    "BrainRegion",
    "Projection",
    "NeuroKernel",
    "FlashDynamic",
]

