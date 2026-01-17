"""
Storage Layer: Binary Persistence for Neuromorphic Networks
Memory-mapped file I/O for efficient storage and retrieval of neural networks.

Components:
- FlashManager: Low-level binary file operations
- FlashColony: Memory-mapped neural network storage
- FlashDynamic: Dynamic synapse allocation
- BlockManager: Block-based memory management
- NeuroKernel: Advanced kernel-level operations
"""

def __getattr__(name):
    """Lazy loading of storage classes"""
    if name == 'FlashBrain':
        from .flash_manager import FlashBrain
        return FlashBrain
    elif name == 'FlashColony':
        from .flash_colony import FlashColony
        return FlashColony
    elif name == 'NeuroKernel':
        from .neuro_kernel import NeuroKernel
        return NeuroKernel
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ['FlashBrain', 'FlashColony', 'NeuroKernel']

