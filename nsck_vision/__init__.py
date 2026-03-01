"""nsck_vision — NSCK Universal Pretrained Model Absorber (NSCK-UPMA).

Version 1.0.0
"""

__version__ = "1.0.0"

from nsck_vision.system import NSCKVisionSystem
from nsck_vision.registry.model_registry import ModelRegistry
from nsck_vision.reference.reference_runner import ReferenceModelRunner

__all__ = [
    "NSCKVisionSystem",
    "ModelRegistry", 
    "ReferenceModelRunner",
    "__version__",
]
