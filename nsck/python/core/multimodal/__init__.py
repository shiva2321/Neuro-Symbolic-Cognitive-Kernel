"""
NSCK Multimodal Module
======================
Text, image, audio, video → unified VSA hypervector representations.
No neural networks — all classical signal processing.
"""

from python.core.multimodal.multimodal_processor import (
    MultimodalProcessor,
    MultimodalInput,
    ProcessedInput,
    ModalityResult,
)
from python.core.multimodal.image_generator import (
    ImageGenerator,
    GenerationConfig,
    VisualFeatures,
)

__all__ = [
    "MultimodalProcessor",
    "MultimodalInput",
    "ProcessedInput",
    "ModalityResult",
    "ImageGenerator",
    "GenerationConfig",
    "VisualFeatures",
]
