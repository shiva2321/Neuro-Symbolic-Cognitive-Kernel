"""
NSCK Core System
===============

This module contains the core components of the NSCK AI system:
- Neural Chat Backend (base trainable system)
- Improved Backend (enhanced with adaptive retrieval, episodic memory, multi-hop reasoning)
- Adaptive Retrieval (handles concept dilution)
- Enhanced Concept Extraction (improved concept quality)
- Production NLG (natural language generation)
- Image Understanding (multimodal capabilities)
"""

__version__ = "2.0.0"
__author__ = "NSCK Team"

from .neural_chat_backend import TrainableChatBackend
from .improved_backend import ImprovedBackend

__all__ = ['TrainableChatBackend', 'ImprovedBackend']
