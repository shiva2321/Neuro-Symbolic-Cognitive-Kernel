"""
NCGN Cortex Package - High-Level Cognitive Functions

The "Pre-Frontal Cortex" layer that manages:
- Document ingestion (ingestion.py)
- Staging memory / Hippocampus (staging.py)
- Dialogue state management (dialogue.py)

Phase 3: Interactive Cognition & Knowledge Acquisition
"""

from .ingestion import Triple, text_to_triples, DocumentReader
from .staging import StagingBuffer, MergeResult
from .dialogue import DialogueState, DialogueContext, DialogueManager

__all__ = [
    # Ingestion
    'Triple',
    'text_to_triples',
    'DocumentReader',
    # Staging
    'StagingBuffer',
    'MergeResult',
    # Dialogue
    'DialogueState',
    'DialogueContext',
    'DialogueManager',
]
