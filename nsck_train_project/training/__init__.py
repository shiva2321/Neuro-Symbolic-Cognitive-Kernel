"""
NSCK Training System
===================

Components for training the NSCK AI system:
- Web Data Collector (fetch training data from reliable sources)
- Training Pipeline (complete training workflow)
"""

__version__ = "2.0.0"

from .web_data_collector import WebDataCollector
from .training_pipeline import ExtendedTrainingPipeline

__all__ = ['WebDataCollector', 'ExtendedTrainingPipeline']
