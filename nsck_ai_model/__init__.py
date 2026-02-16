"""
NSCK AI Model
=============
A genuine text-and-image AI model built on Vector Symbolic Architecture (VSA).
No transformers, no heavy matrix multiplication — pure hypervector algebra.
Zero hardcoded patterns — everything is learned from training data.

Quick Start::

    from nsck_ai_model.ai_engine import NSCKAIEngine
    from nsck_ai_model.data_pipeline import DataPipeline

    engine = NSCKAIEngine()
    pipeline = DataPipeline(engine)
    pipeline.train_from_seed()
    result = engine.chat("What is the capital of France?")
    print(result['response'])
    print(result['trace'])  # full glass-box reasoning chain
"""

__version__ = "1.0.0"
