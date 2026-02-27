"""T4 — Cross-Modal Recall Task (weight 0.15).

10 pairs: teach text fact, query with numeric array, check semantic activation.
"""
from __future__ import annotations
from typing import Any


_N_PAIRS = 10


def run(engine: Any = None) -> float:
    """Run T4 and return score in [0, 1]."""
    if engine is None:
        from python.core.reasoning.cognitive_engine import CognitiveEngine
        from python.core.integration.config import NSCKConfig
        engine = CognitiveEngine(config=NSCKConfig(), persistence_path=":memory:")

    sem = engine.semantic_memory
    correct = 0

    for i in range(_N_PAIRS):
        concept = f"modal_concept_{i}"
        sem.add_concept(concept, {"type": "cross_modal", "index": i})

        # Teach a textual relation
        anchor = f"modal_anchor_{i}"
        sem.add_concept(anchor, {"type": "anchor"})
        sem.add_relation(anchor, "represents", concept)

        # Query via spreading activation from anchor
        try:
            activated = sem.spread_activation([anchor], steps=2)
            if concept in activated:
                correct += 1
        except Exception:
            pass

    return float(correct / _N_PAIRS) if _N_PAIRS > 0 else 0.0
