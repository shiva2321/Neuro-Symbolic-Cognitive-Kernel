"""T3 — Lifelong Retention Task (weight 0.20).

Teach Task A (20 facts) → measure recall → teach Task B (20 facts) →
re-measure Task A → score = 1 - forgetting_ratio.
"""
from __future__ import annotations
from typing import Any, List, Tuple


_TASK_A = [(f"life_a_{i}", {"domain": "A"}) for i in range(20)]
_TASK_B = [(f"life_b_{i}", {"domain": "B"}) for i in range(20)]


def _recall(sem: Any, facts: List[Tuple]) -> float:
    return sum(1 for name, _ in facts if name in sem.concept_hvs) / len(facts)


def run(engine: Any = None) -> float:
    """Run T3 and return score in [0, 1] (1 = no forgetting)."""
    if engine is None:
        from python.core.reasoning.cognitive_engine import CognitiveEngine
        from python.core.integration.config import NSCKConfig
        engine = CognitiveEngine(config=NSCKConfig(), persistence_path=":memory:")

    sem = engine.semantic_memory

    # Teach Task A
    for name, props in _TASK_A:
        sem.add_concept(name, props)
    recall_a_before = _recall(sem, _TASK_A)

    # Teach Task B
    for name, props in _TASK_B:
        sem.add_concept(name, props)

    # Re-measure Task A
    recall_a_after = _recall(sem, _TASK_A)

    if recall_a_before == 0:
        return 1.0
    forgetting_ratio = max(0.0, (recall_a_before - recall_a_after) / recall_a_before)
    return float(1.0 - forgetting_ratio)
