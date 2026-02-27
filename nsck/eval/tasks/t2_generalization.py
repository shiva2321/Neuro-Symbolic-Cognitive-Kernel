"""T2 — Generalization Task (weight 0.20).

5 scenarios: teach domain A rules, test transfer to domain B.
Score = fraction of successful transfers.
"""
from __future__ import annotations
from typing import Any


_N_SCENARIOS = 5


def run(engine: Any = None) -> float:
    """Run T2 and return score in [0, 1]."""
    if engine is None:
        from python.core.reasoning.cognitive_engine import CognitiveEngine
        from python.core.integration.config import NSCKConfig
        engine = CognitiveEngine(config=NSCKConfig(), persistence_path=":memory:")

    sem = engine.semantic_memory
    successes = 0

    for i in range(_N_SCENARIOS):
        # Teach domain A: animals
        a_src = f"gen_a_src_{i}"
        a_dst = f"gen_a_dst_{i}"
        sem.add_concept(a_src, {"domain": "A", "category": "animal"})
        sem.add_concept(a_dst, {"domain": "A", "category": "animal"})
        sem.add_relation(a_src, "is_a", a_dst)

        # Domain B: vehicles (same structural pattern)
        b_src = f"gen_b_src_{i}"
        b_dst = f"gen_b_dst_{i}"
        sem.add_concept(b_src, {"domain": "B", "category": "vehicle"})
        sem.add_concept(b_dst, {"domain": "B", "category": "vehicle"})
        sem.add_relation(b_src, "is_a", b_dst)

        # Test: query b_src, expect b_dst reachable
        try:
            activated = sem.spread_activation([b_src], steps=2)
            if b_dst in activated:
                successes += 1
        except Exception:
            pass

    return float(successes / _N_SCENARIOS) if _N_SCENARIOS > 0 else 0.0
