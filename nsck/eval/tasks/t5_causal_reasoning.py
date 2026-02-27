"""T5 — Causal Reasoning Task (weight 0.15).

20 chains (mix of 2-hop, 3-hop, negative). Feed facts, query causal
reachability via SemanticMemory spreading activation or CausalReasoner.
Score = (correct_positive + correct_negative) / total.
"""
from __future__ import annotations
from typing import Any, List, Tuple


# (cause, intermediate, effect, n_hops, expected_reachable)
_CHAINS: List[Tuple] = []
for _i in range(8):
    _CHAINS.append((f"cr_a_{_i}", None, f"cr_b_{_i}", 2, True))  # 2-hop positive
for _i in range(7):
    _CHAINS.append((f"cr_x_{_i}", f"cr_y_{_i}", f"cr_z_{_i}", 3, True))  # 3-hop positive
for _i in range(5):
    _CHAINS.append((f"cr_neg_src_{_i}", None, f"cr_neg_dst_{_i}", 2, False))  # negative


def run(engine: Any = None) -> float:
    """Run T5 and return score in [0, 1]."""
    if engine is None:
        from python.core.reasoning.cognitive_engine import CognitiveEngine
        from python.core.integration.config import NSCKConfig
        engine = CognitiveEngine(config=NSCKConfig(), persistence_path=":memory:")

    sem = engine.semantic_memory

    # Add positive chains
    for cause, mid, effect, hops, expected in _CHAINS:
        if not expected:
            # Negative: only add concepts, no causal relation
            sem.add_concept(cause, {"type": "event"})
            sem.add_concept(effect, {"type": "event"})
            continue
        sem.add_concept(cause, {"type": "event"})
        sem.add_concept(effect, {"type": "event"})
        if mid:
            sem.add_concept(mid, {"type": "event"})
            sem.add_relation(cause, "causes", mid)
            sem.add_relation(mid, "causes", effect)
        else:
            sem.add_relation(cause, "causes", effect)

    correct = 0
    for cause, mid, effect, hops, expected in _CHAINS:
        try:
            activated = sem.spread_activation([cause], steps=hops + 1)
            is_reachable = effect in activated
            if is_reachable == expected:
                correct += 1
        except Exception:
            pass

    return float(correct / len(_CHAINS)) if _CHAINS else 0.0
