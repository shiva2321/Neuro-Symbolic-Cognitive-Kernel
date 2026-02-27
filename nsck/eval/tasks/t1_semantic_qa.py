"""T1 — Semantic QA Task (weight 0.30).

100 QA pairs across 4 categories. Feeds facts to semantic memory,
runs spreading activation, checks if expected concept is activated.
Score = weighted accuracy across categories.
"""
from __future__ import annotations
from typing import Dict, Any


_CATEGORIES = {
    "direct":    (0.4, 25),
    "property":  (0.3, 25),
    "causal":    (0.2, 25),
    "multihop":  (0.1, 25),
}


def run(engine: Any = None) -> float:
    """Run T1 and return score in [0, 1]."""
    if engine is None:
        from python.core.reasoning.cognitive_engine import CognitiveEngine
        from python.core.integration.config import NSCKConfig
        engine = CognitiveEngine(config=NSCKConfig(), persistence_path=":memory:")

    sem = engine.semantic_memory
    correct: Dict[str, int] = {c: 0 for c in _CATEGORIES}
    total: Dict[str, int] = {c: 0 for c in _CATEGORIES}

    # -- Direct retrieval: teach a→b, query a, expect b activated
    direct_pairs = [(f"d_src_{i}", f"d_dst_{i}") for i in range(25)]
    for src, dst in direct_pairs:
        sem.add_concept(src, {"type": "node"})
        sem.add_concept(dst, {"type": "node"})
        sem.add_relation(src, "relates_to", dst)
    for src, dst in direct_pairs:
        total["direct"] += 1
        try:
            activated = sem.spread_activation([src], steps=2)
            if dst in activated:
                correct["direct"] += 1
        except Exception:
            pass

    # -- Property retrieval: concept with property, query concept, check property concept
    for i in range(25):
        c = f"p_concept_{i}"
        prop = f"p_prop_{i}"
        sem.add_concept(c, {"type": "entity", "color": prop})
        sem.add_concept(prop, {"type": "property"})
        sem.add_relation(c, "has_property", prop)
    for i in range(25):
        c = f"p_concept_{i}"
        prop = f"p_prop_{i}"
        total["property"] += 1
        try:
            activated = sem.spread_activation([c], steps=1)
            if prop in activated:
                correct["property"] += 1
        except Exception:
            pass

    # -- Causal retrieval: cause → effect
    for i in range(25):
        cause = f"c_cause_{i}"
        effect = f"c_effect_{i}"
        sem.add_concept(cause, {"type": "event"})
        sem.add_concept(effect, {"type": "event"})
        sem.add_relation(cause, "causes", effect)
    for i in range(25):
        cause = f"c_cause_{i}"
        effect = f"c_effect_{i}"
        total["causal"] += 1
        try:
            activated = sem.spread_activation([cause], steps=2)
            if effect in activated:
                correct["causal"] += 1
        except Exception:
            pass

    # -- Multi-hop: a→b→c
    for i in range(25):
        a = f"mh_a_{i}"
        b = f"mh_b_{i}"
        c = f"mh_c_{i}"
        sem.add_concept(a, {})
        sem.add_concept(b, {})
        sem.add_concept(c, {})
        sem.add_relation(a, "leads_to", b)
        sem.add_relation(b, "leads_to", c)
    for i in range(25):
        a = f"mh_a_{i}"
        c = f"mh_c_{i}"
        total["multihop"] += 1
        try:
            activated = sem.spread_activation([a], steps=3)
            if c in activated:
                correct["multihop"] += 1
        except Exception:
            pass

    # Weighted accuracy
    score = 0.0
    for cat, (w, _n) in _CATEGORIES.items():
        if total[cat] > 0:
            score += w * (correct[cat] / total[cat])
    return float(score)
