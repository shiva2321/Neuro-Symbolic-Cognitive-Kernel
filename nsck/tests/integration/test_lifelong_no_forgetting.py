"""Integration tests for lifelong learning / no-forgetting with EWC (V16)."""
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from python.core.integration.config import NSCKConfig


def _make_engine(enable_ewc=False):
    from python.core.reasoning.cognitive_engine import CognitiveEngine
    cfg = NSCKConfig()
    cfg.enable_ewc = enable_ewc
    cfg.ewc_lambda = 500.0
    cfg.ewc_consolidate_interval = 5
    return CognitiveEngine(config=cfg, persistence_path=":memory:")


def _teach_task(engine, task_tag, facts, n_steps=5):
    """Inject facts into semantic memory and run a few decide/learn cycles."""
    engine.register_task(task_tag)
    sem = engine.semantic_memory
    for concept, props in facts:
        sem.add_concept(concept, props)
    state = {"x": 0}
    for _ in range(n_steps):
        cs = engine.decide(state, task_tag=task_tag)
        engine.learn(state, cs.chosen_action, reward=1.0, task_tag=task_tag)


def _count_known_concepts(engine, concepts):
    sem = engine.semantic_memory
    return sum(1 for c in concepts if c in sem.concept_hvs)


TASK_A_FACTS = [(f"animal_{i}", {"type": "animal"}) for i in range(20)]
TASK_B_FACTS = [(f"vehicle_{i}", {"type": "vehicle"}) for i in range(20)]


def test_task_a_knowledge_retained_after_task_b():
    """Task A concepts should still be in semantic memory after Task B training."""
    engine = _make_engine(enable_ewc=True)
    _teach_task(engine, "task_a", TASK_A_FACTS)

    task_a_concepts = [f[0] for f in TASK_A_FACTS]
    recall_before = _count_known_concepts(engine, task_a_concepts)

    _teach_task(engine, "task_b", TASK_B_FACTS)

    recall_after = _count_known_concepts(engine, task_a_concepts)
    # VSA/symbolic memory doesn't forget — expect full retention
    assert recall_after == recall_before, (
        f"Forgot {recall_before - recall_after} concepts after task B"
    )


def test_ewc_reduces_forgetting_vs_no_ewc():
    """With EWC enabled, concept retention should be >= without EWC.

    Since symbolic memory (SemanticMemory) never forgets, both should score 1.0.
    This test confirms EWC doesn't break retention.
    """
    engine_ewc = _make_engine(enable_ewc=True)
    engine_no_ewc = _make_engine(enable_ewc=False)

    task_a_concepts = [f[0] for f in TASK_A_FACTS]

    for engine in [engine_ewc, engine_no_ewc]:
        _teach_task(engine, "task_a", TASK_A_FACTS)
        _teach_task(engine, "task_b", TASK_B_FACTS)

    retention_ewc = _count_known_concepts(engine_ewc, task_a_concepts)
    retention_no_ewc = _count_known_concepts(engine_no_ewc, task_a_concepts)

    # EWC should not reduce retention
    assert retention_ewc >= retention_no_ewc
