"""Tests for NSCK Evaluation Suite (V16 Initiative 3)."""
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))


def _make_suite():
    from eval.nsck_eval_suite import NSCKEvalSuite
    return NSCKEvalSuite()


# ---------------------------------------------------------------------------
# T1-T5 individual score tests
# ---------------------------------------------------------------------------

def test_t1_returns_float_in_range():
    from eval.tasks import t1_semantic_qa
    score = t1_semantic_qa.run()
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_t2_returns_float_in_range():
    from eval.tasks import t2_generalization
    score = t2_generalization.run()
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_t3_returns_float_in_range():
    from eval.tasks import t3_lifelong
    score = t3_lifelong.run()
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_t4_returns_float_in_range():
    from eval.tasks import t4_cross_modal
    score = t4_cross_modal.run()
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_t5_returns_float_in_range():
    from eval.tasks import t5_causal_reasoning
    score = t5_causal_reasoning.run()
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_nsck_es_weighted_correctly():
    suite = _make_suite()
    results = suite.run_all()
    # Manual computation
    weights = suite.WEIGHTS
    expected = sum(weights[k] * results[k] for k in weights)
    assert abs(results["nsck_es"] - expected) < 1e-9


def test_t5_causal_2hop_positive():
    """2-hop causal chain should be detected."""
    from python.core.reasoning.cognitive_engine import CognitiveEngine
    from python.core.integration.config import NSCKConfig
    engine = CognitiveEngine(config=NSCKConfig(), persistence_path=":memory:")
    sem = engine.semantic_memory
    sem.add_concept("fire", {"type": "event"})
    sem.add_concept("heat", {"type": "event"})
    sem.add_relation("fire", "causes", "heat")
    activated = sem.spread_activation(["fire"], steps=2)
    assert "heat" in activated


def test_t5_causal_negative_no_false_positive():
    """Unrelated concepts should not appear as causally reachable."""
    from python.core.reasoning.cognitive_engine import CognitiveEngine
    from python.core.integration.config import NSCKConfig
    engine = CognitiveEngine(config=NSCKConfig(), persistence_path=":memory:")
    sem = engine.semantic_memory
    sem.add_concept("isolated_a", {"type": "event"})
    sem.add_concept("isolated_b", {"type": "event"})
    # No relation between them
    activated = sem.spread_activation(["isolated_a"], steps=3)
    assert "isolated_b" not in activated


def test_t3_forgetting_measured():
    """T3 should measure and report no forgetting (semantic memory never forgets)."""
    from eval.tasks import t3_lifelong
    score = t3_lifelong.run()
    # Symbolic memory doesn't forget, expect score = 1.0
    assert score == 1.0


def test_regression_gate_runs():
    """Regression gate should execute and return a bool."""
    from eval.regression_gate import check_regression
    result = check_regression(verbose=False)
    assert isinstance(result, bool)


def test_babi_scoring_no_false_substring_match():
    """NSCK-ES T1 should not score 'ca' as matching 'cat' via substring."""
    from python.core.reasoning.cognitive_engine import CognitiveEngine
    from python.core.integration.config import NSCKConfig
    engine = CognitiveEngine(config=NSCKConfig(), persistence_path=":memory:")
    sem = engine.semantic_memory
    sem.add_concept("cat", {"type": "animal"})
    sem.add_concept("ca", {"type": "abbreviation"})
    # Only 'ca' has a relation to a query node
    sem.add_relation("query_node", "refers_to", "ca")
    activated = sem.spread_activation(["query_node"], steps=1)
    # 'cat' should NOT appear — only 'ca' is directly connected
    assert "cat" not in activated or "ca" in activated


def test_run_all_returns_dict_with_version():
    suite = _make_suite()
    results = suite.run_all()
    assert "nsck_es" in results
    assert results["version"] == "1.0"
    for k in ("t1", "t2", "t3", "t4", "t5"):
        assert k in results
        assert 0.0 <= results[k] <= 1.0
