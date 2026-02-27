"""Tests for EWC wiring in CognitiveEngine (V16 Initiative 2)."""
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from python.core.integration.config import NSCKConfig


def _make_engine(enable_ewc=False):
    from python.core.reasoning.cognitive_engine import CognitiveEngine
    cfg = NSCKConfig()
    cfg.enable_ewc = enable_ewc
    cfg.ewc_lambda = 100.0
    return CognitiveEngine(config=cfg, persistence_path=":memory:")


def test_continual_learner_not_instantiated_by_default():
    engine = _make_engine(enable_ewc=False)
    assert engine._continual_learner is None


def test_continual_learner_instantiated_when_ewc_enabled():
    engine = _make_engine(enable_ewc=True)
    assert engine._continual_learner is not None


def test_task_registered_with_continual_learner():
    engine = _make_engine(enable_ewc=True)
    engine.register_task("test_task")
    assert "test_task" in engine._continual_learner.tasks


def test_importance_buffer_fills_on_decide():
    from python.core.reasoning.cognitive_engine import CognitiveEngine
    engine = _make_engine(enable_ewc=True)
    engine.register_task("ewc_task")
    state = {"robot_x": 0, "robot_y": 0}
    for _ in range(3):
        engine.decide(state, task_tag="ewc_task")
    buf = engine._ewc_importance_buffer.get("ewc_task", [])
    assert len(buf) >= 1


def test_consolidation_called_on_sleep():
    engine = _make_engine(enable_ewc=True)
    engine.register_task("sleep_task")
    state = {"x": 0}
    engine.decide(state, task_tag="sleep_task")
    engine.sleep()
    assert engine.stats.get("tasks_consolidated", 0) >= 1


def test_ewc_trace_fields_present_when_enabled():
    engine = _make_engine(enable_ewc=True)
    engine.register_task("trace_task")
    state = {"x": 1}
    result = engine.decide(state, task_tag="trace_task")
    assert "ewc_protected_concepts" in result.trace
    assert "ewc_loss" in result.trace


def test_get_protected_concepts_returns_list():
    from python.core.learning.continual_learning import ContinualLearner
    import numpy as np
    learner = ContinualLearner(ewc_lambda=100.0)
    learner.register_task("t1")
    params = {"concept_a": np.array([0.8]), "concept_b": np.array([0.3])}
    grads = {"concept_a": np.array([0.64]), "concept_b": np.array([0.09])}
    learner.compute_importance("t1", params, grads)
    result = learner.get_protected_concepts("t1")
    assert isinstance(result, list)
    assert "concept_a" in result


def test_ewc_config_fields_in_research_preset():
    cfg = NSCKConfig.research()
    assert getattr(cfg, 'enable_ewc', False) is True
    assert getattr(cfg, 'ewc_lambda', 0) > 0


def test_ewc_disabled_in_minimal_preset():
    cfg = NSCKConfig.minimal()
    assert getattr(cfg, 'enable_ewc', False) is False
