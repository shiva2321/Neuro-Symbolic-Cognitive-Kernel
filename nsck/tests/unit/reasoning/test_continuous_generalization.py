"""Test continuous generalization fires automatically in decide() loop."""
import pytest

def _make_engine():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../'))
    from python.core.reasoning.cognitive_engine import CognitiveEngine
    from python.core.integration.config import NSCKConfig
    cfg = NSCKConfig(enable_continuous_generalization=True, generalization_interval=10, enable_sleep=False)
    eng = CognitiveEngine(config=cfg, persistence_path=None)
    eng.register_task("gen_test")
    return eng

def test_decision_counter_increments():
    eng = _make_engine()
    assert eng._decision_counter == 0
    eng.decide({"x": 1}, "gen_test")
    assert eng._decision_counter == 1

def test_continuous_generalization_triggers():
    """After generalization_interval decisions, generalization fires without error."""
    eng = _make_engine()
    for i in range(15):
        eng.decide({"x": i}, "gen_test")
    # Should have completed 15 decisions triggering generalize at 10
    assert eng._decision_counter == 15

def test_continuous_generalization_disabled():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../'))
    from python.core.reasoning.cognitive_engine import CognitiveEngine
    from python.core.integration.config import NSCKConfig
    cfg = NSCKConfig(enable_continuous_generalization=False, generalization_interval=5)
    eng = CognitiveEngine(config=cfg, persistence_path=None)
    eng.register_task("gen_test2")
    for i in range(10):
        eng.decide({"x": i}, "gen_test2")
    assert eng._decision_counter == 10
