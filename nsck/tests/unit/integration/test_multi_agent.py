"""Tests for V8 Multi-Agent Cognitive Fusion."""
import pytest

from python.core.integration.brain_fusion import MultiAgentSession
from python.core.cognitive.theory_of_mind import TheoryOfMind
import python.core.vsa.hypervec_shim as hv


class _MockEngine:
    """Minimal mock engine with semantic_memory and get_belief_summary."""
    def __init__(self, seed=1):
        self._seed = seed
        class _Mem:
            def __init__(self, seed):
                import python.core.vsa.hypervec_shim as _hv
                self.concept_hvs = {"c": _hv.HyperVector(seed)}
        self.semantic_memory = _Mem(seed)
    
    def get_belief_summary(self, topic_hv):
        return topic_hv


class TestMultiAgentSessionInit:
    def test_empty_init(self):
        mas = MultiAgentSession()
        assert len(mas) == 0

    def test_init_with_engines(self):
        mas = MultiAgentSession([_MockEngine(1), _MockEngine(2)])
        assert len(mas) == 2

    def test_add_engine(self):
        mas = MultiAgentSession()
        idx = mas.add_engine(_MockEngine(1))
        assert idx == 0
        assert len(mas) == 1

    def test_add_multiple_engines(self):
        mas = MultiAgentSession()
        for i in range(3):
            mas.add_engine(_MockEngine(i))
        assert len(mas) == 3

    def test_engines_list(self):
        e1, e2 = _MockEngine(1), _MockEngine(2)
        mas = MultiAgentSession([e1, e2])
        assert mas.engines[0] is e1
        assert mas.engines[1] is e2


class TestExchangeSnapshots:
    def test_returns_dict(self):
        mas = MultiAgentSession([_MockEngine(1)])
        snaps = mas.exchange_snapshots()
        assert isinstance(snaps, dict)

    def test_keys_are_indices(self):
        mas = MultiAgentSession([_MockEngine(1), _MockEngine(2)])
        snaps = mas.exchange_snapshots()
        assert 0 in snaps
        assert 1 in snaps

    def test_snapshot_is_hv(self):
        mas = MultiAgentSession([_MockEngine(1)])
        snaps = mas.exchange_snapshots()
        assert snaps[0] is not None

    def test_empty_session(self):
        mas = MultiAgentSession()
        snaps = mas.exchange_snapshots()
        assert snaps == {}


class TestNegotiateBeliefs:
    def test_returns_hv(self):
        mas = MultiAgentSession([_MockEngine(1), _MockEngine(2)])
        topic = hv.HyperVector(42)
        result = mas.negotiate_beliefs(topic)
        assert result is not None

    def test_empty_session_returns_topic(self):
        mas = MultiAgentSession()
        topic = hv.HyperVector(42)
        result = mas.negotiate_beliefs(topic)
        assert result is topic

    def test_single_agent(self):
        mas = MultiAgentSession([_MockEngine(1)])
        topic = hv.HyperVector(42)
        result = mas.negotiate_beliefs(topic)
        assert result is not None


class TestTheoryOfMindV8:
    def test_model_other_agent(self):
        tom = TheoryOfMind()
        model = tom.model_other_agent("agent_a", ["pick_up", "move_left"])
        assert "pick_up" in model.intentions
        assert "move_left" in model.intentions

    def test_model_other_agent_dedup(self):
        tom = TheoryOfMind()
        tom.model_other_agent("agent_a", ["action_1"])
        tom.model_other_agent("agent_a", ["action_1", "action_2"])
        model = tom.get_or_create_model("agent_a")
        assert model.intentions.count("action_1") == 1

    def test_perspective_take_returns_dict(self):
        tom = TheoryOfMind()
        tom.update_agent_perspective("agent_b", "room_1", {"key_loc": "shelf"})
        topic = hv.HyperVector(10)
        beliefs = tom.perspective_take(topic, "agent_b")
        assert isinstance(beliefs, dict)

    def test_perspective_take_new_agent(self):
        tom = TheoryOfMind()
        topic = hv.HyperVector(10)
        beliefs = tom.perspective_take(topic, "new_agent")
        assert isinstance(beliefs, dict)
