"""Tests for V8 multi-turn dialogue state tracking."""
import pytest

from python.core.language.dialogue_manager import DialogueManager
from python.core.integration.config import NSCKConfig


class _MockEngine:
    class _MockMem:
        class _MockGraph:
            def nodes(self): return {}
            def out_edges(self, *a, **kw): return []
            def __contains__(self, item): return False
        concept_graph = _MockGraph()
        concept_hvs = {}
    semantic_memory = _MockMem()
    causal_graphs = {}
    def get_stats(self): return {}
    def explain(self): return ""
    def set_mission_goal(self, *a): pass


class _MockLang:
    def understand(self, text):
        return {"structured_output": {}, "grounded_hv": None}
    def generate(self, d, extra=None):
        return str(d)


@pytest.fixture()
def config_v8():
    return NSCKConfig(enable_dialogue_state_tracking=True)


@pytest.fixture()
def dm_v8(config_v8):
    return DialogueManager(_MockEngine(), _MockLang(), config=config_v8)


@pytest.fixture()
def dm_default():
    return DialogueManager(_MockEngine(), _MockLang())


class TestDialogueManagerInit:
    def test_default_config_none(self, dm_default):
        assert dm_default._config is None

    def test_v8_config_set(self, dm_v8, config_v8):
        assert dm_v8._config is config_v8

    def test_turn_count_zero(self, dm_default):
        assert dm_default._turn_count == 0

    def test_history_hv_none_initially(self, dm_default):
        assert dm_default._dialogue_history_hv is None

    def test_entity_register_exists(self, dm_default):
        assert dm_default._entity_register is not None


class TestDialogueStateTracking:
    def test_turn_count_increments(self, dm_v8):
        dm_v8.process_turn("hello world")
        assert dm_v8._turn_count >= 1

    def test_history_hv_set_after_turn(self, dm_v8):
        dm_v8.process_turn("tell me about cats")
        assert dm_v8._dialogue_history_hv is not None

    def test_topic_hv_set_after_turn(self, dm_v8):
        dm_v8.process_turn("what is the weather")
        assert dm_v8._current_topic_hv is not None

    def test_multiple_turns_accumulate(self, dm_v8):
        dm_v8.process_turn("first message")
        dm_v8.process_turn("second message")
        assert dm_v8._turn_count >= 2

    def test_no_tracking_when_disabled(self, dm_default):
        dm_default.process_turn("hello")
        assert dm_default._dialogue_history_hv is None
        assert dm_default._turn_count == 0


class TestGetContextHV:
    def test_get_context_hv_none_before_turn(self, dm_v8):
        assert dm_v8.get_context_hv() is None

    def test_get_context_hv_after_turn(self, dm_v8):
        dm_v8.process_turn("something interesting")
        hv = dm_v8.get_context_hv()
        assert hv is not None

    def test_context_hv_changes_with_turns(self, dm_v8):
        dm_v8.process_turn("first topic")
        h1 = dm_v8.get_context_hv()
        dm_v8.process_turn("second topic about something else")
        h2 = dm_v8.get_context_hv()
        # History should change
        sim = float(h1.similarity(h2))
        assert sim < 1.0


class TestClarificationRequest:
    def test_returns_string(self, dm_default):
        result = dm_default.clarification_request("quantum entanglement")
        assert isinstance(result, str)

    def test_contains_term(self, dm_default):
        result = dm_default.clarification_request("flux capacitor")
        assert "flux capacitor" in result

    def test_question_format(self, dm_default):
        result = dm_default.clarification_request("thing")
        assert "clarify" in result.lower() or "?" in result

    def test_empty_term(self, dm_default):
        result = dm_default.clarification_request("")
        assert isinstance(result, str)


class TestUpdateDialogueState:
    def test_internal_update_called(self, dm_v8):
        dm_v8._update_dialogue_state("test text")
        assert dm_v8._dialogue_history_hv is not None

    def test_empty_text_no_crash(self, dm_v8):
        dm_v8._update_dialogue_state("")  # should not raise

    def test_single_word(self, dm_v8):
        dm_v8._update_dialogue_state("hello")
        assert dm_v8._dialogue_history_hv is not None
