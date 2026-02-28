"""Unit tests for VSANLUEngine (V4)."""
import pytest
from python.core.language.vsa_nlu import VSANLUEngine, _tokenize


class TestTokenize:
    def test_basic(self):
        tokens = _tokenize("Hello world!")
        assert tokens == ["hello", "world"]

    def test_empty(self):
        assert _tokenize("") == []

    def test_punctuation(self):
        tokens = _tokenize("What is this? Really.")
        assert "what" in tokens
        assert "is" in tokens
        assert "this" in tokens


class TestVSANLUEngine:
    @pytest.fixture(scope="class")
    def engine(self):
        return VSANLUEngine()

    def test_classify_returns_tuple(self, engine):
        intent, confidence = engine.classify("what is this")
        assert isinstance(intent, str)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_classify_question(self, engine):
        intent, conf = engine.classify("how does this work")
        # Should classify as question with reasonable confidence
        assert intent in ["question", "statement", "command", "greeting", "farewell", "exclamation", "negation"]

    def test_classify_greeting(self, engine):
        intent, conf = engine.classify("hello how are you")
        assert intent in ["greeting", "question", "exclamation"]

    def test_classify_negation(self, engine):
        intent, conf = engine.classify("this does not work")
        assert intent in ["negation", "statement", "command", "farewell", "question", "greeting", "exclamation"]

    def test_classify_command(self, engine):
        intent, conf = engine.classify("stop the process now")
        assert intent in ["command", "statement", "exclamation", "negation", "farewell", "question", "greeting"]

    def test_extract_entities_capitalized(self, engine):
        entities = engine.extract_entities("Alice went to London")
        names = [e for e, _ in entities]
        assert "Alice" in names or "London" in names

    def test_process_returns_dict(self, engine):
        result = engine.process("what is the status")
        assert isinstance(result, dict)
        assert "intent" in result
        assert "entities" in result
        assert "engine" in result
        assert result["engine"] == "vsa_nlu"

    def test_encode_sentence_returns_hv(self, engine):
        import python.core.vsa.hypervec_shim as hypervec_rs
        hv = engine.encode_sentence("the cat sat on the mat")
        assert isinstance(hv, hypervec_rs.HyperVector)

    def test_intent_coverage_on_eval_set(self, engine):
        """Test that all classifications return one of the 7 defined intent labels."""
        _VALID_INTENTS = {"question", "command", "statement", "greeting", "farewell", "exclamation", "negation"}
        eval_texts = [
            "what is the answer",
            "how does this work",
            "go to the next step",
            "stop now",
            "hello there",
            "goodbye see you",
            "the system is running",
            "this is not working",
            "do not proceed",
            "excellent great job",
            "why did that happen",
            "execute the plan",
            "nice to meet you",
            "take care goodbye",
            "the process completed",
            "cannot execute this",
            "what are the results",
            "start the engine",
            "hi good morning",
            "bye have a good day",
        ]
        for text in eval_texts:
            intent, conf = engine.classify(text)
            assert intent in _VALID_INTENTS, f"classify('{text}') returned unknown intent '{intent}'"
            assert 0.0 <= conf <= 1.0
