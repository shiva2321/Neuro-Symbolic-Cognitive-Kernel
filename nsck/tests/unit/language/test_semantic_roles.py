"""
Tests for the VSA Semantic Role Labeling module.

Validates that SemanticRoleLabeler correctly identifies:
- Predicate (main verb)
- AGENT, PATIENT, THEME, LOCATION, TEMPORAL, MANNER, NEGATION
- Resonator verification path
- Batch labeling
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from python.core.language.semantic_roles import (
    SemanticRoleLabeler,
    SRLFrame,
    _tokenize,
    _word_hv,
    _role_hv,
)


class TestSRLFrame:
    def test_to_dict_basic(self):
        f = SRLFrame(pred="bite", agent="dog", patient="boy", confidence=0.8)
        d = f.to_dict()
        assert d["pred"] == "bite"
        assert d["agent"] == "dog"
        assert d["patient"] == "boy"
        assert "confidence" in d

    def test_to_dict_optional_empty(self):
        f = SRLFrame(pred="run", agent="mary")
        d = f.to_dict()
        # Empty optional fields should not appear
        assert "location" not in d
        assert "temporal" not in d

    def test_str_includes_pred(self):
        f = SRLFrame(pred="bite", agent="dog", patient="boy")
        s = str(f)
        assert "pred='bite'" in s
        assert "agent='dog'" in s


class TestTokenize:
    def test_basic(self):
        tokens = _tokenize("The dog bit the boy.")
        assert "dog" in tokens
        assert "bit" in tokens
        assert "boy" in tokens

    def test_lowercase(self):
        tokens = _tokenize("Alice RAN quickly")
        assert all(t == t.lower() for t in tokens)


class TestSemanticRoleLabeler:
    def setup_method(self):
        self.srl = SemanticRoleLabeler()

    def test_basic_sentence(self):
        frame = self.srl.label("The dog bit the boy.")
        assert frame.pred != ""

    def test_agent_extraction(self):
        frame = self.srl.label("Alice gave the book to Bob.")
        # Agent should be alice or similar
        assert frame.agent != ""

    def test_predicate_is_verb(self):
        frame = self.srl.label("Mary quickly ran to the store.")
        # The predicate should be some verb form
        assert len(frame.pred) > 0

    def test_location_extraction(self):
        frame = self.srl.label("The cat sat in the garden.")
        assert frame.location != ""

    def test_temporal_extraction(self):
        frame = self.srl.label("She arrived yesterday.")
        assert frame.temporal == "yesterday"

    def test_manner_adverb(self):
        frame = self.srl.label("He quickly ran away.")
        assert frame.manner == "quickly"

    def test_negation_detection(self):
        frame = self.srl.label("He did not eat the apple.")
        assert frame.negation is True

    def test_no_negation(self):
        frame = self.srl.label("She ate the apple.")
        assert frame.negation is False

    def test_confidence_range(self):
        frame = self.srl.label("The dog barked loudly.")
        assert 0.0 <= frame.confidence <= 1.0

    def test_to_dict(self):
        frame = self.srl.label("Mary ran quickly.")
        d = frame.to_dict()
        assert isinstance(d, dict)
        assert "pred" in d
        assert "confidence" in d

    def test_batch_label(self):
        sentences = [
            "The dog bit the boy.",
            "Alice gave a book to Bob.",
            "She ran quickly in the park.",
        ]
        frames = self.srl.label_batch(sentences)
        assert len(frames) == len(sentences)
        for f in frames:
            assert isinstance(f, SRLFrame)

    def test_update_codebook(self):
        words = ["apple", "tree", "garden", "grow"]
        self.srl.update_codebook(words)
        for w in words:
            assert w in self.srl._codebook

    def test_empty_sentence(self):
        # Should not crash on empty/trivial input
        frame = self.srl.label("")
        assert isinstance(frame, SRLFrame)

    def test_passive_voice_has_pred(self):
        frame = self.srl.label("The book was written by Alice.")
        assert frame.pred != ""

    def test_resonator_verify_doesnt_corrupt(self):
        """After resonator verification the frame should still have a predicate."""
        self.srl.update_codebook(["dog", "bit", "boy", "quickly", "park"])
        frame = self.srl.label("The dog quickly bit the boy in the park.")
        assert frame.pred != ""
        assert frame.confidence > 0.0


class TestVSAHVFunctions:
    def test_word_hv_deterministic(self):
        hv1 = _word_hv("apple")
        hv2 = _word_hv("apple")
        assert float(hv1.similarity(hv2)) == pytest.approx(1.0)

    def test_role_hv_deterministic(self):
        hv1 = _role_hv("AGENT")
        hv2 = _role_hv("AGENT")
        assert float(hv1.similarity(hv2)) == pytest.approx(1.0)

    def test_different_words_not_identical(self):
        hv1 = _word_hv("apple")
        hv2 = _word_hv("orange")
        assert float(hv1.similarity(hv2)) < 0.9

    def test_different_roles_not_identical(self):
        hv1 = _role_hv("AGENT")
        hv2 = _role_hv("PATIENT")
        assert float(hv1.similarity(hv2)) < 0.9
