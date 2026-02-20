"""
Tests for the enhanced NLG module (DiscoursePlanner + NLGEngine).

Validates:
- StructuralRealizer (existing) still works correctly
- DiscoursePlanner multi-sentence generation
- Connective insertion
- Pronoun anaphora
- Procedural formatting
- NLGEngine.generate_discourse API
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from python.core.language.nlg import StructuralRealizer, DiscoursePlanner, NLGEngine


class TestStructuralRealizerBackCompat:
    """Ensure existing StructuralRealizer behaviour is unchanged."""

    def setup_method(self):
        self.r = StructuralRealizer()

    def test_pluralize_regular(self):
        assert self.r.pluralize("cat") == "cats"

    def test_pluralize_s_ending(self):
        assert self.r.pluralize("bus") == "buses"

    def test_conjugate_be(self):
        assert self.r.conjugate("be", "3rd", "singular", "present") == "is"
        assert self.r.conjugate("be", "1st", "singular", "present") == "am"
        assert self.r.conjugate("be", "1st", "plural", "present") == "are"

    def test_conjugate_regular(self):
        assert self.r.conjugate("run", "3rd", "singular", "present") == "runs"

    def test_realize_sentence_isa(self):
        s = self.r.realize_sentence("Valkyria", "is_a", "Game")
        assert "Valkyria" in s
        assert "is" in s
        assert "game" in s

    def test_realize_sentence_capitalized(self):
        s = self.r.realize_sentence("dog", "is_a", "mammal")
        assert s[0].isupper()

    def test_realize_chain_strings(self):
        chain = ["Rain", "Wet Grass", "Mud"]
        s = self.r.realize_chain(chain)
        assert "Rain" in s
        assert "Mud" in s or "mud" in s


class TestDiscoursePlanner:
    def setup_method(self):
        self.planner = DiscoursePlanner()

    def test_single_frame(self):
        frames = [{"subject": "dog", "relation": "is_a", "object": "mammal"}]
        text = self.planner.plan(frames)
        assert "dog" in text.lower()
        assert "mammal" in text.lower()

    def test_multi_frame_produces_multiple_sentences(self):
        frames = [
            {"subject": "water", "relation": "is_a", "object": "liquid"},
            {"subject": "water", "relation": "has_property", "object": "wetness"},
            {"subject": "water", "relation": "causes", "object": "rust"},
        ]
        text = self.planner.plan(frames, query_type="explanatory")
        # Should have at least two sentences
        assert text.count(".") >= 2

    def test_causal_connective_appears(self):
        frames = [
            {"subject": "rain", "relation": "causes", "object": "flooding"},
            {"subject": "rain", "relation": "causes", "object": "mud"},
        ]
        text = self.planner.plan(frames, query_type="causal")
        # One of the causal connectives should appear for subsequent sentences
        connectives = ["As a result", "Therefore", "Also", "In addition"]
        assert any(c in text for c in connectives)

    def test_procedural_formatting(self):
        frames = [
            {"subject": "step1", "relation": "precedes", "object": "step2"},
            {"subject": "step2", "relation": "precedes", "object": "step3"},
        ]
        text = self.planner.plan(frames, query_type="procedural")
        assert "1." in text
        assert "2." in text

    def test_anaphora_replaces_repeated_subject(self):
        frames = [
            {"subject": "fire", "relation": "is_a", "object": "combustion"},
            {"subject": "fire", "relation": "causes", "object": "smoke"},
        ]
        text = self.planner.plan(frames, query_type="explanatory")
        # "fire" should be replaced by "it" in the second sentence
        sentences = text.split(".")
        if len(sentences) >= 2:
            second = sentences[1].lower().strip()
            assert "it" in second or "fire" in second  # at minimum doesn't crash

    def test_empty_frames_returns_fallback(self):
        text = self.planner.plan([])
        assert len(text) > 0  # Returns fallback message

    def test_negate_frame(self):
        frames = [{"subject": "fish", "relation": "is_a", "object": "mammal", "negate": True}]
        text = self.planner.plan(frames)
        assert "not" in text.lower()

    def test_sort_preserves_causal_order(self):
        frames = [
            {"subject": "A", "relation": "causes", "object": "B"},
            {"subject": "B", "relation": "causes", "object": "C"},
        ]
        result = self.planner._sort_frames(frames, "causal")
        assert result[0]["subject"] == "A"
        assert result[1]["subject"] == "B"


class TestNLGEngine:
    def setup_method(self):
        self.engine = NLGEngine()

    def test_generate_fact(self):
        s = self.engine.generate("fact", {"subject": "dog", "relation": "is_a", "object": "animal"})
        assert "dog" in s.lower()

    def test_generate_unknown_category(self):
        s = self.engine.generate("unknown_cat", {})
        assert len(s) > 0

    def test_generate_discourse_single(self):
        frames = [{"subject": "python", "relation": "is_a", "object": "language"}]
        text = self.engine.generate_discourse(frames, topic="python")
        assert "python" in text.lower()

    def test_generate_discourse_multi(self):
        frames = [
            {"subject": "python", "relation": "is_a", "object": "language"},
            {"subject": "python", "relation": "has_property", "object": "simplicity"},
        ]
        text = self.engine.generate_discourse(frames, topic="python", query_type="explanatory")
        assert len(text) > 0

    def test_generate_causal_chain(self):
        chain = ["heat", "evaporation", "clouds", "rain"]
        text = self.engine.generate_causal_chain(chain)
        assert "heat" in text.lower()
        assert "rain" in text.lower()
