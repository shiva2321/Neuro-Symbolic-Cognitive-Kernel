"""Tests for V8 MathReasoner integration into CognitiveEngine."""
import pytest

from python.core.reasoning.math_reasoning import MathReasoner
from python.core.language.universal_input import UniversalInput


@pytest.fixture()
def mr():
    return MathReasoner()


@pytest.fixture()
def ui():
    return UniversalInput()


class TestMathReasoner:
    def test_solve_expression_add(self, mr):
        assert mr.solve_expression("3 + 5") == pytest.approx(8.0)

    def test_solve_expression_subtract(self, mr):
        assert mr.solve_expression("10 - 4") == pytest.approx(6.0)

    def test_solve_expression_multiply(self, mr):
        assert mr.solve_expression("6 * 7") == pytest.approx(42.0)

    def test_solve_expression_divide(self, mr):
        assert mr.solve_expression("20 / 4") == pytest.approx(5.0)

    def test_solve_word_problem_addition(self, mr):
        r = mr.solve_word_problem("Alice has 5 apples. Bob gives her 3 more. How many?")
        assert r is not None
        assert abs(float(r.get("answer", 0)) - 8.0) < 0.5

    def test_solve_word_problem_returns_dict(self, mr):
        r = mr.solve_word_problem("What is 2 plus 2?")
        assert isinstance(r, dict)

    def test_solve_word_problem_has_answer_key(self, mr):
        r = mr.solve_word_problem("What is 3 times 3?")
        assert "answer" in r


class TestIsMathematical:
    def test_arithmetic_expr(self, ui):
        assert ui.is_mathematical("3 + 5") is True

    def test_word_plus(self, ui):
        assert ui.is_mathematical("what is 5 plus 3?") is True

    def test_word_times(self, ui):
        assert ui.is_mathematical("what is 3 times 4?") is True

    def test_solve_keyword(self, ui):
        assert ui.is_mathematical("solve x + 3 = 7") is True

    def test_calculate_keyword(self, ui):
        assert ui.is_mathematical("calculate 10 * 2") is True

    def test_non_math(self, ui):
        assert ui.is_mathematical("what is the capital of France?") is False

    def test_non_math_story(self, ui):
        assert ui.is_mathematical("the cat sat on the mat") is False

    def test_what_is_number(self, ui):
        assert ui.is_mathematical("what is 10 minus 4?") is True


class TestCognitiveEngineBeliefSummary:
    def test_get_belief_summary_exists(self):
        from python.core.reasoning.cognitive_engine import CognitiveEngine
        engine = CognitiveEngine()
        hv = engine.get_concept_hv("test")
        result = engine.get_belief_summary(hv)
        assert result is not None
