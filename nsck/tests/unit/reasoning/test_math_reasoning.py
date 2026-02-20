"""
Tests for the Math Reasoning module.

Validates arithmetic, algebra, word-problem parsing, FPE magnitude
similarity, and math-query detection.
"""

import sys
import os
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from python.core.reasoning.math_reasoning import (
    MathReasoner,
    ExpressionEvaluator,
    LinearSolver,
    WordProblemParser,
    FPECodebook,
)


class TestExpressionEvaluator:
    def setup_method(self):
        self.ev = ExpressionEvaluator()

    def test_addition(self):
        assert self.ev.evaluate("3 + 5") == pytest.approx(8.0)

    def test_subtraction(self):
        assert self.ev.evaluate("10 - 4") == pytest.approx(6.0)

    def test_multiplication(self):
        assert self.ev.evaluate("3 * 4") == pytest.approx(12.0)

    def test_division(self):
        assert self.ev.evaluate("15 / 3") == pytest.approx(5.0)

    def test_parentheses(self):
        assert self.ev.evaluate("(3 + 5) * 2") == pytest.approx(16.0)

    def test_power(self):
        assert self.ev.evaluate("2 ** 10") == pytest.approx(1024.0)

    def test_unary_minus(self):
        assert self.ev.evaluate("-3 + 5") == pytest.approx(2.0)

    def test_nested_parentheses(self):
        assert self.ev.evaluate("((2 + 3) * (4 - 1)) / 5") == pytest.approx(3.0)

    def test_division_by_zero(self):
        with pytest.raises(ValueError):
            self.ev.evaluate("5 / 0")

    def test_float(self):
        assert self.ev.evaluate("1.5 + 2.5") == pytest.approx(4.0)


class TestLinearSolver:
    def setup_method(self):
        self.solver = LinearSolver()

    def test_simple_addition(self):
        result = self.solver.solve("x + 3 = 7")
        assert "x" in result
        assert result["x"] == pytest.approx(4.0)

    def test_rhs_constant(self):
        result = self.solver.solve("3 = x + 1")
        assert result["x"] == pytest.approx(2.0)

    def test_coefficient(self):
        result = self.solver.solve("2x = 10")
        assert list(result.values())[0] == pytest.approx(5.0)

    def test_negative_solution(self):
        result = self.solver.solve("x + 5 = 2")
        assert list(result.values())[0] == pytest.approx(-3.0)


class TestWordProblemParser:
    def setup_method(self):
        self.parser = WordProblemParser()

    def test_addition_problem(self):
        r = self.parser.parse("Alice has 5 apples. Bob gives her 3 more.")
        assert r.answer == pytest.approx(8.0)

    def test_subtraction_problem(self):
        r = self.parser.parse("Tom has 10 cookies. He eats 4.")
        assert r.answer == pytest.approx(6.0)

    def test_multiplication_problem(self):
        r = self.parser.parse("There are 4 groups of 3 students.")
        assert r.answer == pytest.approx(12.0)

    def test_direct_expression(self):
        r = self.parser.parse("What is 7 + 8?")
        assert r.answer == pytest.approx(15.0)

    def test_result_has_expression(self):
        r = self.parser.parse("What is 3 + 4?")
        assert "3" in r.expression or "4" in r.expression

    def test_result_has_explanation(self):
        r = self.parser.parse("What is 2 + 2?")
        assert len(r.explanation) > 0


class TestFPECodebook:
    def setup_method(self):
        self.cb = FPECodebook(max_int=100)

    def test_encode_returns_hv(self):
        hv = self.cb.encode(5)
        assert hv is not None

    def test_same_number_same_hv(self):
        hv1 = self.cb.encode(7)
        hv2 = self.cb.encode(7)
        assert float(hv1.similarity(hv2)) == pytest.approx(1.0)

    def test_nearby_numbers_more_similar_than_distant(self):
        sim_close = self.cb.similarity(1, 2)
        sim_far = self.cb.similarity(1, 50)
        assert sim_close > sim_far

    def test_find_nearest(self):
        hv = self.cb.encode(7)
        nearest = self.cb.find_nearest(hv, candidates=list(range(20)))
        assert nearest == 7


class TestMathReasoner:
    def setup_method(self):
        self.mr = MathReasoner()

    def test_solve_expression_basic(self):
        assert self.mr.solve_expression("3 + 5") == pytest.approx(8.0)

    def test_solve_expression_complex(self):
        assert self.mr.solve_expression("(10 - 4) * 3 / 2") == pytest.approx(9.0)

    def test_solve_algebra(self):
        result = self.mr.solve_algebra("x + 3 = 7")
        assert "x" in result
        assert result["x"] == pytest.approx(4.0)

    def test_solve_word_problem(self):
        result = self.mr.solve_word_problem("Alice has 5 apples. Bob gives her 3 more.")
        assert result["answer"] == pytest.approx(8.0)
        assert "expression" in result
        assert "explanation" in result

    def test_compare_less(self):
        cmp = self.mr.compare(3.0, 7.0)
        assert cmp["less"] is True
        assert cmp["greater"] is False
        assert cmp["equal"] is False

    def test_compare_equal(self):
        cmp = self.mr.compare(5.0, 5.0)
        assert cmp["equal"] is True

    def test_magnitude_similarity_close(self):
        sim = self.mr.magnitude_similarity(5, 6)
        assert 0 < sim <= 1.0

    def test_magnitude_similarity_same(self):
        sim = self.mr.magnitude_similarity(10, 10)
        assert sim == pytest.approx(1.0)

    def test_is_math_query_true(self):
        assert self.mr.is_math_query("What is 3 + 5?") is True
        assert self.mr.is_math_query("Calculate 2 * 4") is True
        assert self.mr.is_math_query("Solve x + 2 = 5") is True

    def test_is_math_query_false(self):
        assert self.mr.is_math_query("Tell me about dogs") is False

    def test_extract_numbers(self):
        nums = self.mr.extract_numbers("There are 3 cats and 7 dogs, totaling 10 animals.")
        assert 3.0 in nums
        assert 7.0 in nums
        assert 10.0 in nums

    def test_verbalize_answer(self):
        result = {"answer": 8.0, "expression": "5 + 3"}
        text = self.mr.verbalize_answer(result)
        assert "8" in text
        assert "5 + 3" in text

    def test_encode_number_returns_hv(self):
        hv = self.mr.encode_number(42)
        assert hv is not None
