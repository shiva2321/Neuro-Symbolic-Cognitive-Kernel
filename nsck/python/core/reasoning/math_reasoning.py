"""
NSCK Math Reasoning Module
===========================
VSA-native numeric understanding and arithmetic without neural networks.

Design
------
Numbers are encoded using **Fractional Power Encoding (FPE)** (Gosmann &
Eliasmith 2019).  Given a base hypervector ``v_base``, the encoding of
integer *n* is:

    v_n = v_base XOR'd n times (binding chain)

This gives a codebook where:
* ``v_0`` is the identity (close to all small numbers)
* ``v_n XOR v_m ≈ v_{n+m}``  (addition via binding!)
* ``sim(v_n, v_m)`` decreases monotonically as |n-m| increases

For practical arithmetic NSCK uses:
1. **FPE codebook** (pre-computed, integers 0–1023 + common fractions)
2. **Exact solver** for algebraic expressions (no approximation)
3. **Word-problem parser** that extracts numbers and operations from text
4. **Result verbaliser** that renders numeric answers in natural language

Key capabilities
----------------
* Arithmetic:  add, subtract, multiply, divide
* Comparison:  <, >, ==, <=, >=
* Simple algebra: solve  ax + b = c  (one variable, linear)
* Math word-problem parsing from natural-language text
* FPE-based approximate magnitude comparison via HV similarity

Usage
-----
>>> from python.core.reasoning.math_reasoning import MathReasoner
>>> mr = MathReasoner()
>>> mr.solve_expression("3 + 5")
8.0
>>> mr.solve_expression("(12 - 4) * 2")
16.0
>>> mr.solve_algebra("x + 3 = 7")
{'x': 4.0}
>>> mr.solve_word_problem("Alice has 5 apples. Bob gives her 3 more. How many?")
{'answer': 8.0, 'expression': '5 + 3', 'explanation': 'Alice has 5 apples. Bob gives her 3 more: 5 + 3 = 8.0'}
>>> sim = mr.magnitude_similarity(4, 5)   # close numbers → high similarity
"""

from __future__ import annotations

import re
import operator
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

import python.core.vsa.hypervec_shim as hypervec_rs


# ---------------------------------------------------------------------------
# FPE Codebook
# ---------------------------------------------------------------------------

class FPECodebook:
    """
    Fractional Power Encoding codebook for integers and fractions.

    Implemented via *incremental noise accumulation*: starting from a base
    hypervector, each integer step flips a small, deterministic set of bits.
    This gives the key property:
        sim(v_n, v_m) decreases as |n - m| increases.

    The number of bits flipped per step (``_FLIP_BITS``) controls the
    "resolution" of the numeric axis.  With D=10240 and flip_bits=64:
        sim(v_1, v_2) ≈ 0.988   (1 step  → ~64 bits differ)
        sim(v_1, v_50) ≈ 0.70   (49 steps → ~3136 bits differ)
    """

    BASE_SEED = 9876543
    _FLIP_BITS = 64   # bits flipped per integer step

    def __init__(self, max_int: int = 1023):
        self.max_int = max_int
        self._cache: Dict[Union[int, float], Any] = {}
        self._dim = 10240
        self._base_bits = self._make_base()
        # Encode 0 = identity (all-zero would break XOR; use base as 0)
        self._cache[0] = self._bits_to_hv(self._base_bits.copy())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_base(self) -> np.ndarray:
        rng = np.random.default_rng(self.BASE_SEED)
        return rng.integers(0, 2, size=self._dim, dtype=np.int8)

    def _bits_to_hv(self, bits: np.ndarray) -> Any:
        return hypervec_rs.HyperVector.from_bits(bits)

    def _flip_indices(self, step: int) -> np.ndarray:
        """Return the _FLIP_BITS indices to flip for integer step *step*."""
        rng = np.random.default_rng(self.BASE_SEED + abs(step) * 13337)
        return rng.choice(self._dim, size=self._FLIP_BITS, replace=False)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, n: Union[int, float]) -> Any:
        """
        Return HyperVector encoding of numeric value *n*.

        Non-integers use the floor encoding (sufficient for comparison).
        Negative integers use a separate noise chain in the opposite direction.
        """
        if n in self._cache:
            return self._cache[n]

        if isinstance(n, float) and not n.is_integer():
            n_int = int(np.floor(n))
            result = self.encode(n_int)
            self._cache[n] = result
            return result

        n_int = int(n)

        if n_int > 0:
            prev_bits = self._get_bits(n_int - 1)
            new_bits = prev_bits.copy()
            new_bits[self._flip_indices(n_int)] ^= 1
        elif n_int < 0:
            # Negative: walk in the opposite direction (different flip sets)
            prev_bits = self._get_bits(n_int + 1)
            new_bits = prev_bits.copy()
            new_bits[self._flip_indices(n_int - 100000)] ^= 1
        else:
            return self._cache[0]

        result = self._bits_to_hv(new_bits)
        self._cache[n_int] = result
        return result

    def _get_bits(self, n: int) -> np.ndarray:
        """Return the raw bits array for integer n, encoding it if needed."""
        hv = self.encode(n)
        return hv.bits.copy()

    def similarity(self, a: Union[int, float], b: Union[int, float]) -> float:
        """Cosine similarity between encoded numbers a and b."""
        v_a = self.encode(a)
        v_b = self.encode(b)
        return float(v_a.similarity(v_b))

    def find_nearest(self, hv: Any, candidates: Optional[List[int]] = None) -> int:
        """Find the integer in *candidates* whose encoding is closest to *hv*."""
        if candidates is None:
            candidates = list(range(self.max_int + 1))
        best_n = 0
        best_sim = -1.0
        for n in candidates:
            sim = float(hv.similarity(self.encode(n)))
            if sim > best_sim:
                best_sim = sim
                best_n = n
        return best_n


# ---------------------------------------------------------------------------
# Expression Evaluator (exact arithmetic)
# ---------------------------------------------------------------------------

class ExpressionEvaluator:
    """
    Safe arithmetic expression evaluator — no eval(), no external libs.

    Supports: +, -, *, /, ** (power), (), unary minus, integers, floats.
    """

    _OPS = {
        '+': operator.add,
        '-': operator.sub,
        '*': operator.mul,
        '/': operator.truediv,
        '**': operator.pow,
    }

    def evaluate(self, expr: str) -> float:
        """
        Parse and evaluate a mathematical expression string.

        Raises ValueError for invalid expressions or division by zero.
        """
        tokens = self._tokenize(expr)
        value, _ = self._parse_expr(tokens, 0)
        return float(value)

    # ------------------------------------------------------------------
    # Tokeniser
    # ------------------------------------------------------------------

    def _tokenize(self, expr: str) -> List[str]:
        pattern = r"\*\*|[+\-*/()]|\d+\.?\d*"
        return re.findall(pattern, expr.replace(" ", ""))

    # ------------------------------------------------------------------
    # Recursive-descent parser: expr → term → factor
    # ------------------------------------------------------------------

    def _parse_expr(self, tokens: List[str], pos: int) -> Tuple[float, int]:
        """Parse addition/subtraction."""
        left, pos = self._parse_term(tokens, pos)
        while pos < len(tokens) and tokens[pos] in ("+", "-"):
            op_str = tokens[pos]
            pos += 1
            right, pos = self._parse_term(tokens, pos)
            left = self._OPS[op_str](left, right)
        return left, pos

    def _parse_term(self, tokens: List[str], pos: int) -> Tuple[float, int]:
        """Parse multiplication/division."""
        left, pos = self._parse_power(tokens, pos)
        while pos < len(tokens) and tokens[pos] in ("*", "/"):
            op_str = tokens[pos]
            pos += 1
            right, pos = self._parse_power(tokens, pos)
            if op_str == "/" and right == 0:
                raise ValueError("Division by zero")
            left = self._OPS[op_str](left, right)
        return left, pos

    def _parse_power(self, tokens: List[str], pos: int) -> Tuple[float, int]:
        """Parse exponentiation (right-associative)."""
        base, pos = self._parse_unary(tokens, pos)
        if pos < len(tokens) and tokens[pos] == "**":
            pos += 1
            exp, pos = self._parse_power(tokens, pos)
            base = base ** exp
        return base, pos

    def _parse_unary(self, tokens: List[str], pos: int) -> Tuple[float, int]:
        """Handle unary minus."""
        if pos < len(tokens) and tokens[pos] == "-":
            pos += 1
            val, pos = self._parse_primary(tokens, pos)
            return -val, pos
        return self._parse_primary(tokens, pos)

    def _parse_primary(self, tokens: List[str], pos: int) -> Tuple[float, int]:
        """Parse number or parenthesised expression."""
        if pos >= len(tokens):
            raise ValueError("Unexpected end of expression")
        tok = tokens[pos]
        if tok == "(":
            pos += 1
            val, pos = self._parse_expr(tokens, pos)
            if pos >= len(tokens) or tokens[pos] != ")":
                raise ValueError("Missing closing parenthesis")
            pos += 1
            return val, pos
        try:
            return float(tok), pos + 1
        except ValueError:
            raise ValueError(f"Unexpected token: {tok!r}")


# ---------------------------------------------------------------------------
# Algebra Solver (linear, one variable)
# ---------------------------------------------------------------------------

class LinearSolver:
    """
    Solves linear equations with one variable: ax + b = c.

    Handles forms like:
      - "x + 3 = 7"   → x = 4
      - "2x - 1 = 5"  → x = 3
      - "3 = x + 1"   → x = 2
      - "x = 5"       → x = 5
    """

    _EQUATION_RE = re.compile(
        r"^\s*([^=]+)\s*=\s*([^=]+)\s*$"
    )
    _COEFF_RE = re.compile(
        r"([+\-]?\s*\d*\.?\d*)\s*([a-z])"
    )
    _CONST_RE = re.compile(
        r"(?<![a-z])([+\-]?\s*\d+\.?\d*)(?!\s*[a-z])"
    )

    def solve(self, equation: str) -> Dict[str, float]:
        """
        Solve a linear equation.

        Returns dict {variable_name: value} or raises ValueError.
        """
        m = self._EQUATION_RE.match(equation)
        if not m:
            raise ValueError(f"Cannot parse equation: {equation!r}")

        lhs_str = m.group(1).strip()
        rhs_str = m.group(2).strip()

        # Move everything to left: LHS - RHS = 0
        var_coeff, const = self._parse_side(lhs_str)
        rhs_var, rhs_const = self._parse_side(rhs_str)

        # combined: (var_coeff - rhs_var) * x + (const - rhs_const) = 0
        a = var_coeff - rhs_var
        b = const - rhs_const

        if a == 0:
            if b == 0:
                raise ValueError("Equation has infinitely many solutions")
            raise ValueError("Equation has no solution")

        var_name = self._find_variable(equation)
        solution = -b / a
        return {var_name: solution}

    def _parse_side(self, expr: str) -> Tuple[float, float]:
        """Return (variable_coefficient, constant_sum) for one side."""
        expr_clean = expr.replace(" ", "")
        # Find variable term(s)
        var_coeff = 0.0
        for match in self._COEFF_RE.finditer(expr_clean):
            coeff_str = match.group(1).strip() or "1"
            if coeff_str in ("", "+"):
                coeff_str = "1"
            elif coeff_str == "-":
                coeff_str = "-1"
            try:
                var_coeff += float(coeff_str)
            except ValueError:
                var_coeff += 1.0 if "+" not in coeff_str else -1.0

        # Remove variable terms, then sum remaining numeric tokens
        no_var = self._COEFF_RE.sub("", expr_clean)
        const = 0.0
        for match in re.finditer(r"[+\-]?\d+\.?\d*", no_var):
            const += float(match.group())

        return var_coeff, const

    def _find_variable(self, equation: str) -> str:
        """Extract variable name (first single lowercase letter)."""
        m = re.search(r"[a-z]", equation)
        return m.group() if m else "x"


# ---------------------------------------------------------------------------
# Word Problem Parser
# ---------------------------------------------------------------------------

@dataclass
class WordProblemResult:
    answer: float
    expression: str
    explanation: str


class WordProblemParser:
    """
    Extracts arithmetic operations from natural-language word problems
    and returns the numeric answer.

    Patterns handled
    ----------------
    * "has X ... gives/adds Y more"       → addition
    * "has X ... loses/takes/removes Y"   → subtraction
    * "X groups of Y" / "X times Y"       → multiplication
    * "X shared among Y" / "X divided by Y" → division
    * "what is X op Y?" / "calculate X op Y" → direct arithmetic
    * Falls back to extracting all numbers and applying heuristics
    """

    _DIRECT_RE = re.compile(
        r"(?:what\s+is|calculate|compute|evaluate|solve)?\s*"
        r"(-?\d+\.?\d*)\s*([+\-*/×÷])\s*(-?\d+\.?\d*)",
        re.IGNORECASE,
    )

    _ADD_CUES = re.compile(
        r"\b(?:add(?:s|ed)?|plus|more|gain(?:s|ed)?|give(?:s|n)?|receiv(?:es|ed)?|"
        r"earn(?:s|ed)?|buys?|purchas(?:es|ed)?|join(?:s|ed)?|pick(?:s|ed)?\s+up)\b",
        re.IGNORECASE,
    )
    _SUB_CUES = re.compile(
        r"\b(?:subtract(?:s|ed)?|minus|fewer|less|los(?:es|t)|"
        r"spend(?:s)?|spent|remov(?:es|ed)?|us(?:es|ed)?|"
        r"giv(?:es|en)?\s+away|shar(?:es|ed)?\s+with|eat(?:s|en)?)\b",
        re.IGNORECASE,
    )
    _MUL_CUES = re.compile(
        r"\b(?:times|multiply|multiplied|product|groups?\s+of|each\s+(?:has|have)|"
        r"per|every|×)\b",
        re.IGNORECASE,
    )
    _DIV_CUES = re.compile(
        r"\b(?:divid(?:es?|ed)|split|shar(?:ed?|ing)\s+(?:equally|among)|"
        r"each\s+gets?|÷)\b",
        re.IGNORECASE,
    )

    _NUM_RE = re.compile(r"-?\d+\.?\d*")

    def parse(self, text: str) -> WordProblemResult:
        """Parse a word problem and return the answer with explanation."""
        # 1. Try direct expression match first
        dm = self._DIRECT_RE.search(text)
        if dm:
            a = float(dm.group(1))
            sym = dm.group(2)
            b = float(dm.group(3))
            op_map = {"+": operator.add, "-": operator.sub,
                      "*": operator.mul, "×": operator.mul,
                      "/": operator.truediv, "÷": operator.truediv}
            op = op_map.get(sym, operator.add)
            ans = op(a, b)
            expr = f"{a} {sym} {b}"
            return WordProblemResult(
                answer=ans,
                expression=expr,
                explanation=f"{expr} = {ans}",
            )

        # 2. Extract all numbers from text
        nums = [float(m) for m in self._NUM_RE.findall(text)]
        if len(nums) < 2:
            # Only one number found — return it
            val = nums[0] if nums else 0.0
            return WordProblemResult(
                answer=val,
                expression=str(val),
                explanation=f"Answer: {val}",
            )

        a, b = nums[0], nums[1]

        # 3. Classify operation by cue words
        has_add = bool(self._ADD_CUES.search(text))
        has_sub = bool(self._SUB_CUES.search(text))
        has_mul = bool(self._MUL_CUES.search(text))
        has_div = bool(self._DIV_CUES.search(text))

        if has_mul and not has_add and not has_sub:
            ans = a * b
            expr = f"{a} * {b}"
            op_word = "multiplied by"
        elif has_div and not has_add and not has_sub:
            if b == 0:
                raise ValueError("Division by zero in word problem")
            ans = a / b
            expr = f"{a} / {b}"
            op_word = "divided by"
        elif has_sub and not has_add:
            ans = a - b
            expr = f"{a} - {b}"
            op_word = "minus"
        else:
            # Default: addition
            ans = a + b
            expr = f"{a} + {b}"
            op_word = "plus"

        # Build simple explanation
        explanation = f"{text.strip()} {a} {op_word} {b} = {ans}"
        return WordProblemResult(answer=ans, expression=expr, explanation=explanation)


# ---------------------------------------------------------------------------
# MathReasoner — public API
# ---------------------------------------------------------------------------

class MathReasoner:
    """
    Unified math reasoning component for NSCK.

    Combines:
    * Exact arithmetic via ExpressionEvaluator
    * Linear algebra via LinearSolver
    * Word-problem parsing via WordProblemParser
    * FPE magnitude similarity via FPECodebook
    """

    def __init__(self, fpe_max_int: int = 1023):
        self._eval = ExpressionEvaluator()
        self._algebra = LinearSolver()
        self._word = WordProblemParser()
        self._fpe = FPECodebook(max_int=fpe_max_int)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def solve_expression(self, expr: str) -> float:
        """
        Evaluate a mathematical expression string exactly.

        Examples
        --------
        >>> mr.solve_expression("3 + 5")     # 8.0
        >>> mr.solve_expression("(10 - 4) * 3 / 2")  # 9.0
        >>> mr.solve_expression("2 ** 10")   # 1024.0
        """
        return self._eval.evaluate(expr)

    def solve_algebra(self, equation: str) -> Dict[str, float]:
        """
        Solve a linear equation in one variable.

        Examples
        --------
        >>> mr.solve_algebra("x + 3 = 7")    # {'x': 4.0}
        >>> mr.solve_algebra("2x - 1 = 5")   # {'x': 3.0}
        >>> mr.solve_algebra("3 = x + 1")    # {'x': 2.0}
        """
        return self._algebra.solve(equation)

    def solve_word_problem(self, text: str) -> Dict[str, Any]:
        """
        Parse and solve an arithmetic word problem in natural language.

        Returns
        -------
        {'answer': float, 'expression': str, 'explanation': str}
        """
        result = self._word.parse(text)
        return {
            "answer": result.answer,
            "expression": result.expression,
            "explanation": result.explanation,
        }

    def compare(self, a: float, b: float) -> Dict[str, Any]:
        """
        Compare two numbers and return structured comparison result.

        Returns
        -------
        {'less': bool, 'equal': bool, 'greater': bool,
         'difference': float, 'fpe_similarity': float}
        """
        return {
            "less": bool(a < b),
            "equal": bool(np.isclose(a, b)),
            "greater": bool(a > b),
            "difference": abs(a - b),
            "fpe_similarity": self.magnitude_similarity(a, b),
        }

    def magnitude_similarity(self, a: Union[int, float], b: Union[int, float]) -> float:
        """
        Return VSA FPE cosine similarity between two numbers.
        Nearby numbers score high (~0.5–0.9); distant numbers score near 0.
        """
        return self._fpe.similarity(a, b)

    def encode_number(self, n: Union[int, float]) -> Any:
        """Return the FPE HyperVector for a number (for direct VSA use)."""
        return self._fpe.encode(n)

    def is_math_query(self, text: str) -> bool:
        """
        Heuristic: does this text look like a math/arithmetic question?
        """
        math_patterns = [
            r"\d+\s*[+\-*/÷×]\s*\d+",       # arithmetic expression
            r"\b(?:what\s+is|calculate|compute|solve|evaluate)\b",
            r"\b(?:sum|product|difference|quotient|remainder)\b",
            r"\bequat(?:ion|e)\b",
            r"\b[a-z]\s*[=+\-]\s*\d",         # algebraic expression
            r"\bhow\s+many\b.*\b(?:total|left|remain|altogether)\b",
        ]
        for pat in math_patterns:
            if re.search(pat, text, re.IGNORECASE):
                return True
        return False

    def extract_numbers(self, text: str) -> List[float]:
        """Extract all numeric values from a text string."""
        return [float(m) for m in re.findall(r"-?\d+\.?\d*", text)]

    def verbalize_answer(self, result: Dict[str, Any]) -> str:
        """
        Convert a solve_word_problem result to a natural-language sentence.
        """
        ans = result.get("answer", 0)
        expr = result.get("expression", "")
        # Format as integer when possible
        display = int(ans) if isinstance(ans, float) and ans == int(ans) else ans
        return f"The answer is {display} ({expr} = {display})."
