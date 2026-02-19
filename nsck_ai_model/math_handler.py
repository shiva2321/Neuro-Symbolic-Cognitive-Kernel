"""
NSCK Symbolic Math Handler
===========================

Pure symbolic math evaluation — NO neural networks.

Handles:
    - Arithmetic: +, -, *, /, **, %, parentheses
    - Word problems: extracting numbers and operations from natural language
    - Unit conversions: length, weight, temperature, time
    - Basic algebra: simple equation solving (ax + b = c)
    - Fractions and percentages
    - Comparisons and ordering
    - Sequences and patterns

All computation is symbolic / algorithmic. No matrix math, no training.
"""

import re
import math
import operator
from typing import Optional, Tuple, List, Dict, Any, Union
from dataclasses import dataclass
from fractions import Fraction
from collections import Counter, defaultdict


@dataclass
class MathResult:
    """Result of a math evaluation."""
    question: str
    answer: str
    numeric_value: Optional[float] = None
    steps: List[str] = None
    confidence: float = 1.0
    method: str = "symbolic"

    def __post_init__(self):
        if self.steps is None:
            self.steps = []


# ═══════════════════════════════════════════════════════════════════════════
# Safe arithmetic evaluator (no eval())
# ═══════════════════════════════════════════════════════════════════════════

class _TokenType:
    NUMBER = 'NUMBER'
    PLUS = '+'
    MINUS = '-'
    MUL = '*'
    DIV = '/'
    POW = '**'
    MOD = '%'
    LPAREN = '('
    RPAREN = ')'
    EOF = 'EOF'


class _Token:
    def __init__(self, type_, value):
        self.type = type_
        self.value = value


class _Lexer:
    """Tokenize a math expression."""

    def __init__(self, text: str):
        self.text = text.replace(' ', '')
        self.pos = 0

    def _number(self) -> _Token:
        start = self.pos
        while self.pos < len(self.text) and (
                self.text[self.pos].isdigit() or self.text[self.pos] == '.'):
            self.pos += 1
        return _Token(_TokenType.NUMBER, float(self.text[start:self.pos]))

    def next_token(self) -> _Token:
        if self.pos >= len(self.text):
            return _Token(_TokenType.EOF, None)

        ch = self.text[self.pos]

        if ch.isdigit() or (ch == '.' and self.pos + 1 < len(self.text)
                            and self.text[self.pos + 1].isdigit()):
            return self._number()

        self.pos += 1
        if ch == '+':
            return _Token(_TokenType.PLUS, '+')
        if ch == '-':
            return _Token(_TokenType.MINUS, '-')
        if ch == '*':
            if self.pos < len(self.text) and self.text[self.pos] == '*':
                self.pos += 1
                return _Token(_TokenType.POW, '**')
            return _Token(_TokenType.MUL, '*')
        if ch == '/':
            return _Token(_TokenType.DIV, '/')
        if ch == '%':
            return _Token(_TokenType.MOD, '%')
        if ch == '(':
            return _Token(_TokenType.LPAREN, '(')
        if ch == ')':
            return _Token(_TokenType.RPAREN, ')')
        if ch == '^':
            return _Token(_TokenType.POW, '**')

        raise ValueError(f"Unexpected character: {ch}")


class _Parser:
    """Recursive descent parser for arithmetic expressions.

    Grammar:
        expr   → term (('+' | '-') term)*
        term   → power (('*' | '/' | '%') power)*
        power  → unary ('**' unary)*
        unary  → ('-' | '+') unary | atom
        atom   → NUMBER | '(' expr ')'
    """

    def __init__(self, lexer: _Lexer):
        self.lexer = lexer
        self.current = self.lexer.next_token()

    def _eat(self, token_type):
        if self.current.type == token_type:
            val = self.current.value
            self.current = self.lexer.next_token()
            return val
        raise ValueError(
            f"Expected {token_type}, got {self.current.type}")

    def parse(self) -> float:
        result = self._expr()
        if self.current.type != _TokenType.EOF:
            raise ValueError("Unexpected token after expression")
        return result

    def _expr(self) -> float:
        result = self._term()
        while self.current.type in (_TokenType.PLUS, _TokenType.MINUS):
            if self.current.type == _TokenType.PLUS:
                self._eat(_TokenType.PLUS)
                result += self._term()
            else:
                self._eat(_TokenType.MINUS)
                result -= self._term()
        return result

    def _term(self) -> float:
        result = self._power()
        while self.current.type in (
                _TokenType.MUL, _TokenType.DIV, _TokenType.MOD):
            if self.current.type == _TokenType.MUL:
                self._eat(_TokenType.MUL)
                result *= self._power()
            elif self.current.type == _TokenType.DIV:
                self._eat(_TokenType.DIV)
                divisor = self._power()
                if divisor == 0:
                    raise ValueError("Division by zero")
                result /= divisor
            else:
                self._eat(_TokenType.MOD)
                result %= self._power()
        return result

    def _power(self) -> float:
        base = self._unary()
        if self.current.type == _TokenType.POW:
            self._eat(_TokenType.POW)
            exp = self._unary()
            return base ** exp
        return base

    def _unary(self) -> float:
        if self.current.type == _TokenType.MINUS:
            self._eat(_TokenType.MINUS)
            return -self._unary()
        if self.current.type == _TokenType.PLUS:
            self._eat(_TokenType.PLUS)
            return self._unary()
        return self._atom()

    def _atom(self) -> float:
        if self.current.type == _TokenType.NUMBER:
            return self._eat(_TokenType.NUMBER)
        if self.current.type == _TokenType.LPAREN:
            self._eat(_TokenType.LPAREN)
            result = self._expr()
            self._eat(_TokenType.RPAREN)
            return result
        raise ValueError(
            f"Unexpected token: {self.current.type}")


def safe_eval_math(expr: str) -> float:
    """Safely evaluate a math expression without using eval().

    >>> safe_eval_math("2 + 3 * 4")
    14.0
    >>> safe_eval_math("(10 - 3) ** 2")
    49.0
    """
    lexer = _Lexer(expr)
    parser = _Parser(lexer)
    return parser.parse()


# ═══════════════════════════════════════════════════════════════════════════
# Word-number mapping
# ═══════════════════════════════════════════════════════════════════════════

_WORD_TO_NUM = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
    'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
    'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
    'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
    'eighteen': 18, 'nineteen': 19, 'twenty': 20,
    'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60,
    'seventy': 70, 'eighty': 80, 'ninety': 90,
    'hundred': 100, 'thousand': 1000, 'million': 1_000_000,
    'billion': 1_000_000_000,
    'a': 1, 'an': 1, 'first': 1, 'second': 2, 'third': 3,
    'half': 0.5, 'quarter': 0.25, 'dozen': 12, 'pair': 2,
    'couple': 2, 'score': 20, 'gross': 144,
}

# Seed operator words — used as fallback when no learned mapping exists.
# The LearnableOperatorDetector can override these from training data.
_SEED_OP_WORDS: Dict[str, str] = {
    'plus': '+', 'add': '+', 'added': '+', 'sum': '+',
    'and': '+', 'combined': '+', 'total': '+', 'together': '+',
    'minus': '-', 'subtract': '-', 'less': '-', 'difference': '-',
    'take away': '-', 'fewer': '-', 'removed': '-',
    'times': '*', 'multiply': '*', 'multiplied': '*', 'product': '*',
    'of': '*',
    'divided': '/', 'over': '/', 'split': '/',
    'remainder': '%', 'modulo': '%',
    'squared': '**2', 'cubed': '**3',
    'power': '**', 'raised': '**', 'exponent': '**',
    'percent': '%_of', 'percentage': '%_of',
}

# Backward compatibility alias
_OP_WORDS = _SEED_OP_WORDS


# ═══════════════════════════════════════════════════════════════════════════
# Learnable Operator Detector
# ═══════════════════════════════════════════════════════════════════════════

class LearnableOperatorDetector:
    """Learns to map natural-language words to arithmetic operators.

    Instead of relying on a fixed dictionary, this class discovers
    operator words from training data.  When the model sees
    ``"3 plus 5 equals 8"``, it reverse-engineers that ``"plus"`` means
    addition and stores that mapping.

    Learned mappings take priority; ``_SEED_OP_WORDS`` is the fallback.
    """

    def __init__(self):
        # word → {operator → observation_count}
        self._word_op_counts: Dict[str, Counter] = defaultdict(Counter)
        # Confident learned mappings (word → operator)
        self.learned_ops: Dict[str, str] = {}
        # Minimum observations before trusting a learned mapping
        self._min_examples: int = 2
        # Words that should never be treated as operators
        self._stop_words: frozenset = frozenset({
            'is', 'are', 'was', 'the', 'a', 'an', 'to', 'of', 'in',
            'for', 'it', 'equals', 'equal', 'what', 'how', 'find',
            'answer', 'result', 'if', 'then', 'by', 'from', 'with',
        })

    # ── Learning ──

    def learn_from_example(self, text: str, numbers: List[float],
                           result: float, operation: str):
        """Learn operator words from a solved math example.

        Finds words positioned between the operand numbers and records
        them as potential operator indicators for ``operation``.
        """
        words = re._compile(r'[^\w\s]').sub(' ', text.lower()).split()
        number_strs = set()
        for n in list(numbers) + [result]:
            number_strs.add(str(int(n)) if n == int(n) else str(n))
            number_strs.add(f"{n:.1f}")

        for w in words:
            if w in number_strs or w in _WORD_TO_NUM or w in self._stop_words:
                continue
            if len(w) < 2 or w.isdigit():
                continue
            self._word_op_counts[w][operation] += 1

        self._rebuild()

    def learn_from_text(self, text: str):
        """Auto-learn from sentences containing math patterns.

        Scans for ``"N1 WORD N2 equals/is N3"`` and deduces which
        arithmetic operation ``WORD`` represents.
        """
        patterns = [
            r'(\d+(?:\.\d+)?)\s+(\w+)\s+(\d+(?:\.\d+)?)\s+(?:equals?|is|=)\s+(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s+(\w+)\s+(?:by\s+)?(\d+(?:\.\d+)?)\s+(?:equals?|is|=)\s+(\d+(?:\.\d+)?)',
        ]
        for pat in patterns:
            for m in re.finditer(pat, text.lower()):
                n1 = float(m.group(1))
                word = m.group(2)
                n2 = float(m.group(3))
                res = float(m.group(4))
                op = self._deduce_operation(n1, n2, res)
                if op and word not in self._stop_words:
                    self._word_op_counts[word][op] += 1
        self._rebuild()

    def learn_from_word_problem(self, text: str, answer: float):
        """Learn from a word problem whose answer is known.

        Extracts numbers, checks all four operations, and records
        context words for the matching operation.
        """
        numbers = [float(n) for n in re.findall(r'-?\d+(?:\.\d+)?', text)]
        if len(numbers) < 2:
            return
        a, b = numbers[0], numbers[1]
        op = self._deduce_operation(a, b, answer)
        if op:
            words = re._compile(r'[^\w\s]').sub(' ', text.lower()).split()
            number_strs = {str(int(n)) if n == int(n) else str(n)
                           for n in numbers + [answer]}
            for w in words:
                if w in number_strs or w in _WORD_TO_NUM or w in self._stop_words:
                    continue
                if len(w) < 3:
                    continue
                self._word_op_counts[w][op] += 1
            self._rebuild()

    # ── Querying ──

    def get_operator(self, word: str) -> Optional[str]:
        """Return the operator symbol for *word*.

        Checks learned mappings first, then falls back to seed words.
        """
        wl = word.lower()
        if wl in self.learned_ops:
            return self.learned_ops[wl]
        return _SEED_OP_WORDS.get(wl)

    def get_all_operators(self) -> Dict[str, str]:
        """Return the combined operator mapping (learned + seed)."""
        combined = dict(_SEED_OP_WORDS)
        combined.update(self.learned_ops)
        return combined

    def get_stats(self) -> Dict[str, Any]:
        """Return learning statistics."""
        return {
            'learned_operators': len(self.learned_ops),
            'seed_operators': len(_SEED_OP_WORDS),
            'total_observations': sum(
                sum(c.values()) for c in self._word_op_counts.values()),
            'learned_words': dict(self.learned_ops),
        }

    # ── Internals ──

    @staticmethod
    def _deduce_operation(a: float, b: float,
                          result: float) -> Optional[str]:
        """Deduce which operation maps ``(a, b) → result``."""
        eps = 0.001
        if abs(a + b - result) < eps:
            return '+'
        if abs(a - b - result) < eps:
            return '-'
        if b != 0 and abs(a * b - result) < eps:
            return '*'
        if b != 0 and abs(a / b - result) < eps:
            return '/'
        if abs(b - a - result) < eps:
            return '-'
        return None

    def _rebuild(self):
        """Rebuild confident operator mappings from observation counts."""
        self.learned_ops.clear()
        for word, ops in self._word_op_counts.items():
            total = sum(ops.values())
            if total < self._min_examples:
                continue
            best_op, count = ops.most_common(1)[0]
            if count / total >= 0.6:
                self.learned_ops[word] = best_op


def _words_to_number(text: str) -> Optional[float]:
    """Convert a word number like 'twenty three' to 23."""
    text = text.strip().lower()
    if not text:
        return None

    # Direct number
    try:
        return float(text.replace(',', ''))
    except ValueError:
        pass

    words = text.split()
    total = 0
    current = 0

    for w in words:
        w = w.strip(',').strip()
        if w in _WORD_TO_NUM:
            val = _WORD_TO_NUM[w]
            if val == 100:
                current = (current or 1) * 100
            elif val >= 1000:
                current = (current or 1) * val
                total += current
                current = 0
            else:
                current += val
        elif w == 'and':
            continue
        else:
            return None

    total += current
    return float(total) if total > 0 else None


# ═══════════════════════════════════════════════════════════════════════════
# Unit conversions
# ═══════════════════════════════════════════════════════════════════════════

_CONVERSIONS = {
    # Length (to metres)
    'km': ('m', 1000), 'kilometer': ('m', 1000), 'kilometres': ('m', 1000),
    'meters': ('m', 1), 'meter': ('m', 1), 'metres': ('m', 1),
    'm': ('m', 1),
    'cm': ('m', 0.01), 'centimeter': ('m', 0.01),
    'mm': ('m', 0.001), 'millimeter': ('m', 0.001),
    'mile': ('m', 1609.344), 'miles': ('m', 1609.344),
    'yard': ('m', 0.9144), 'yards': ('m', 0.9144),
    'foot': ('m', 0.3048), 'feet': ('m', 0.3048), 'ft': ('m', 0.3048),
    'inch': ('m', 0.0254), 'inches': ('m', 0.0254), 'in': ('m', 0.0254),
    # Weight (to kg)
    'kg': ('kg', 1), 'kilogram': ('kg', 1), 'kilograms': ('kg', 1),
    'gram': ('kg', 0.001), 'grams': ('kg', 0.001), 'g': ('kg', 0.001),
    'mg': ('kg', 0.000001), 'milligram': ('kg', 0.000001),
    'pound': ('kg', 0.453592), 'pounds': ('kg', 0.453592),
    'lb': ('kg', 0.453592), 'lbs': ('kg', 0.453592),
    'ounce': ('kg', 0.0283495), 'ounces': ('kg', 0.0283495),
    'oz': ('kg', 0.0283495),
    'ton': ('kg', 907.185), 'tons': ('kg', 907.185),
    'tonne': ('kg', 1000), 'tonnes': ('kg', 1000),
    # Temperature (special handling)
    'celsius': ('temp', 'C'), 'fahrenheit': ('temp', 'F'),
    'kelvin': ('temp', 'K'),
    'c': ('temp', 'C'), 'f': ('temp', 'F'), 'k': ('temp', 'K'),
    # Time (to seconds)
    'second': ('s', 1), 'seconds': ('s', 1), 'sec': ('s', 1),
    'minute': ('s', 60), 'minutes': ('s', 60), 'min': ('s', 60),
    'hour': ('s', 3600), 'hours': ('s', 3600), 'hr': ('s', 3600),
    'day': ('s', 86400), 'days': ('s', 86400),
    'week': ('s', 604800), 'weeks': ('s', 604800),
    'month': ('s', 2592000), 'months': ('s', 2592000),
    'year': ('s', 31536000), 'years': ('s', 31536000),
}


def _convert_temperature(value: float, from_unit: str,
                          to_unit: str) -> float:
    """Convert temperature between C, F, K."""
    # First convert to Celsius
    if from_unit == 'F':
        celsius = (value - 32) * 5 / 9
    elif from_unit == 'K':
        celsius = value - 273.15
    else:
        celsius = value

    # Then convert to target
    if to_unit == 'F':
        return celsius * 9 / 5 + 32
    elif to_unit == 'K':
        return celsius + 273.15
    return celsius


# ═══════════════════════════════════════════════════════════════════════════
# Math question classifier
# ═══════════════════════════════════════════════════════════════════════════

def is_math_question(text: str) -> bool:
    """Determine if a text is a math question."""
    text_lower = text.lower().strip()

    # Explicit math expressions
    if re.search(r'\d+\s*[\+\-\*/\^%]\s*\d+', text):
        return True

    # Math keywords
    math_words = [
        'calculate', 'compute', 'solve', 'evaluate', 'what is',
        'how much', 'how many', 'sum of', 'product of', 'difference',
        'divided by', 'multiplied by', 'plus', 'minus', 'times',
        'percent', 'percentage', 'fraction', 'ratio', 'average',
        'mean', 'median', 'square root', 'factorial', 'equation',
        'convert', 'conversion', 'area', 'perimeter', 'volume',
        'circumference', 'diameter', 'radius',
    ]

    for word in math_words:
        if word in text_lower:
            return True

    # Contains numbers with question words
    if re.search(r'\d', text) and re.search(
            r'(what|how|find|equal|result|answer)', text_lower):
        return True

    return False


# ═══════════════════════════════════════════════════════════════════════════
# Main Math Handler
# ═══════════════════════════════════════════════════════════════════════════

class MathHandler:
    """Symbolic math problem solver.

    Handles arithmetic, word problems, conversions, and basic algebra.
    All computation is symbolic — no neural networks involved.

    Contains a ``LearnableOperatorDetector`` that progressively learns
    operator words from training data.  The more math examples fed via
    ``learn()``, the less it depends on the seed dictionary.

    Usage::

        handler = MathHandler()
        handler.learn("3 plus 5 equals 8")   # learns "plus" → +
        result = handler.solve("What is 25 * 4 + 10?")
        print(result.answer)    # "110"
        print(result.steps)     # ["25 * 4 + 10", "= 110.0"]
    """

    def __init__(self):
        self.operator_detector = LearnableOperatorDetector()

    def learn(self, text: str):
        """Learn math patterns from training text.

        Discovers operator words, number words, and math relationships
        from examples like ``"3 plus 5 equals 8"`` or GSM8K solutions.
        """
        self.operator_detector.learn_from_text(text)

    def learn_from_word_problem(self, text: str, answer: float):
        """Learn operator words from a word problem with a known answer."""
        self.operator_detector.learn_from_word_problem(text, answer)

    def get_learning_stats(self) -> Dict[str, Any]:
        """Return statistics about what the math handler has learned."""
        return self.operator_detector.get_stats()

    def solve(self, question: str) -> MathResult:
        """Attempt to solve a math question."""
        question = question.strip()

        # Try each method in order of specificity.
        # Percentage, conversion, and sequence MUST come before
        # word_problem to avoid false matches.
        for method in [
            self._try_unit_conversion,
            self._try_percentage,
            self._try_fraction,
            self._try_basic_algebra,
            self._try_direct_arithmetic,
            self._try_sequence,
            self._try_comparison,
            self._try_word_problem,
        ]:
            result = method(question)
            if result is not None:
                return result

        return MathResult(
            question=question,
            answer="I could not solve this math problem.",
            confidence=0.0,
            method="none")

    # ── Direct arithmetic ──

    def _try_direct_arithmetic(self, q: str) -> Optional[MathResult]:
        """Try to evaluate a direct arithmetic expression."""
        # Extract expression from question
        expr = q.lower()
        # Reject if it looks like a conversion, sequence, or complex query
        if any(kw in expr for kw in [
                'convert', 'celsius', 'fahrenheit', 'kelvin',
                'sequence', 'pattern', 'series', 'next number',
                'miles', 'kilometers', 'pounds', 'kilograms',
                'percent of', '% of']):
            return None

        # Remove question framing
        for prefix in ['what is', 'calculate', 'compute', 'evaluate',
                        'solve', 'find', 'what\'s', 'whats']:
            if expr.startswith(prefix):
                expr = expr[len(prefix):]
        expr = expr.strip().rstrip('?').rstrip('.').strip()

        # Replace word operators — use learned mappings first, then seed
        all_ops = self.operator_detector.get_all_operators()
        for word, sym in sorted(all_ops.items(), key=lambda x: -len(x[0])):
            if word in expr:
                expr = expr.replace(word, sym)
        # Common aliases that aren't in operator dict
        expr = expr.replace('x', '*')
        expr = expr.replace('to the power of', '**')

        # Clean up
        expr = re.sub(r'[^0-9+\-*/().%\s\^]', '', expr).strip()

        if not expr or not re.search(r'\d', expr):
            return None

        # Reject if only a single number (no operator)
        if re.match(r'^-?\d+\.?\d*$', expr):
            return None

        try:
            result = safe_eval_math(expr)
            # Format nicely
            if result == int(result):
                answer = str(int(result))
            else:
                answer = f"{result:.6g}"

            return MathResult(
                question=q, answer=answer,
                numeric_value=result,
                steps=[expr, f"= {answer}"],
                confidence=0.95,
                method="arithmetic")
        except (ValueError, ZeroDivisionError, OverflowError):
            return None

    # ── Unit conversion ──

    def _try_unit_conversion(self, q: str) -> Optional[MathResult]:
        """Try to solve a unit conversion problem."""
        q_lower = q.lower()
        if 'convert' not in q_lower and ' to ' not in q_lower and \
                ' in ' not in q_lower:
            return None

        # Pattern: "convert X unit1 to unit2" or "X unit1 in unit2"
        patterns = [
            r'convert\s+([\d.]+)\s+(\w+)\s+to\s+(\w+)',
            r'([\d.]+)\s+(\w+)\s+(?:to|in|into)\s+(\w+)',
            r'how\s+many\s+(\w+)\s+(?:in|are\s+in)\s+([\d.]+)\s+(\w+)',
        ]

        for pattern in patterns:
            m = re.search(pattern, q_lower)
            if not m:
                continue

            groups = m.groups()
            if pattern.startswith('how'):
                # "how many X in Y Z" → convert Y Z to X
                to_unit, value_str, from_unit = groups
            else:
                value_str, from_unit, to_unit = groups

            try:
                value = float(value_str)
            except ValueError:
                continue

            from_info = _CONVERSIONS.get(from_unit)
            to_info = _CONVERSIONS.get(to_unit)

            if not from_info or not to_info:
                continue

            # Temperature special case
            if from_info[0] == 'temp' and to_info[0] == 'temp':
                result = _convert_temperature(
                    value, from_info[1], to_info[1])
                if result == int(result):
                    answer = str(int(result))
                else:
                    answer = f"{result:.2f}"
                return MathResult(
                    question=q,
                    answer=f"{answer} {to_unit}",
                    numeric_value=result,
                    steps=[
                        f"{value} {from_unit} → {to_unit}",
                        f"= {answer} {to_unit}"],
                    confidence=0.9,
                    method="unit_conversion")

            # Same dimension conversion
            if from_info[0] != to_info[0]:
                continue

            # Convert: from → base → to
            base_value = value * from_info[1]
            result = base_value / to_info[1]

            if result == int(result):
                answer = str(int(result))
            else:
                answer = f"{result:.4g}"

            return MathResult(
                question=q,
                answer=f"{answer} {to_unit}",
                numeric_value=result,
                steps=[
                    f"{value} {from_unit}",
                    f"= {value * from_info[1]:.4g} {from_info[0]}",
                    f"= {answer} {to_unit}"],
                confidence=0.9,
                method="unit_conversion")

        return None

    # ── Percentage ──

    def _try_percentage(self, q: str) -> Optional[MathResult]:
        """Solve percentage problems."""
        q_lower = q.lower()

        # "what is X% of Y"
        m = re.search(
            r'(?:what\s+is\s+)?(\d+(?:\.\d+)?)\s*%\s*of\s+(\d+(?:\.\d+)?)',
            q_lower)
        if m:
            pct, value = float(m.group(1)), float(m.group(2))
            result = pct / 100 * value
            answer = f"{result:.4g}" if result != int(result) else str(int(result))
            return MathResult(
                question=q, answer=answer,
                numeric_value=result,
                steps=[
                    f"{pct}% of {value}",
                    f"= {pct}/100 × {value}",
                    f"= {answer}"],
                confidence=0.95,
                method="percentage")

        # "X is what percent of Y"
        m = re.search(
            r'(\d+(?:\.\d+)?)\s+is\s+what\s+percent\s+of\s+(\d+(?:\.\d+)?)',
            q_lower)
        if m:
            part, whole = float(m.group(1)), float(m.group(2))
            if whole == 0:
                return None
            result = (part / whole) * 100
            answer = f"{result:.2f}%"
            return MathResult(
                question=q, answer=answer,
                numeric_value=result,
                steps=[
                    f"{part} / {whole} × 100",
                    f"= {answer}"],
                confidence=0.95,
                method="percentage")

        return None

    # ── Word problems ──

    def _try_word_problem(self, q: str) -> Optional[MathResult]:
        """Extract numbers and operations from word problems.

        Uses the ``LearnableOperatorDetector`` to discover operator hints.
        Falls back to keyword heuristics when the detector has no mapping.
        """
        q_lower = q.lower()

        # Extract all numbers
        numbers = [float(n) for n in re.findall(
            r'-?\d+(?:\.\d+)?', q)]
        if len(numbers) < 2:
            return None

        steps = [f"Numbers found: {numbers}"]

        # --- Strategy 1: Check learned operator words in query ---
        detected_op = None
        for word in re.sub(r'[^\w\s]', ' ', q_lower).split():
            op = self.operator_detector.get_operator(word)
            if op and op in ('+', '-', '*', '/'):
                detected_op = op
                steps.append(f"Learned operator: '{word}' → {op}")
                break

        # --- Strategy 2: Keyword heuristics (fallback) ---
        if detected_op is None:
            if any(w in q_lower for w in [
                    'total', 'altogether', 'combined', 'sum',
                    'many does.*have', 'how many.*all',
                    'in all', 'together']):
                if 'each' in q_lower and 'bought' in q_lower or \
                        'groups' in q_lower:
                    detected_op = '*'
                elif any(w in q_lower for w in [
                        'gave away', 'spent', 'lost', 'ate', 'sold',
                        'used', 'less', 'fewer', 'left']):
                    detected_op = '-'
                else:
                    detected_op = '+'
            elif any(w in q_lower for w in [
                    'left', 'remain', 'remaining', 'still has',
                    'difference', 'fewer', 'less than', 'after']):
                detected_op = '-'
            elif any(w in q_lower for w in [
                    'each', 'per', 'every', 'groups of',
                    'rows of', 'sets of']):
                if any(w in q_lower for w in [
                        'divide', 'split', 'share', 'distribute',
                        'each get', 'per person', 'how many groups']):
                    detected_op = '/'
                else:
                    detected_op = '*'
            elif any(w in q_lower for w in [
                    'divide', 'split', 'share equally',
                    'distributed', 'ratio']):
                detected_op = '/'
            elif any(w in q_lower for w in ['times', 'multiply', 'product']):
                detected_op = '*'
            elif any(w in q_lower for w in ['average', 'mean']):
                detected_op = 'avg'
            else:
                detected_op = '+'  # default

        # --- Compute result ---
        if detected_op == '+':
            result = sum(numbers)
            op_name = "addition"
        elif detected_op == '-':
            result = numbers[0] - sum(numbers[1:])
            op_name = "subtraction"
        elif detected_op == '*':
            result = numbers[0] * numbers[1]
            op_name = "multiplication"
        elif detected_op == '/':
            if numbers[1] == 0:
                return None
            result = numbers[0] / numbers[1]
            op_name = "division"
        elif detected_op == 'avg':
            result = sum(numbers) / len(numbers)
            op_name = "average"
        else:
            result = sum(numbers)
            op_name = "addition (default)"

        steps.append(f"Operation: {op_name}")
        if result == int(result):
            answer = str(int(result))
        else:
            answer = f"{result:.4g}"
        steps.append(f"= {answer}")

        # Auto-learn: record what words correlated with this operation
        self.operator_detector.learn_from_example(
            q, numbers, result, detected_op if detected_op != 'avg' else '+')

        return MathResult(
            question=q, answer=answer,
            numeric_value=float(result),
            steps=steps,
            confidence=0.7,
            method="word_problem")

    # ── Sequences ──

    def _try_sequence(self, q: str) -> Optional[MathResult]:
        """Find the next number in a sequence."""
        q_lower = q.lower()
        if 'next' not in q_lower and 'sequence' not in q_lower and \
                'pattern' not in q_lower and 'series' not in q_lower:
            return None

        numbers = [float(n) for n in re.findall(
            r'-?\d+(?:\.\d+)?', q)]
        if len(numbers) < 3:
            return None

        steps = [f"Sequence: {numbers}"]

        # Check arithmetic sequence (constant difference)
        diffs = [numbers[i+1] - numbers[i]
                 for i in range(len(numbers)-1)]
        if len(set(diffs)) == 1:
            next_val = numbers[-1] + diffs[0]
            steps.append(f"Arithmetic sequence, d = {diffs[0]}")
            answer = str(int(next_val)) if next_val == int(next_val) \
                else f"{next_val:.4g}"
            steps.append(f"Next = {answer}")
            return MathResult(
                question=q, answer=answer,
                numeric_value=next_val,
                steps=steps,
                confidence=0.9,
                method="sequence")

        # Check geometric sequence (constant ratio)
        if all(n != 0 for n in numbers[:-1]):
            ratios = [numbers[i+1] / numbers[i]
                      for i in range(len(numbers)-1)]
            if len(set([round(r, 6) for r in ratios])) == 1:
                next_val = numbers[-1] * ratios[0]
                steps.append(f"Geometric sequence, r = {ratios[0]}")
                answer = str(int(next_val)) if next_val == int(next_val) \
                    else f"{next_val:.4g}"
                steps.append(f"Next = {answer}")
                return MathResult(
                    question=q, answer=answer,
                    numeric_value=next_val,
                    steps=steps,
                    confidence=0.85,
                    method="sequence")

        # Check quadratic (constant second difference)
        if len(diffs) >= 2:
            second_diffs = [diffs[i+1] - diffs[i]
                            for i in range(len(diffs)-1)]
            if len(set(second_diffs)) == 1:
                next_diff = diffs[-1] + second_diffs[0]
                next_val = numbers[-1] + next_diff
                steps.append(
                    f"Quadratic sequence, 2nd diff = {second_diffs[0]}")
                answer = str(int(next_val)) if next_val == int(next_val) \
                    else f"{next_val:.4g}"
                steps.append(f"Next = {answer}")
                return MathResult(
                    question=q, answer=answer,
                    numeric_value=next_val,
                    steps=steps,
                    confidence=0.8,
                    method="sequence")

        return None

    # ── Comparisons ──

    def _try_comparison(self, q: str) -> Optional[MathResult]:
        """Compare numbers or compute differences."""
        q_lower = q.lower()
        if not any(w in q_lower for w in [
                'greater', 'larger', 'bigger', 'smaller',
                'less', 'more than', 'compare', 'which is']):
            return None

        numbers = [float(n) for n in re.findall(
            r'-?\d+(?:\.\d+)?', q)]
        if len(numbers) < 2:
            return None

        if 'how much' in q_lower:
            diff = abs(numbers[0] - numbers[1])
            answer = str(int(diff)) if diff == int(diff) \
                else f"{diff:.4g}"
            return MathResult(
                question=q,
                answer=f"The difference is {answer}.",
                numeric_value=diff,
                steps=[f"|{numbers[0]} - {numbers[1]}| = {answer}"],
                confidence=0.85,
                method="comparison")

        if numbers[0] > numbers[1]:
            answer = f"{numbers[0]} is greater than {numbers[1]}."
        elif numbers[0] < numbers[1]:
            answer = f"{numbers[1]} is greater than {numbers[0]}."
        else:
            answer = f"{numbers[0]} and {numbers[1]} are equal."

        return MathResult(
            question=q, answer=answer,
            confidence=0.9,
            method="comparison")

    # ── Fractions ──

    def _try_fraction(self, q: str) -> Optional[MathResult]:
        """Handle fraction operations."""
        q_lower = q.lower()
        if 'fraction' not in q_lower and '/' not in q and \
                'half' not in q_lower and 'third' not in q_lower and \
                'quarter' not in q_lower:
            return None

        # Pattern: X/Y + A/B etc.
        fractions = re.findall(r'(\d+)/(\d+)', q)
        if len(fractions) >= 2:
            f1 = Fraction(int(fractions[0][0]), int(fractions[0][1]))
            f2 = Fraction(int(fractions[1][0]), int(fractions[1][1]))

            if '+' in q or 'add' in q_lower or 'sum' in q_lower:
                result = f1 + f2
                op = '+'
            elif '-' in q or 'subtract' in q_lower:
                result = f1 - f2
                op = '-'
            elif '*' in q or 'times' in q_lower or 'multiply' in q_lower:
                result = f1 * f2
                op = '×'
            elif 'divide' in q_lower:
                if f2 == 0:
                    return None
                result = f1 / f2
                op = '÷'
            else:
                result = f1 + f2
                op = '+'

            answer = str(result)
            return MathResult(
                question=q, answer=answer,
                numeric_value=float(result),
                steps=[f"{f1} {op} {f2} = {result}"],
                confidence=0.9,
                method="fraction")

        return None

    # ── Basic algebra ──

    def _try_basic_algebra(self, q: str) -> Optional[MathResult]:
        """Solve simple equations like ax + b = c."""
        q_lower = q.lower()
        if 'solve' not in q_lower and '=' not in q:
            return None

        # Pattern: ax + b = c  or  ax - b = c
        m = re.search(
            r'(-?\d*)\s*[x]\s*([+-])\s*(\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)',
            q)
        if m:
            a_str, op, b_str, c_str = m.groups()
            a = float(a_str) if a_str and a_str != '-' else (
                -1.0 if a_str == '-' else 1.0)
            b = float(b_str)
            c = float(c_str)

            if op == '-':
                b = -b

            if a == 0:
                return None

            x = (c - b) / a
            answer = str(int(x)) if x == int(x) else f"{x:.4g}"

            return MathResult(
                question=q, answer=f"x = {answer}",
                numeric_value=x,
                steps=[
                    f"{a}x + {b} = {c}",
                    f"{a}x = {c} - {b}",
                    f"{a}x = {c - b}",
                    f"x = {answer}"],
                confidence=0.85,
                method="algebra")

        # Pattern: x + a = b
        m = re.search(
            r'[x]\s*([+-])\s*(\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)',
            q)
        if m:
            op, a_str, b_str = m.groups()
            a = float(a_str)
            b = float(b_str)

            if op == '+':
                x = b - a
            else:
                x = b + a

            answer = str(int(x)) if x == int(x) else f"{x:.4g}"
            return MathResult(
                question=q, answer=f"x = {answer}",
                numeric_value=x,
                steps=[f"x {op} {a} = {b}", f"x = {answer}"],
                confidence=0.85,
                method="algebra")

        return None


# ═══════════════════════════════════════════════════════════════════════════
# Convenience
# ═══════════════════════════════════════════════════════════════════════════

_HANDLER = MathHandler()


def solve_math(question: str) -> MathResult:
    """Convenience entry point."""
    return _HANDLER.solve(question)
