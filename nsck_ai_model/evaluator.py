"""
NSCK Model Evaluator
=====================

Real-world evaluation benchmarks for the NSCK AI model.
Tests text understanding, knowledge retrieval, math, reasoning,
conversation quality, and image understanding.

All metrics are computed without neural networks.
"""

import re
import time
import logging
import random
from typing import Dict, Any, List, Tuple, Optional
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger("nsck_ai.evaluator")


# ═══════════════════════════════════════════════════════════════════════════
# Scoring Utilities
# ═══════════════════════════════════════════════════════════════════════════

def _tokenize(text: str) -> List[str]:
    """Simple whitespace tokenizer."""
    return re.sub(r'[^\w\s]', ' ', text.lower()).split()


def _jaccard(a: set, b: set) -> float:
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def _overlap_score(reference: str, candidate: str) -> float:
    """Word overlap score between reference and candidate (recall)."""
    ref_words = set(_tokenize(reference))
    cand_words = set(_tokenize(candidate))
    if not ref_words:
        return 0.0
    return len(ref_words & cand_words) / len(ref_words)


def _f1_score(reference: str, candidate: str) -> float:
    """Token-level F1 score."""
    ref_words = set(_tokenize(reference))
    cand_words = set(_tokenize(candidate))
    if not ref_words or not cand_words:
        return 0.0
    common = ref_words & cand_words
    if not common:
        return 0.0
    precision = len(common) / len(cand_words)
    recall = len(common) / len(ref_words)
    return 2 * precision * recall / (precision + recall)


def _fuzzy_contains(text: str, target: str, threshold: float = 0.6) -> bool:
    """Check if text approximately contains the target phrase."""
    text_lower = text.lower()
    target_lower = target.lower().strip()

    # Exact substring
    if target_lower in text_lower:
        return True

    # Word-level check
    target_words = set(_tokenize(target_lower))
    text_words = set(_tokenize(text_lower))
    if target_words and target_words <= text_words:
        return True

    # Fuzzy: most words present
    if target_words:
        present = len(target_words & text_words) / len(target_words)
        if present >= threshold:
            return True

    return False


def _numeric_close(a: float, b: float, tolerance: float = 0.01) -> bool:
    """Check if two numbers are close."""
    if a == 0 and b == 0:
        return True
    if a == 0 or b == 0:
        return abs(a - b) <= tolerance
    return abs(a - b) / max(abs(a), abs(b)) <= tolerance


# ═══════════════════════════════════════════════════════════════════════════
# Benchmark Results
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BenchmarkResult:
    """Result of a single evaluation question."""
    question: str
    expected: str
    actual: str
    correct: bool
    score: float       # 0.0 to 1.0
    category: str
    latency_ms: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CategoryScore:
    """Aggregated score for a category."""
    category: str
    total: int
    correct: int
    accuracy: float
    avg_score: float
    avg_latency_ms: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalReport:
    """Complete evaluation report."""
    timestamp: float
    total_questions: int
    total_correct: int
    overall_accuracy: float
    overall_avg_score: float
    categories: Dict[str, CategoryScore]
    results: List[BenchmarkResult]
    training_stats: Dict[str, Any]
    duration_s: float

    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"NSCK Model Evaluation Report",
            f"{'='*60}",
            f"  Total questions:  {self.total_questions}",
            f"  Correct:          {self.total_correct}",
            f"  Accuracy:         {self.overall_accuracy:.1%}",
            f"  Average score:    {self.overall_avg_score:.3f}",
            f"  Duration:         {self.duration_s:.1f}s",
            f"",
            f"Category breakdown:",
        ]
        for cat, score in sorted(self.categories.items()):
            lines.append(f"  {cat:30s}  "
                         f"{score.accuracy:6.1%}  "
                         f"({score.correct}/{score.total})  "
                         f"avg_score={score.avg_score:.3f}  "
                         f"avg_lat={score.avg_latency_ms:.0f}ms")
        lines.append(f"{'='*60}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Built-in Evaluation Benchmarks
# ═══════════════════════════════════════════════════════════════════════════

# These test REAL knowledge with REAL expected answers.
# Each tuple: (question, expected_answer, category, difficulty)

_FACTUAL_QA = [
    ("What is the capital of France?", "Paris", "factual", "easy"),
    ("What planet is closest to the Sun?", "Mercury", "factual", "easy"),
    ("What is the largest ocean on Earth?", "Pacific", "factual", "easy"),
    ("What is the chemical symbol for water?", "H2O", "factual", "easy"),
    ("How many continents are there?", "7", "factual", "easy"),
    ("What is the tallest mountain on Earth?", "Everest", "factual", "medium"),
    ("Who developed the theory of relativity?", "Einstein", "factual", "medium"),
    ("What is the longest river in the world?", "Nile", "factual", "medium"),
    ("What is photosynthesis?", "plants convert carbon dioxide and water into glucose and oxygen using sunlight", "factual", "medium"),
    ("What are Newton's three laws?", "objects at rest stay at rest, force equals mass times acceleration, every action has an equal and opposite reaction", "factual", "hard"),
    ("What is the Amazon Rainforest known for?", "produces about 20 percent of the world's oxygen", "factual", "medium"),
    ("When was the United Nations founded?", "1945", "factual", "medium"),
    ("What did Tim Berners-Lee invent?", "World Wide Web", "factual", "medium"),
    ("What is encryption?", "converts readable text into coded text", "factual", "medium"),
    ("What is a database?", "organized collection of data stored electronically", "factual", "medium"),
]

_REASONING_QA = [
    ("If it rains and the ground is flat, what happens?", "puddles will form", "reasoning", "easy"),
    ("Whales are mammals. Do whales breathe air?", "yes", "reasoning", "easy"),
    ("Does correlation imply causation?", "no", "reasoning", "medium"),
    ("What is an analogy?", "help us understand new concepts by comparing them to familiar ones", "reasoning", "medium"),
    ("How should you solve a complex problem?", "break it into smaller parts", "reasoning", "medium"),
]

_MATH_QA = [
    ("What is 25 + 17?", "42", "math", "easy"),
    ("What is 144 / 12?", "12", "math", "easy"),
    ("What is 7 * 8?", "56", "math", "easy"),
    ("What is 100 - 37?", "63", "math", "easy"),
    ("What is 2 ** 10?", "1024", "math", "easy"),
    ("What is 15% of 200?", "30", "math", "medium"),
    ("What is 3/4 + 1/4?", "1", "math", "medium"),
    ("What is (10 + 5) * 3?", "45", "math", "medium"),
    ("What is the next number in the sequence 2, 4, 6, 8?", "10", "math", "easy"),
    ("Convert 100 celsius to fahrenheit", "212", "math", "medium"),
]

_CONVERSATION_QA = [
    ("Hello, how are you?", "fine", "conversation", "easy"),
    ("Thank you for your help!", "welcome", "conversation", "easy"),
    ("Can you explain something simply?", "start with simple concepts", "conversation", "medium"),
    ("What makes a good conversation?", "listening", "conversation", "medium"),
    ("How can I write more clearly?", "short sentences", "conversation", "medium"),
]

_SCIENCE_QA = [
    ("What is the water cycle?", "water evaporates from oceans, forms clouds, falls as rain", "science", "medium"),
    ("What are cells?", "basic units of life", "science", "easy"),
    ("What is evolution?", "species change over time", "science", "medium"),
    ("What does the electromagnetic spectrum include?", "radio waves", "science", "hard"),
    ("How does the heart work?", "beats approximately 100,000 times per day, pumping blood", "science", "medium"),
]

_TECHNOLOGY_QA = [
    ("What is a computer program?", "set of instructions", "technology", "easy"),
    ("What is cloud computing?", "servers and services accessed over the internet", "technology", "medium"),
    ("What is machine learning?", "computers learn patterns from data", "technology", "medium"),
]


def get_all_benchmarks() -> List[Tuple[str, str, str, str]]:
    """Return all built-in benchmark questions."""
    return (
        _FACTUAL_QA + _REASONING_QA + _MATH_QA +
        _CONVERSATION_QA + _SCIENCE_QA + _TECHNOLOGY_QA
    )


# ═══════════════════════════════════════════════════════════════════════════
# Main Evaluator
# ═══════════════════════════════════════════════════════════════════════════

class ModelEvaluator:
    """Evaluates the NSCK AI model on real-world benchmarks.

    Usage::

        evaluator = ModelEvaluator(engine)
        report = evaluator.run_full_evaluation()
        print(report.summary())
    """

    def __init__(self, engine):
        """
        Args:
            engine: An ``NSCKAIEngine`` instance (or compatible).
        """
        self.engine = engine
        self._math_handler = None

    def _get_math_handler(self):
        """Lazy import of MathHandler."""
        if self._math_handler is None:
            from nsck_ai_model.math_handler import MathHandler
            self._math_handler = MathHandler()
        return self._math_handler

    def evaluate_question(
        self, question: str, expected: str, category: str
    ) -> BenchmarkResult:
        """Evaluate a single question.

        Returns:
            BenchmarkResult with correctness, score, and latency.
        """
        start = time.time()

        # Route math questions to math handler
        if category == "math":
            try:
                from nsck_ai_model.math_handler import is_math_question
                if is_math_question(question):
                    math_handler = self._get_math_handler()
                    math_result = math_handler.solve(question)
                    actual = math_result.answer
                    latency = (time.time() - start) * 1000

                    correct, score = self._score_answer(
                        expected, actual, category)
                    return BenchmarkResult(
                        question=question, expected=expected,
                        actual=actual, correct=correct,
                        score=score, category=category,
                        latency_ms=latency,
                        details={"method": math_result.method,
                                 "steps": math_result.steps})
            except ImportError:
                pass

        # Use the engine for all other questions
        try:
            result = self.engine.chat(question, auto_learn=False)
            actual = result.get('response', '')
        except Exception as e:
            actual = f"ERROR: {e}"

        latency = (time.time() - start) * 1000
        correct, score = self._score_answer(expected, actual, category)

        return BenchmarkResult(
            question=question, expected=expected,
            actual=actual, correct=correct,
            score=score, category=category,
            latency_ms=latency)

    def _score_answer(
        self, expected: str, actual: str, category: str
    ) -> Tuple[bool, float]:
        """Score an answer against the expected.

        Returns:
            (is_correct, score_0_to_1)
        """
        if not actual or 'need more training data' in actual.lower():
            return False, 0.0

        expected_lower = expected.lower().strip()
        actual_lower = actual.lower().strip()

        # For math: check numeric equality
        if category == "math":
            try:
                exp_nums = re.findall(r'-?\d+\.?\d*', expected_lower)
                act_nums = re.findall(r'-?\d+\.?\d*', actual_lower)
                if exp_nums and act_nums:
                    exp_val = float(exp_nums[0])
                    act_val = float(act_nums[0])
                    if _numeric_close(exp_val, act_val, 0.01):
                        return True, 1.0
            except (ValueError, IndexError):
                pass

        # Check fuzzy containment
        if _fuzzy_contains(actual, expected, threshold=0.5):
            return True, 1.0

        # Check if expected is a short keyword and it appears in response
        exp_words = set(_tokenize(expected))
        act_words = set(_tokenize(actual))
        if exp_words and len(exp_words) <= 3:
            if exp_words <= act_words:
                return True, 0.9

        # Partial overlap scoring
        f1 = _f1_score(expected, actual)
        overlap = _overlap_score(expected, actual)
        score = max(f1, overlap)

        # Bonus if the key concept word is present
        key_words = [w for w in _tokenize(expected) if len(w) > 3]
        if key_words:
            present = sum(1 for w in key_words if w in actual_lower)
            key_ratio = present / len(key_words)
            score = max(score, key_ratio * 0.8)

        correct = score >= 0.5
        return correct, score

    def run_builtin_benchmarks(
        self,
        categories: Optional[List[str]] = None
    ) -> EvalReport:
        """Run all built-in benchmarks.

        Args:
            categories: Which categories to test. None = all.
        """
        benchmarks = get_all_benchmarks()
        if categories:
            benchmarks = [(q, a, c, d) for q, a, c, d in benchmarks
                          if c in categories]
        return self._run_benchmarks(benchmarks)

    def run_qa_evaluation(
        self,
        qa_pairs: List[Tuple[str, str]],
        category: str = "custom"
    ) -> EvalReport:
        """Evaluate on custom Q&A pairs."""
        benchmarks = [(q, a, category, "custom") for q, a in qa_pairs]
        return self._run_benchmarks(benchmarks)

    def run_full_evaluation(self) -> EvalReport:
        """Run the complete evaluation suite."""
        return self.run_builtin_benchmarks()

    def _run_benchmarks(
        self,
        benchmarks: List[Tuple[str, str, str, str]]
    ) -> EvalReport:
        """Run a set of benchmark questions."""
        start = time.time()
        results = []

        for question, expected, category, difficulty in benchmarks:
            result = self.evaluate_question(question, expected, category)
            result.details['difficulty'] = difficulty
            results.append(result)

        duration = time.time() - start

        # Aggregate
        total = len(results)
        correct = sum(1 for r in results if r.correct)
        overall_acc = correct / total if total > 0 else 0.0
        overall_score = np.mean([r.score for r in results]) if results else 0.0

        # Per-category
        categories = defaultdict(list)
        for r in results:
            categories[r.category].append(r)

        cat_scores = {}
        for cat, cat_results in categories.items():
            cat_correct = sum(1 for r in cat_results if r.correct)
            cat_total = len(cat_results)
            cat_scores[cat] = CategoryScore(
                category=cat,
                total=cat_total,
                correct=cat_correct,
                accuracy=cat_correct / cat_total if cat_total else 0.0,
                avg_score=float(np.mean([r.score for r in cat_results])),
                avg_latency_ms=float(np.mean(
                    [r.latency_ms for r in cat_results])),
            )

        # Get training stats
        try:
            training_stats = self.engine.get_system_stats()
        except Exception:
            training_stats = {}

        return EvalReport(
            timestamp=time.time(),
            total_questions=total,
            total_correct=correct,
            overall_accuracy=overall_acc,
            overall_avg_score=float(overall_score),
            categories=cat_scores,
            results=results,
            training_stats=training_stats,
            duration_s=duration,
        )

    def evaluate_response_quality(self, n_samples: int = 20) -> Dict[str, Any]:
        """Evaluate general response quality metrics.

        Tests:
        - Response length (not too short, not too long)
        - Coherence (does it make sense?)
        - Diversity (different responses to different questions)
        - Relevance (response relates to question)
        """
        questions = [q for q, _, _, _ in get_all_benchmarks()]
        sample = random.sample(questions, min(n_samples, len(questions)))

        responses = []
        for q in sample:
            try:
                result = self.engine.chat(q, auto_learn=False)
                responses.append((q, result.get('response', '')))
            except Exception:
                responses.append((q, ''))

        # Metrics
        lengths = [len(r.split()) for _, r in responses]
        non_empty = sum(1 for _, r in responses
                        if r and 'need more training data' not in r.lower())

        # Diversity: unique response ratio
        unique_responses = len(set(r for _, r in responses))
        diversity = unique_responses / len(responses) if responses else 0.0

        # Relevance: does response share words with question?
        relevance_scores = []
        for q, r in responses:
            if r and 'need more training data' not in r.lower():
                q_words = set(_tokenize(q))
                r_words = set(_tokenize(r))
                if q_words:
                    rel = len(q_words & r_words) / len(q_words)
                    relevance_scores.append(rel)

        return {
            'total_questions': len(sample),
            'non_empty_responses': non_empty,
            'response_rate': non_empty / len(sample) if sample else 0.0,
            'avg_response_length': float(np.mean(lengths)) if lengths else 0.0,
            'min_response_length': min(lengths) if lengths else 0,
            'max_response_length': max(lengths) if lengths else 0,
            'diversity': diversity,
            'avg_relevance': float(np.mean(relevance_scores))
            if relevance_scores else 0.0,
        }
