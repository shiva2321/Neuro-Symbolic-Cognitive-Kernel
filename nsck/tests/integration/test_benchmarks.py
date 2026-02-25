"""Integration tests for benchmark suite (V8)."""
import sys
import os
import pytest

# Ensure the nsck directory is first in sys.path so our benchmarks package
# takes priority over nsck/tests/benchmarks/ which would otherwise shadow it.
_nsck_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))
if not sys.path or sys.path[0] != _nsck_dir:
    sys.path.insert(0, _nsck_dir)

import importlib
_babi = importlib.import_module('benchmarks.babi_tasks')
_math = importlib.import_module('benchmarks.math_word_problems')
_transfer = importlib.import_module('benchmarks.cross_domain_transfer')
_runner_mod = importlib.import_module('benchmarks.runner')


class TestMathWordProblemsBenchmark:
    def test_run_math_benchmark_returns_float(self):
        score = _math.run_math_benchmark()
        assert isinstance(score, float)

    def test_run_math_benchmark_range(self):
        score = _math.run_math_benchmark()
        assert 0.0 <= score <= 100.0

    def test_math_problems_list_length(self):
        assert len(_math.MATH_PROBLEMS) == 50

    def test_math_score_positive(self):
        score = _math.run_math_benchmark()
        assert score > 0.0


class TestCrossDomainTransferBenchmark:
    def test_run_transfer_returns_float(self):
        score = _transfer.run_transfer_benchmark()
        assert isinstance(score, float)

    def test_run_transfer_range(self):
        score = _transfer.run_transfer_benchmark()
        assert 0.0 <= score <= 100.0

    def test_transfer_tasks_list(self):
        assert len(_transfer.TRANSFER_TASKS) >= 5


class TestBabiTasksBenchmark:
    def test_babi_tasks_list(self):
        assert len(_babi.BABI_TASKS) == 20

    def test_babi_task_format(self):
        story, question, answer = _babi.BABI_TASKS[0]
        assert isinstance(story, list)
        assert isinstance(question, str)
        assert isinstance(answer, str)


class TestBenchmarkRunner:
    def test_runner_instantiation(self):
        runner = _runner_mod.BenchmarkRunner()
        assert runner is not None

    def test_benchmark_names(self):
        assert "math_word_problems" in _runner_mod.BenchmarkRunner.BENCHMARK_NAMES
        assert "babi_tasks" in _runner_mod.BenchmarkRunner.BENCHMARK_NAMES

    def test_run_all_returns_dict(self):
        runner = _runner_mod.BenchmarkRunner()
        results = runner.run_all()
        assert isinstance(results, dict)

    def test_run_all_has_all_keys(self):
        runner = _runner_mod.BenchmarkRunner()
        results = runner.run_all()
        for name in _runner_mod.BenchmarkRunner.BENCHMARK_NAMES:
            assert name in results
