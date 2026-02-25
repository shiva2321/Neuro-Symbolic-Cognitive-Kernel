"""BenchmarkRunner — runs all NSCK benchmarks and reports scores."""
from __future__ import annotations
import sys
import os
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class BenchmarkRunner:
    """Runs all NSCK evaluation benchmarks and returns a score dict."""

    BENCHMARK_NAMES = [
        "babi_tasks",
        "math_word_problems",
        "cross_domain_transfer",
        "nlg_quality",
        "dialogue_coherence",
    ]

    def __init__(self):
        self._results: Dict[str, float] = {}

    def run_all(self) -> Dict[str, float]:
        """Run all benchmarks and return {name: score_percent} dict."""
        try:
            from benchmarks import babi_tasks, math_word_problems
            from benchmarks import cross_domain_transfer, nlg_quality, dialogue_coherence
        except ImportError:
            from nsck.benchmarks import babi_tasks, math_word_problems
            from nsck.benchmarks import cross_domain_transfer, nlg_quality, dialogue_coherence

        self._results["babi_tasks"] = babi_tasks.run_babi_benchmark()
        self._results["math_word_problems"] = math_word_problems.run_math_benchmark()
        self._results["cross_domain_transfer"] = cross_domain_transfer.run_transfer_benchmark()
        self._results["nlg_quality"] = nlg_quality.run_nlg_benchmark()
        self._results["dialogue_coherence"] = dialogue_coherence.run_dialogue_benchmark()
        self._print_summary()
        return self._results

    def _print_summary(self):
        print("\n=== NSCK Benchmark Results ===")
        for name, score in self._results.items():
            bar = "█" * int(score / 5)
            print(f"  {name:<30s} {score:5.1f}%  {bar}")
        total = sum(self._results.values()) / len(self._results) if self._results else 0
        print(f"  {'AVERAGE':<30s} {total:5.1f}%")
        print("=" * 50)


if __name__ == "__main__":
    runner = BenchmarkRunner()
    runner.run_all()
