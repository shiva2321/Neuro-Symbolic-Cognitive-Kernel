"""NSCK Evaluation Suite (NSCK-ES) — V1.0.

Composite score across 5 cognitive tasks.
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional


class NSCKEvalSuite:
    """NSCK Evaluation Suite runner.

    .. code-block:: python

        suite = NSCKEvalSuite()
        results = suite.run_all()
        print(f"NSCK-ES: {results['nsck_es']:.3f}")
    """

    VERSION = "1.0"
    WEIGHTS: Dict[str, float] = {
        "t1": 0.30,
        "t2": 0.20,
        "t3": 0.20,
        "t4": 0.15,
        "t5": 0.15,
    }

    def __init__(self, engine: Optional[Any] = None) -> None:
        self._engine = engine

    def _make_engine(self) -> Any:
        if self._engine is not None:
            return self._engine
        from python.core.reasoning.cognitive_engine import CognitiveEngine
        from python.core.integration.config import NSCKConfig
        return CognitiveEngine(config=NSCKConfig(), persistence_path=":memory:")

    def run_all(self) -> Dict[str, float]:
        """Run all tasks and return scores dict including 'nsck_es' composite."""
        from eval.tasks import t1_semantic_qa, t2_generalization, t3_lifelong
        from eval.tasks import t4_cross_modal, t5_causal_reasoning

        engine = self._make_engine()

        results: Dict[str, float] = {}
        start = time.time()

        task_runners = [
            ("t1", t1_semantic_qa.run),
            ("t2", t2_generalization.run),
            ("t3", t3_lifelong.run),
            ("t4", t4_cross_modal.run),
            ("t5", t5_causal_reasoning.run),
        ]

        for key, runner in task_runners:
            try:
                score = float(runner(engine))
                # Clamp to [0, 1]
                results[key] = max(0.0, min(1.0, score))
            except Exception:
                results[key] = 0.0

        # Composite weighted score
        nsck_es = sum(self.WEIGHTS[k] * results.get(k, 0.0) for k in self.WEIGHTS)
        results["nsck_es"] = float(nsck_es)
        results["version"] = self.VERSION  # type: ignore[assignment]
        results["elapsed_s"] = time.time() - start

        return results

    def save_report(self, path: str) -> None:
        """Save evaluation report to JSON file."""
        results = self.run_all()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
