"""Decision Latency Benchmark for NSCK V3."""
from __future__ import annotations
import json
import os
import sys
import time
from typing import Dict, Any

_HERE = os.path.dirname(os.path.abspath(__file__))
_NSCK_ROOT = os.path.join(_HERE, "..")
if _NSCK_ROOT not in sys.path:
    sys.path.insert(0, _NSCK_ROOT)


def run(n_calls: int = 100) -> Dict[str, Any]:
    from python.core.reasoning.cognitive_engine import CognitiveEngine
    from python.core.integration.config import NSCKConfig

    config = NSCKConfig()
    engine = CognitiveEngine(config)

    latencies = []
    state = {"x": 1, "y": 2, "score": 0}

    for i in range(n_calls):
        state["score"] = i
        t0 = time.perf_counter()
        try:
            engine.decide(state, task="default")
        except Exception:
            pass
        latencies.append(time.perf_counter() - t0)

    latencies.sort()
    n = len(latencies)
    summary = {
        "benchmark": "decision",
        "n_calls": n,
        "avg_latency_ms": round(sum(latencies) / n * 1000, 3),
        "p50_latency_ms": round(latencies[n // 2] * 1000, 3),
        "p95_latency_ms": round(latencies[int(n * 0.95)] * 1000, 3),
        "p99_latency_ms": round(latencies[min(int(n * 0.99), n - 1)] * 1000, 3),
        "min_latency_ms": round(latencies[0] * 1000, 3),
        "max_latency_ms": round(latencies[-1] * 1000, 3),
    }

    os.makedirs(os.path.join(_HERE, "results"), exist_ok=True)
    out_path = os.path.join(_HERE, "results", "bench_decision.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
