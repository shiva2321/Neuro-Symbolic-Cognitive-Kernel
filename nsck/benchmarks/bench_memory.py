"""Memory Query Latency Benchmark for NSCK V3."""
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


def _measure_query_latency(memory, n_queries: int = 20) -> float:
    import python.core.vsa.hypervec_shim as hypervec_rs
    query = hypervec_rs.HyperVector(42)
    latencies = []
    for _ in range(n_queries):
        t0 = time.perf_counter()
        memory.query(query, k=5)
        latencies.append(time.perf_counter() - t0)
    return sum(latencies) / len(latencies)


def run() -> Dict[str, Any]:
    import python.core.vsa.hypervec_shim as hypervec_rs
    from python.core.memory.semantic_memory import SemanticMemory

    results_by_size = {}
    for target_size in [100, 1000]:
        sem = SemanticMemory(use_rust=False)
        for i in range(target_size):
            concept = f"concept_{i}"
            sem.add_concept(concept, {"index": i})
        avg_latency = _measure_query_latency(sem)
        results_by_size[str(target_size)] = {
            "concepts": target_size,
            "avg_query_latency_ms": round(avg_latency * 1000, 4),
        }

    summary = {
        "benchmark": "memory",
        "results": results_by_size,
    }

    os.makedirs(os.path.join(_HERE, "results"), exist_ok=True)
    out_path = os.path.join(_HERE, "results", "bench_memory.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
