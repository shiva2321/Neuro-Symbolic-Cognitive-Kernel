"""SNN Perception Latency Benchmark for NSCK V3."""
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


def run(n_trials: int = 100) -> Dict[str, Any]:
    import numpy as np
    from python.core.perception.snn_perception import SNNPerceptionModule

    snn = SNNPerceptionModule()
    latencies = []
    spike_counts = []

    for i in range(n_trials):
        obs = np.random.rand(10).tolist()
        t0 = time.perf_counter()
        try:
            result = snn.perceive(obs)
            elapsed = time.perf_counter() - t0
            if hasattr(result, "spike_count"):
                spike_counts.append(result.spike_count)
            elif isinstance(result, dict) and "spike_count" in result:
                spike_counts.append(result["spike_count"])
        except Exception:
            elapsed = time.perf_counter() - t0
        latencies.append(elapsed)

    latencies.sort()
    n = len(latencies)
    summary = {
        "benchmark": "snn",
        "n_trials": n,
        "avg_latency_ms": round(sum(latencies) / n * 1000, 3),
        "p50_latency_ms": round(latencies[n // 2] * 1000, 3),
        "p95_latency_ms": round(latencies[int(n * 0.95)] * 1000, 3),
        "avg_spike_count": round(sum(spike_counts) / len(spike_counts), 2) if spike_counts else None,
        "spike_count_std": round(float(np.std(spike_counts)), 3) if spike_counts else None,
    }

    os.makedirs(os.path.join(_HERE, "results"), exist_ok=True)
    out_path = os.path.join(_HERE, "results", "bench_snn.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
