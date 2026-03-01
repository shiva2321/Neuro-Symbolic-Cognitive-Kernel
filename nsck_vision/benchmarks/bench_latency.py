"""Benchmark: end-to-end latency. Target: <500ms on CPU."""
from __future__ import annotations

import os
import sys
import json
import time
import logging
import statistics

os.environ["NSCK_USE_RUST"] = "1"

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_NSCK_DIR = os.path.join(_REPO_ROOT, "nsck")
if _NSCK_DIR not in sys.path:
    sys.path.insert(0, _NSCK_DIR)

import numpy as np

logger = logging.getLogger("nsck_vision.bench_latency")
logging.basicConfig(level=logging.INFO)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
TARGET_MS = 500.0


def run_latency_benchmark(n_warmup: int = 3, n_measure: int = 10) -> dict:
    """Measure end-to-end analysis latency."""
    from nsck_vision.system import NSCKVisionSystem
    
    system = NSCKVisionSystem()
    rng = np.random.default_rng(42)
    
    # Absorb a small model first
    dim = 64
    train_data = [(rng.standard_normal(dim).astype(np.float32), f"class_{i%5}") for i in range(20)]
    
    def identity_model(x): return x
    system.absorb(
        model_or_name=identity_model,
        domain="latency_test",
        dataset=train_data,
        max_samples=20,
        model_id="latency_bench_model",
    )
    
    test_images = [rng.standard_normal(dim).astype(np.float32) for _ in range(n_warmup + n_measure)]
    
    # Warmup
    for img in test_images[:n_warmup]:
        system.analyze(img)
    
    # Measure
    latencies = []
    for img in test_images[n_warmup:]:
        t0 = time.perf_counter()
        response = system.analyze(img)
        latencies.append((time.perf_counter() - t0) * 1000)
    
    mean_ms = statistics.mean(latencies) if latencies else 0.0
    p50_ms = statistics.median(latencies) if latencies else 0.0
    p95_ms = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0.0
    
    result = {
        "n_warmup": n_warmup,
        "n_measure": n_measure,
        "mean_ms": mean_ms,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "target_ms": TARGET_MS,
        "target_met": mean_ms < TARGET_MS,
        "latencies_ms": latencies,
    }
    return result


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    logger.info("Running latency benchmark (target: <%.0fms)...", TARGET_MS)
    results = run_latency_benchmark()
    logger.info("Mean latency: %.1fms (target: %.0fms, met=%s)", 
                results["mean_ms"], TARGET_MS, results["target_met"])
    
    out_path = os.path.join(RESULTS_DIR, "latency_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results written to %s", out_path)
    return results


if __name__ == "__main__":
    main()
