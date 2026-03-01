"""Run all NSCK-UPMA benchmarks and produce combined summary."""
from __future__ import annotations

import os
import sys
import json
import time
import logging

os.environ["NSCK_USE_RUST"] = "1"

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_NSCK_DIR = os.path.join(_REPO_ROOT, "nsck")
if _NSCK_DIR not in sys.path:
    sys.path.insert(0, _NSCK_DIR)

logger = logging.getLogger("nsck_vision.run_all")
logging.basicConfig(level=logging.INFO)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary = {"timestamp": time.time(), "benchmarks": {}}
    
    from nsck_vision.benchmarks.bench_absorption import main as bench_absorption
    from nsck_vision.benchmarks.bench_fusion import main as bench_fusion
    from nsck_vision.benchmarks.bench_crossdomain import main as bench_crossdomain
    from nsck_vision.benchmarks.bench_latency import main as bench_latency
    
    benchmarks = [
        ("absorption", bench_absorption),
        ("fusion", bench_fusion),
        ("crossdomain", bench_crossdomain),
        ("latency", bench_latency),
    ]
    
    for name, fn in benchmarks:
        logger.info("=== Running %s benchmark ===", name)
        t0 = time.perf_counter()
        try:
            result = fn()
            summary["benchmarks"][name] = {"status": "passed", "result": result, "time_s": time.perf_counter() - t0}
        except Exception as e:
            logger.error("Benchmark %s failed: %s", name, e)
            summary["benchmarks"][name] = {"status": "failed", "error": str(e), "time_s": time.perf_counter() - t0}
    
    out_path = os.path.join(RESULTS_DIR, "benchmark_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("All benchmarks complete. Summary: %s", out_path)
    return summary


if __name__ == "__main__":
    main()
