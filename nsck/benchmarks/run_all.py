"""Master benchmark runner for NSCK V3."""
from __future__ import annotations
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_NSCK_ROOT = os.path.join(_HERE, "..")
if _NSCK_ROOT not in sys.path:
    sys.path.insert(0, _NSCK_ROOT)

RESULTS_DIR = os.path.join(_HERE, "results")


def run_all():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = {}

    benchmarks = [
        ("nlu", "bench_nlu", "run"),
        ("memory", "bench_memory", "run"),
        ("snn", "bench_snn", "run"),
        ("decision", "bench_decision", "run"),
    ]

    for name, module_name, func_name in benchmarks:
        print(f"Running {name} benchmark...")
        t0 = time.perf_counter()
        try:
            import importlib
            mod = importlib.import_module(module_name)
            func = getattr(mod, func_name)
            result = func()
            results[name] = result
        except Exception as e:
            results[name] = {"error": str(e)}
        print(f"  {name}: {time.perf_counter() - t0:.2f}s")

    report_path = os.path.join(RESULTS_DIR, "benchmark_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    lines = ["NSCK V3 Benchmark Report", "=" * 40]
    for name, data in results.items():
        lines.append(f"\n[{name.upper()}]")
        if "error" in data:
            lines.append(f"  ERROR: {data['error']}")
        else:
            for k, v in data.items():
                if k not in ("details", "results"):
                    lines.append(f"  {k}: {v}")
    text = "\n".join(lines) + "\n"

    text_path = os.path.join(RESULTS_DIR, "capability_report.txt")
    with open(text_path, "w") as f:
        f.write(text)

    print(text)
    return results


if __name__ == "__main__":
    run_all()
