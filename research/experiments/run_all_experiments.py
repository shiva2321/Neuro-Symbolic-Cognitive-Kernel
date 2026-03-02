"""
Master Experiment Runner
========================
Run from: cd nsck && python ../research/experiments/run_all_experiments.py

Runs all 5 paper experiment scripts sequentially, collects all JSON results,
and prints a summary. Handles failures gracefully.
"""

import sys
import json
import time
import importlib.util
from pathlib import Path

_here = Path(__file__).resolve()
_nsck_dir = _here.parent.parent.parent / "nsck"
_exp_dir = _here.parent
if str(_nsck_dir) not in sys.path:
    sys.path.insert(0, str(_nsck_dir))

RESULTS_DIR = _here.parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SCRIPTS = [
    ("Paper 4 — SNN Benchmarks",           _exp_dir / "paper4_snn_benchmarks.py"),
    ("Paper 5 — Transplant Benchmarks",    _exp_dir / "paper5_transplant_benchmarks.py"),
    ("Paper 6 — Active Inference + GWT",   _exp_dir / "paper6_active_inference_benchmarks.py"),
    ("Paper 7 — Memory Benchmarks",        _exp_dir / "paper7_memory_benchmarks.py"),
    ("Paper 8 — Safety Benchmarks",        _exp_dir / "paper8_safety_benchmarks.py"),
    ("Paper 9 — Societal HV Benchmarks",   _exp_dir / "paper9_societal_benchmarks.py"),
]

RESULT_FILES = [
    RESULTS_DIR / "paper4_results.json",
    RESULTS_DIR / "paper5_results.json",
    RESULTS_DIR / "paper6_results.json",
    RESULTS_DIR / "paper7_results.json",
    RESULTS_DIR / "paper8_results.json",
]


def run_script(name: str, script_path: Path) -> bool:
    """Run a script by importing and executing it, return True on success."""
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"Script:  {script_path}")
    print(f"{'='*60}")
    t0 = time.perf_counter()
    try:
        spec = importlib.util.spec_from_file_location("_exp_module", script_path)
        mod = importlib.util.module_from_spec(spec)
        # Make __name__ == "__main__" to trigger the if __name__ == "__main__" block
        mod.__name__ = "__main__"
        spec.loader.exec_module(mod)
        elapsed = time.perf_counter() - t0
        print(f"\n✓ {name} completed in {elapsed:.1f}s")
        return True
    except SystemExit as e:
        print(f"\n✗ {name} exited with code {e.code}")
        return False
    except Exception as e:
        import traceback
        elapsed = time.perf_counter() - t0
        print(f"\n✗ {name} FAILED after {elapsed:.1f}s: {e}")
        traceback.print_exc()
        return False


def print_summary():
    """Load all result files and print a compact summary."""
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")

    for result_file in RESULT_FILES:
        paper_name = result_file.stem
        if not result_file.exists():
            print(f"\n[{paper_name}] NOT FOUND (experiment may have failed)")
            continue
        with open(result_file) as f:
            data = json.load(f)

        print(f"\n[{paper_name}]")
        backend = data.get("backend", {})
        print(f"  Backend: VSA={'Rust' if backend.get('rust_vsa') else 'Python'}  SNN={'Rust' if backend.get('rust_snn') else 'Python'}")

        for key, val in data.items():
            if key == "backend":
                continue
            if isinstance(val, dict) and "error" in val:
                print(f"  {key}: ERROR — {val['error']}")
            else:
                print(f"  {key}: OK")


if __name__ == "__main__":
    successes = []
    for name, script_path in SCRIPTS:
        ok = run_script(name, script_path)
        successes.append((name, ok))

    print_summary()

    print(f"\n{'='*60}")
    print("EXPERIMENT RUN STATUS")
    print(f"{'='*60}")
    for name, ok in successes:
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {status}  {name}")

    n_pass = sum(1 for _, ok in successes if ok)
    print(f"\n{n_pass}/{len(SCRIPTS)} experiments completed successfully.")
    print(f"Results written to: {RESULTS_DIR}/")
